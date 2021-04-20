import torch
from torch import nn
from torchvision import transforms
import h5py
from torch.autograd import Variable
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from pytorchtools import EarlyStopping
from sklearn.metrics import roc_auc_score
import pandas as pd 
import tables 
torch.manual_seed(1)


# Defining AlexNet; adding a multiple instance learning module to 
# pool representations of multiple patches of the lesion
class AlexNet(nn.Module):

    def __init__(self, num_classes=2):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
        )

        self.attention = nn.Sequential(
            nn.Linear(4096, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

        self.final = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def max_pool(self,x):
        maxs, inds = torch.max(x,0)
        maxs = maxs[None,:]
        return maxs  

    def forward(self, x):
        A = None
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        feat = self.classifier(x)

        # M = self.max_pool(feat)
        A = self.attention(feat)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N
        M = torch.mm(A, feat)  # 1 x N 

        Y_prob = self.final(M)

        return Y_prob,feat,A

# Defining the torch Dataset 
class ProstateDatasetHDF5(Dataset):

    def __init__(self, fname,transforms = None):
        self.fname=fname
        self.file = tables.open_file(fname)
        self.tables = self.file.root
        self.nitems=self.tables.data.shape[0]
        
        self.file.close()
        self.data = None
        self.mask = None
        self.names = None
        self.labels = None 
        self.transforms = transforms
         
    def __getitem__(self, index):
                
        self.file = tables.open_file(self.fname)
        self.tables = self.file.root
        self.data = self.tables.data
        self.labels = self.tables.labels
                
        if "names" in self.tables:
            self.names = self.tables.names

        data = self.data[index,:,:,:]
        img = data[:,(0,1,2),:,:]
        mask = data[:,2,:,:]
                
    
        slices = np.unique(np.nonzero(mask)[0])
        
        if slices.size != 0:
            img = img[slices]
        else:
            slices = np.unique(np.nonzero(img)[0])
            img = img[slices]
                
        mask[mask > 1] = 1 

        if self.names is not None:
            name = self.names[index]
            
        label = self.labels[index]
        self.file.close()
            
        
        out = torch.from_numpy(img)
        
        return out,label,name

    def __len__(self):
        return self.nitems


def get_data(splitspathname,batch_size):

    """
    Obtaining the torch dataloader
    splitspathname: path to splits dictionary, 
        key: lesion name, value: phase (train, val or test)
    batch_size: The batch size for training the network
    
    """

    trainfilename = fr"**path to train.h5 hdf5 file**"
    valfilename = fr"**path to val.h5 hdf5 file**"
    testfilename = fr"**path to test.h5 hdf5 file**"

    train = h5py.File(trainfilename,libver='latest',mode='r')
    val = h5py.File(valfilename,libver='latest',mode='r')
    test = h5py.File(testfilename,libver='latest',mode='r')

    trainlabels = np.array(train["labels"])
    vallabels = np.array(val["labels"])
    testlabels = np.array(test["labels"])

    train.close()
    val.close()
    test.close()

    data_train = ProstateDatasetHDF5(trainfilename)
    data_val = ProstateDatasetHDF5(valfilename)
    data_test = ProstateDatasetHDF5(testfilename)

    num_workers = 8

    trainLoader = torch.utils.data.DataLoader(dataset=data_train,batch_size = batch_size,num_workers = num_workers,shuffle = True)
    valLoader = torch.utils.data.DataLoader(dataset=data_val,batch_size = batch_size,num_workers = num_workers,shuffle = False) 
    testLoader = torch.utils.data.DataLoader(dataset=data_test,batch_size = batch_size,num_workers = num_workers,shuffle = False) 

    dataLoader = {}
    dataLoader['train'] = trainLoader
    dataLoader['val'] = valLoader
    dataLoader['test'] = testLoader

    dataLabels = {}
    dataLabels["train"] = trainlabels
    dataLabels["val"] = vallabels
    dataLabels["test"] = testlabels

    return dataLoader, dataLabels

def get_model(modelname,device):
    """
    Get torch model object.
    Modify accordingly for other architectures. 
    
    modelname: name of the model 
    device: GPU device number
    """
    if modelname == "AlexNet":
        model = AlexNet()
    model.to(device)
    
    return model 

def run(model,modelname,device,num_epochs, learning_rate, weightdecay,batch_size, patience,dataLabels,cv):

    """
    model: The model object (AlexNet, DenseNet....)
    modelname: Name of the model for purpose of 
        creating an output directory of that name
    device: GPU device ID
    num_epochs: Maximum number of epochs the network has to train
                (Note: An early stopping criteria stops the 
                network before max epochs defined based on the 
                patience defined for early stopping in terms of 
                validation loss) 

    learning_rate: Learning rate hyper-parameter of network training 
    weightdecay: A hyper-parameter for Adam optimizer for regularization
    batch_size:The batch size for training the network
    patience: The patience set for early stopping criteria in 
            terms of validation loss

    dataLabels: Targer labels as a dictionary; 
                list of labels for 'train', 'val' and 'test'

    cv: The cross-validation split no. 

    """


    trainlabels = dataLabels["train"]

    zeros = (trainlabels == 0).sum()
    ones = (trainlabels == 1).sum()

    weights = [0,0]

    # Determining class weights to account for class imbalance.
    if zeros > ones:
        weights[0] = 1
        weights[1] = float(zeros)/ones
    else:

        weights[1] = 1
        weights[0] = float(ones)/zeros
        
    class_weights = torch.FloatTensor(weights).cuda(device)

    # Defining loss function and optimizer
    criterion=nn.CrossEntropyLoss(weight = class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weightdecay)


    # Defining early stopping criteria
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    modelname = fr"{modelname}_{cv}"

    niter_total=len(dataLoader['train'].dataset)/batch_size

    display = ["val","test"]

    results = {} 

    results["patience"] = patience

    early_stopping = EarlyStopping(patience=patience, verbose=True)

    parentfolder = r"Data/"

    # looping through epochs
    for epoch in range(num_epochs):

        pred_df_dict = {} 
        results_dict = {} 
        
        
        for phase in ["train","val"]:


            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            
            confusion_matrix=np.zeros((2,2))
            
            loss_vector=[]
            ytrue = [] 
            ypred = [] 
            ynames = [] 
            features = None 

            # looping through batches 
            for ii,(data,label,name) in enumerate(dataLoader[phase]):


                data = data[0]

                label=label.long().to(device)
                            
                data = Variable(data.float().cuda(device))

                with torch.set_grad_enabled(phase == 'train'):

                    # Obtaining network output
                    output,feat,att = model(data)

                    # Extracting deep learning representations
                    feat = feat.detach().data.cpu().numpy()
                    features = feat if features is None else np.vstack((features,feat))
                    
                    try:

                        _,pred_label=torch.max(output,1)

                    except:
                        import pdb 
                        pdb.set_trace()

                    # Calculating loss
                    loss = criterion(output, label)
                    probs = F.softmax(output,dim = 1)

                    probs = probs[:,1]

                    loss_vector.append(loss.detach().data.cpu().numpy())

                    # Backpropagation 
                    if phase=="train":
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()        

                    ypred.extend(probs.cpu().data.numpy().tolist())
                    ytrue.extend(label.cpu().data.numpy().tolist())
                    ynames.extend(list(name))

                    pred_label=pred_label.cpu()
                    label=label.cpu()
                    for p,l in zip(pred_label,label):
                        confusion_matrix[p,l]+=1
                    
                    torch.cuda.empty_cache()
            
            
            print(att)
            torch.cuda.empty_cache()
            total=confusion_matrix.sum()        
            acc=confusion_matrix.trace()/total
            loss_avg=np.mean(loss_vector)
            auc = roc_auc_score(ytrue,ypred)

            # Saving representations and predictions in a pandas dataframe. 
            pred_df = pd.DataFrame(np.column_stack((ynames,ytrue,ypred,[phase]*len(ynames))), columns = ["FileName","True", "Pred","Phase"])
            pred_df_dict[phase] = pred_df

            columns = ["FileName","True", "Pred","Phase"]

            for fno in range(features.shape[1]):
                columns.append(fr"feat_{fno}")
                
            pred_df_dict[phase] = pred_df

            results_dict[phase] = {} 
            results_dict[phase]["loss"] = loss_avg
            results_dict[phase]["auc"] = auc 
            results_dict[phase]["acc"] = acc 

            if phase == 'train':
                print("Epoch : {}, Phase : {}, Loss : {}, Acc: {}, Auc : {}".format(epoch,phase,loss_avg,acc,auc))
            elif phase in display:
                print("                 Epoch : {}, Phase : {}, Loss : {}, Acc: {}, Auc : {}".format(epoch,phase,loss_avg,acc,auc))
                
            for cl in range(confusion_matrix.shape[0]):
                cl_tp=confusion_matrix[cl,cl]/confusion_matrix[:,cl].sum()

            if phase == 'val':
                df = pred_df_dict["val"].append(pred_df_dict["train"], ignore_index=True)

                # Applying the earlystopping criteria
                early_stopping(loss_avg, model, modelname, df, results_dict,parentfolder =None)

            if early_stopping.early_stop:
                print("Early stopping")
                break

        if early_stopping.early_stop:
            break

if __name__ == "__main__":
    
    device = torch.device("cuda:0")
    batch_size = 1
    
    # Setting the hyper-parameters for network training
    num_epochs = 200

    learning_rate = 1e-6
    weightdecay = 1e-4

    patience = 10

    dataset = "**datasetname**"

    # Scale S0, S1, S2, S3, S4 based on addition of peritumoral region. 
    regions = ["**scalename**"]

    # looping through multiple scales and cross validation splits. 
    for region in regions:
        for cv in range(3):
            splitspathname = fr"{region}/{dataset}_{cv}"

            dataLoader, dataLabels = get_data(splitspathname,batch_size)

            modelname = fr"{region}_{dataset}"

            print(modelname,cv)
        
            model = get_model(modelname,device)
            run(model,modelname,device,num_epochs, learning_rate, weightdecay,batch_size,patience,dataLabels,cv)

