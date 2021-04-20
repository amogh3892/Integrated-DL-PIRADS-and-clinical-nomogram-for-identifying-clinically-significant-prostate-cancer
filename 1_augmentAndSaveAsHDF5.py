import SimpleITK as sitk 
from pathlib import Path
import sys 
sys.path.append(fr"../Code_general")
from dataUtil import DataUtil 
import tables
import os 
import numpy as np
from augmentation3DUtil import Augmentation3DUtil
from augmentation3DUtil import Transforms
from skimage.measure import regionprops
from skimage.transform import resize as skresize
import pandas as pd 


def _getAugmentedData(imgs,masks,nosamples):
    
    """ 
    This function defines different augmentations/transofrmation sepcified for a single image 
    imgs,masks : to be provided SimpleITK images 
    nosamples : (int) number of augmented samples to be returned
    """
    au = Augmentation3DUtil(imgs,masks=masks)

    au.add(Transforms.SHEAR,probability = 0.3, magnitude = (0.05,0.05))
    au.add(Transforms.SHEAR,probability = 0.3, magnitude = (0.02,0.05))
    au.add(Transforms.SHEAR,probability = 0.3, magnitude = (0.01,0.05))
    au.add(Transforms.SHEAR,probability = 0.3, magnitude = (0.03,0.05))

    au.add(Transforms.ROTATE2D,probability = 0.4, degrees = 1)
    au.add(Transforms.ROTATE2D,probability = 0.4, degrees = -1)
    au.add(Transforms.ROTATE2D,probability = 0.4, degrees = 2)
    au.add(Transforms.ROTATE2D,probability = 0.4, degrees = -2)

    au.add(Transforms.FLIPHORIZONTAL,probability = 0.5)

    imgs, augs = au.process(nosamples)

    return imgs,augs



def createHDF5(hdf5path,splitsdict,patchSize,depth):
    
    """
    hdf5path: path to where hdf5 file has to be saved
    splitsdict: A dictionary of the training/testing splits with key as name 
                of the case and value as whether being 'train', 'val' or 'test'
    patchSize: Size of 2D patches as a tuple; ex (256,256)
    depth: 1 for 2D and depth value for 3D 
    """
    
    outputfolder = fr"outputs/hdf5/{hdf5path}"
    Path(outputfolder).mkdir(parents=True, exist_ok=True)

    img_dtype = tables.Float32Atom()

    if depth > 1:
        data_shape = (0, depth,4, patchSize[0], patchSize[1])
        data_chuck_shape = (1, depth,4,patchSize[0],patchSize[1])

    else:
        data_shape = (0, 3, patchSize[0], patchSize[1])
        data_chuck_shape = (1,3,patchSize[0],patchSize[1])


    filters = tables.Filters(complevel=5)

    splitsdict = DataUtil.readJson(splitspath)

    phases = np.unique(list(splitsdict.values()))

    for phase in phases:
        hdf5_path = fr'{outputfolder}/{phase}.h5'

        if os.path.exists(hdf5_path):
            Path(hdf5_path).unlink()

        hdf5_file = tables.open_file(hdf5_path, mode='w')


        data = hdf5_file.create_earray(hdf5_file.root, "data", img_dtype,
                                            shape=data_shape,
                                            chunkshape = data_chuck_shape,
                                            filters = filters)


        hdf5_file.close()


def _addToHDF5(sample,phase,splitspathname):
    
    """
    sample : Data as 3D array with n channels of 2D patches
    phase : phase of that image (train,test,val)
    splitspathname : name of the file (json) which has train test splits info 
    """
    outputfolder = fr"outputs/hdf5/{splitspathname}"

    hdf5_file = tables.open_file(fr'{outputfolder}/{phase}.h5', mode='a')

    data = hdf5_file.root["data"]

    data.append(sample[None])
    
    hdf5_file.close()


def getAugmentedData(folderpath,lesion, nosamples = None):
    
    """
    folderpath : path to folder containing images, mask
    lesion: Lesion number indicating the lesion mask for 
            which augmentation has to be performed
    nosamples : Number of augmentations to be performed
    """
    folderpath = Path(folderpath)

    try:
        ext = folderpath.glob(fr"T2W_std*").__next__().suffix
    except:
        import pdb 
        pdb.set_trace()

    if ext == ".gz":
        ext = ".".join(glob(fr"{folderpath}/**")[0].split("/")[-1].split(".")[-2:])
    else:
        ext = ext.replace(".","")


    t2w = sitk.ReadImage(str(folderpath.joinpath(fr"T2W_std.{ext}")))
    adc = sitk.ReadImage(str(folderpath.joinpath(fr"ADC_reg.{ext}")))

    imgs = [t2w,adc] 

    pm = sitk.ReadImage(str(folderpath.joinpath(fr"PM.{ext}")))
    pm = DataUtil.convert2binary(pm)

    ls = sitk.ReadImage(str(folderpath.joinpath(fr"LS{lesion}.{ext}")))
    ls = DataUtil.convert2binary(ls)

    masks = [pm,ls]

    ret = []
    
    orgimg,augs = _getAugmentedData(imgs,masks,nosamples)
    ret.append((orgimg))

    if augs is not None:
        for i in range(len(augs)):
            ret.append(augs[i])

    return ret

def normalizeImage(img,_min,_max,clipValue=None):

    """
    img: img as SimpleITK image object
    _min: Min value for normalization
    _max: Max value for normalization
    clipValue: clipping the image to a max value.
    """

    imgarr = sitk.GetArrayFromImage(img)

    if clipValue is not None:
        imgarr[imgarr > clipValue] = clipValue 

    imgarr[imgarr < _min] = _min
    imgarr[imgarr > _max] = _max

    imgarr = (imgarr - _min)/(_max - _min)

    imgarr = imgarr.astype(np.float32)

    return imgarr



def addToHDF5(t2w,adc,pm,ls,phase,splitspathname,patchSize,t2wmin,t2wmax,adcmin,adcmax,name,label,dilate=None):
    
    """ 
    Collect samples from the cropped volume and add them into HDF5 file 

    t2w : T2W MRI as SimpleITK image object 
    ADC : ADC map as SimpleITK image object 
    pm : prostate mask as SimpleITK image object
    ls : lesion segmentation mask as SimpleITK image object
    phase: phase of that image (train,test,val)
    splitspathname : name of the splits files (json file of split)
    patchSize: Size of 2D patches as a tuple; ex (256,256)
    t2wmin : min value of T2W for normalization
    t2wmax : max value of T2W for normalization
    adcmin : min value of ADC for normalization
    adcmax : max value of ADC for normalization 
    label: target label (as being ciPCa or csPCa)
    dilate: in mm to expand the scale of patches using binary dilation. 
    """

    names = [] 
    labels = []

    # Get patches based on scale defined for patches 
    # Calculate dilate pixels based on mm to dilate and spacing information. 
    if dilate is None:
        lsperi = ls 
    else:
        spacing = t2w.GetSpacing()
        dilate = int(dilate/spacing[0])
        lsperi = sitk.BinaryDilate(ls,(dilate,dilate,0),sitk.sitkBall)

    # normalizing t2w mri and adc maps
    t2warr = normalizeImage(t2w,t2wmin,t2wmax,clipValue=t2wmax)
    adcarr = normalizeImage(adc,adcmin,adcmax,clipValue=3000)

    # converting SimpleITK image object to numpy array
    lsarr = sitk.GetArrayFromImage(ls)
    lsperiarr = sitk.GetArrayFromImage(lsperi)
    pmarr = sitk.GetArrayFromImage(pm)

    # obtain slice nos where the lesion is present
    slnos = np.unique(np.nonzero(lsarr)[0])

    samples = None 

    for i in slnos:

        sample = np.zeros((3,patchSize[1],patchSize[0]))

        # obain a particular slice where the lesion is present 
        slc = lsperiarr[i]
        
        # obtain the bounding box of the lesion at that slice. 
        props = regionprops(slc)

        if not slc.sum() == 0:
            try:
                starty, startx, endy, endx = props[0].bbox 
            except:
                import pdb 
                pdb.set_trace()

            # extract patches based on the bounding box obtained.
            t2wsample = t2warr[i, starty:endy, startx:endx]
            adcsample = adcarr[i, starty:endy, startx:endx]
            mask  = lsarr[i, starty:endy, startx:endx]

            orgmask = slc[starty:endy, startx:endx]

            # resize the patches to defined patchSize 
            sample[0] = skresize(t2wsample,(patchSize[1],patchSize[0]),order=1)
            sample[1] = skresize(adcsample,(patchSize[1],patchSize[0]),order=1)
            sample[2] = sitk.GetArrayFromImage(DataUtil.resampleimagebysize(sitk.GetImageFromArray(mask),(patchSize[1],patchSize[0]),sitk.sitkNearestNeighbor))

            # add the patches to the hdf5 file
            _addToHDF5(sample,phase,splitspathname)

            names.append(fr"{name}_{i}")
            labels.append(label)


    return names,labels 


def _getminmax(templatefolder,modality):
    img = sitk.ReadImage(fr"{templatefolder}/{modality}.nii")
    pm = sitk.ReadImage(fr"{templatefolder}/PM.nii")

    pm = DataUtil.convert2binary(pm)

    imgarr = sitk.GetArrayFromImage(img)
    pmarr = sitk.GetArrayFromImage(pm)

    maskedarr = imgarr*pmarr
    _min = maskedarr.min()
    _max = maskedarr.max()   

    return _min,_max  


if __name__ == "__main__":

    # read csv file which contains the lesion name 
    # and the corresponding labels as two separate columns 
    # Here lesion name as 'Lesion' column and label as 'Sig'
    labelsdf = pd.read_csv(fr"**path to csv file**")
    labelsdict = labelsdf.set_index("Lesion")["Sig"].to_dict()

    # Define the number of splits for cross-validation. 
    cvsplits = 3

    # Name 
    regions = ["intra","peri3","peri6","peri9","peri12"]
    region_dilations = [None, 3, 6, 9, 12]

    for k,dilate in enumerate(region_dilations):

        for cv in range(cvsplits):

            # Name of splits path which contains information of 
            # whether a lesion belongs to 'train', 'val' or 'test' split 
            splitspathname = fr"cspca_{cv}"

            inputfoldername = fr"1_Original_Organized"
            
            # patchSize in 2D
            newsize2D = (224,224) 

            # depth = 1 for 2D and depth value for 3D
            depth = 1
            
            splitspath = fr"outputs/splits/{splitspathname}.json"
            splitsdict = DataUtil.readJson(splitspath)

            cases = list(splitsdict.keys())
 
            hd5fpath = fr"{regions[k]}/{splitspathname}"

            createHDF5(hd5fpath,splitsdict,newsize2D,depth)
            
            casenames = {} 
            casenames["train"] = [] 
            casenames["val"] = []
            casenames["test"] = [] 

            caselabels = {} 
            caselabels["train"] = [] 
            caselabels["val"] = [] 
            caselabels["test"] = [] 

            # read template image that was used for intensity standardization
            # to obtain min and max of the template image to normalize images to (0,1) 
            templateimg = sitk.ReadImage(fr"../Data/Template/T2W.nii.gz")
            templatepm = sitk.ReadImage(fr"../Data/Template/PM.nii.gz")
            templatepm = DataUtil.convert2binary(templatepm)
            masked = sitk.Mask(templateimg,templatepm)
            maskedarr = sitk.GetArrayFromImage(masked)
            
            t2wmin = np.min(maskedarr)
            t2wmax = np.max(maskedarr)

            # min and max value for adc maps. 
            adcmin = 0 
            adcmax = 3000

            for j,name in enumerate(cases):

                dataset,pat = name.split("_")
                sb = Path(fr"../Data/{dataset}/{inputfoldername}/{name}")

                pat = int(pat)

                name = sb.stem
                print(name,cv,float(j)/len(cases))

                lsfiles = sb.glob(fr"LS*")

                for lsfile in lsfiles:
                    
                    lsfile = str(lsfile)
                    lesion = lsfile.split("/")[-1].split(".")[0][-1]

                    if fr"{name}_L{lesion}" in labelsdict:
                        label = labelsdict[fr"{name}_L{lesion}"]
                    else:
                        label = labelsdict[fr"{name}_L{lesion}".replace(name,mapdct[name])]

                    # ratio of augmentation to balance postive and negative classes. 
                    nosamples = 5 if label == 1 else 2
                    
                    phase = splitsdict[name]

                    # augmenting image if present in training set. 
                    if phase == "train":
                        ret = getAugmentedData(sb,lesion,nosamples=nosamples)
                    else:
                        ret = getAugmentedData(sb,lesion,nosamples=None)

                    for k,aug in enumerate(ret):
                    
                        augt2w = aug[0][0]
                        augadc = aug[0][1]
                        augpm = aug[1][0]
                        augls = aug[1][1]
                        
                        casename = fr"{name}_L{lesion}" if k == 0 else fr"{name}_L{lesion}_A{k}"

                        # adding original and augmented images to hdf5 file. 
                        names,labels = addToHDF5(augt2w,augadc,augpm,augls,phase,hd5fpath,newsize2D,t2wmin,t2wmax,adcmin,adcmax,casename,label,dilate=dilate)

                        casenames[phase].extend(names)
                        caselabels[phase].extend(labels)

            outputfolder = fr"outputs/hdf5/{hd5fpath}"

            for phase in ["train","val","test"]:
                hdf5_file = tables.open_file(fr'{outputfolder}/{phase}.h5', mode='a')
                hdf5_file.create_array(hdf5_file.root, fr'names', casenames[phase])
                hdf5_file.create_array(hdf5_file.root, fr'labels', caselabels[phase])

                hdf5_file.close()
