rm(list=ls())

library(pROC)
library(rms)
library(data.table)


# loading training and testing csvs for building the nomogram. The csv should contain vairables used in the nomogram. 
nomotrain = read.csv(file = '**path to training csv**')
nomotest = read.csv(file = '**path to testing csv**')

# renaming a few columns if required
colnames(nomotrain)[2] <- "DIP"
colnames(nomotest)[2] <- "DIP"

ddist <- datadist(nomotrain); options(datadist='ddist')

# Training a logistic regression classifier
fit1 <- lrm(True ~ DIP+PIRADS+ProstateVolume, data=nomotrain,penalty = 1)
nomopred1 = predict(fit1,nomotest,type="fitted.ind")

# Predictions on train 
nomotrainpred = predict(fit1,nomotrain,type="fitted.ind")

# Predictions on test and evaluating AUC 
roc1 <- roc(nomotest$True, nomopred1)
auc(roc1)

# Plotting the nomogram. 
nom <- nomogram(fit1, fun=function(x)1/(1+exp(-x)),fun.at=c(.001,.01,.05,seq(.1,.9,by=.1),.95,.99,.999),funlabel="Predictived Value")
plot(nom, xfrac=.22)

# Saving predictions for further analysis. 
dftrain <- data.frame(nomotrain$Case,nomotrain$True,nomotrainpred)
dftest <- data.frame(nomotest$Case,nomotest$True,nomopred1)

write.csv(dftrain,'/Volumes/GoogleDrive/My Drive/Projects/Code_csPCa_attention_2D/csvsv2/cladtrain.csv', row.names = FALSE)
write.csv(dftest,'/Volumes/GoogleDrive/My Drive/Projects/Code_csPCa_attention_2D/csvsv2/cladtest.csv', row.names = FALSE)