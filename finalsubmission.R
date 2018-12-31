rm(list = ls())
library(MLmetrics)
library(corrgram)
library(DMwR)
Train=read.csv("Train_data.csv")
Test=read.csv("Test_data.csv")
Train$ID=1
Test$ID=2
Churn_data=rbind(Train,Test)
Churn_data_2=Churn_data
#========================================================================
#FUN=the function to be applied to each element of x
#========================================================================
lapply(Churn_data, FUN =function(x)length(unique(x)))
lapply(Churn_data, FUN=function(x)sum(is.na(x)))
#===============================================================================================================
#pre-processing Because phone number is not a ssignificant metric in deciding the churn rate
#only voice mail and international plan are the two variables which are infactor form
#===============================================================================================================
Churn_data$Churn=factor(x=Churn_data$Churn,labels = 0:(length(levels(Churn_data$Churn))-1))
Churn_data$voice.mail.plan=factor(x=Churn_data$voice.mail.plan,labels = 0:(length(levels(Churn_data$voice.mail.plan))-1))
Churn_data$international.plan=factor(x=Churn_data$international.plan,labels = 0:(length(levels(Churn_data$international.plan))-1))
Churn_data$phone.number=NULL


library(ggplot2)

ggplot(Churn_data,aes(x=Churn_data$total.day.minutes))+
  geom_histogram(binwidth = 1,fill="white",colour="black")+
  xlab("Total day minutes")+
  ylab("Total day calls")

ggplot(Churn_data,aes(x=Churn_data$total.eve.minutes))+
  geom_histogram(binwidth = 1,fill="white",colour="black")+
  xlab("Total eve minutes")+
  ylab("Total eve calls")


plot.new()
plot(Churn_data$Churn~Churn_data$total.day.minutes)
plot(Churn_data$Churn~Churn_data$total.eve.minutes)
plot(Churn_data$Churn~Churn_data$total.night.minutes)
plot(Churn_data$Churn~Churn_data$total.intl.minutes)


par(mar=c(5,5,5,5))
corrgram(Churn_data,cex.labels = 1)
Churn_data$total.day.minutes=NULL
Churn_data$total.eve.minutes=NULL
Churn_data$total.night.minutes=NULL
Churn_data$total.intl.minutes=NULL

str(Churn_data)
#==============================================================================================================
#Feature Selection
#==============================================================================================================
library(Boruta)
#include all predictors except ID
churn_Boruta=Boruta(Churn ~ .-ID,data=Churn_data,doTrace=1)
par(mar=c(4,3,4,3))
plot(churn_Boruta,las=2)
Boruta_all=attStats(churn_Boruta)
Boruta_all=Boruta_all[Boruta_all$decision=='Confirmed',]
selected_cols=rownames(Boruta_all)
selected_cols

#=========================================================================================
#creating a new training data
#=========================================================================================
Churn_data=Churn_data[,c(selected_cols,"ID")]
Churn_data$Churn=Churn_data_2$Churn
Churn_data$Churn=factor(x=Churn_data$Churn,labels=0:(length(levels(Churn_data$Churn))-1))

#===========================================================================================
#OUTLIER ANALYSIS
#==========================================================================================
num_cols=lapply(Churn_data,FUN = function(x)is.numeric(x))
sum=0
for (item in num_cols) {
  sum=sum+item
  }
print(c("Total numeric cols:",sum))

par(mar=c(1,1,1,1))
par(mfrow=c(1,(sum-1)))
for (x in colnames(Churn_data[,c(-10,-11)])) {
  total=0
  if(is.numeric(Churn_data[,x])==T)
  {
    boxplot(Churn_data[,x],main=x,range = 5)
    out_val=boxplot.stats(Churn_data[,x],coef = 5)$out
    total=total+length(out_val)
    print(x)
    print(c("Total outliers:",total))
    print(out_val)
    cat("\n\n")
  }
}

#=====================================================================
#Setting outlliers to NA
#=====================================================================
par(mar=c(1,1,1,1))
for (x in colnames(Churn_data[,c(-10,-11)])) {
  if(is.numeric(Churn_data[,x]==T))
  {
    out_val=boxplot.stats(Churn_data[,x],coef=5)$out
    if(length(out_val)!=0){
      Churn_data[Churn_data[,x] %in% out_val,][,x]=NA
    }
    
  }
  
}

Churn_data=knnImputation(data=Churn_data,k=5)
#=============================================================================
#lets check weather we can apply standardlization or not
#=============================================================================
par(mfrow=c(2,6))
par(col.lab="red")
par(mar=c(3,2,5,2))
colnames(Churn_data[,4:9])
for (x in colnames(Churn_data[,4:9])) {
  hist(Churn_data[,x],main="Before Normalization",xlab=x,ylim=c(0,1500))
  abline(h=1000,lty=3)
  }
#=======================================================================================================
#from the graph it is clearly evident that most of th data is normally distributed except column 7 and 9
#=======================================================================================================
par(mfrow=c(4,2))
par(mar=c(2,2,1,1)+0.1)
for (x in colnames(Churn_data[,c(4:6,8)]))
{
  Churn_data[,x]=(Churn_data[,x]-mean(Churn_data[,x]))/(sd(Churn_data[,x]))
}
for (x in colnames(Churn_data[,c(4:6,8)])) {
  hist(Churn_data[,x],xlab=x,main="After Standardization",col="green",ylim = c(0,1500),xlim=c(-4,4))
  }
#==============================================================================
#splitting the data
#==============================================================================
Train=Churn_data[Churn_data$ID==1,]
Test=Churn_data[Churn_data$ID==2,]
Train$ID=NULL
Test$ID=NULL


#===================================================================================
#Model development
#===================================================================================
accuracy=vector()
model_name=vector()
f1_score=vector()
false_negative=vector()

model_scores=function(name,acc,f1,fn)
{
  model_name=append(model_name,name)
  accuracy=append(accuracy,acc)
  f1_score=append(f1_score,f1)
  false_negative=append(false_negative,fn)
}

par(mfrow=c(1,1))
par(mar=c(4,4,4,4))

#================================================================================================
#Decision tree
#================================================================================================
library(rpart)
library(tree)
library(MLmetrics)

name='Decision tree'
dt=tree(Train$Churn~.,Train,method = "class")
par(mar=c(0,0,0,0))
plot(dt,uniform=TRUE)
text(dt,use.n = TRUE,all=TRUE,cex=0.8)
tree_pred=predict(dt,Test,type = "class")
tab=table(actual=Test$Churn,perdicted=tree_pred)
tab
acc = sum(diag(tab))/length(tree_pred)
acc
f1 = F1_Score (y_true = Test$Churn, y_pred = tree_pred)
model_scores(name, acc, f1, tab[2])

#after prunig the tree found that it actually decreased the accurACY So leaving it as it is.
#set.seed(3)
#cv_tree= cv.tree(dt,FUN=prune.misclass)
#names(cv_tree)
#plot(cv_tree$size,cv_tree$dev,type="b")
#pruned_model=prune.misclass(dt,best = 10)
#plot(pruned_model)
#text(pruned_model,pretty=0)
#tree_pred=predict(pruned_model,Test,type = "class")




#===================================================================================================
#Random forest
#====================================================================================================
name='Random forest'
par(mfrow=c(1,1))
par(mar=c(4,4,4,4))
library(randomForest)
rf=randomForest(Train$Churn~.,data = Train,ntree=500)
rf_model=predict(rf,Test,type="class")
tab = table(actual = Test$Churn, predicted=rf_model)
tab
acc = sum(diag(tab))/length(rf_model)
f1 = F1_Score(y_true = Test$Churn, y_pred = rf_model)
model_scores(name, acc, f1, tab[2])


#========================================================================================================
#naive bayes
#========================================================================================================
library(e1071)
nb = naiveBayes(Train$Churn ~ ., data = Train, type="class")
nb_model = predict(nb, newdata = Test[, names(Test) != "Churn"],type = "class")
tab = table(actual = Test$Churn, predicted=nb_model)
tab
acc = sum(diag(tab))/length(nb_model)
f1 = F1_Score(y_true = Test$Churn, y_pred = nb_model)
model_scores(name, acc, f1, tab[2])
acc
f1

# ======================
# MODEL'S SCORE SUMMARY
# ======================
models_df = data.frame(name,acc, f1, tab[2])
models_df

