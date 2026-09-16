#neural network algorithm_______________________________________________________
rm(list=ls())
df2 <- read.csv("creditcards.csv")
library(nnet)
str(df2)

#clean 
df2$ID <- NULL

#convert target variable to factor
df2$default.payment.next.month <- as.factor(df2$default.payment.next.month)

#randomize
set.seed(12345)
df2r <- df2[sample(nrow(df2)),]
str(df2r)

#scale
df2s <- df2r
df2s[, 1:23] <- scale(df2r[,1:23])


#partition
# 60% training, 40% testing
train <- df2s[1:18000,]
test <- df2s[18001:30000,]

#nn model
df2nnet <- nnet(default.payment.next.month ~., data = train, size = 3,linout = FALSE, maxit = 1500)

df2predict <- predict(df2nnet, test[,1:23], type = "class")

#table
cm <- table(test[,24], df2predict)
cm

#accuracy
sum(diag(cm))/sum(cm)

#__ACCURACY_____________________________________________________________________

#accuracy for size =1  is 0.7990833
#accuracy for size =2  is 0.8148333
#accuracy for size =3  is 0.81575 <- highest
#accuracy for size =5  is 0.814
#accuracy for size =7  is 0.8123333
#accuracy for size =10 is 0.8071667
#accuracy for size =20 is 0.7983333
#accuracy for size =25 is 0.795
#accuracy for size =30 is 0.7930833

#accuracy cited by paper= .81 
#(1-.19(error rate)= .81)

################################################################################
#K-Nearest Neighbor algorithm___________________________________________________

#import data
df3 <- read.csv("creditcards.csv")
str(df3)
library("class")

#clean 
df3$ID <- NULL

df3$default.payment.next.month <- as.factor(df3$default.payment.next.month)
#Set the random number generator 
set.seed(12345)

#randomize
df3r <- df3[sample(nrow(df3)),]

#scaling
df3s <- as.data.frame(scale(df3r[,1:23]))
#add back the target variable
df3s$default.payment.next.month <- df3r$default.payment.next.month

#partitioning
#85% for training, 15% for testing
trainingdata1 <-  df3s[1:25500,]
trainingclass1<-  df3s[1:25500,24]
testingdata1  <-  df3s[25501:30000,]
testingclass1 <-  df3s[25501:30000,24]



df3_knn <- knn(trainingdata1, testingdata1, trainingclass1, k = 5)

mytable1 <- table(testingclass1, df3_knn)
accuracy <- sum(diag(mytable1))/sum(mytable1)
print(accuracy)

#__ACCURACY________________________________________________________________________________________________________
#k = 5 accuracy   is 0.9284444 #<- highest
#k = 10 accuracy  is 0.9222222
#k = 20 accuracy  is 0.9042222
#k = 32 accuracy  is 0.8908889 
#k = 50 accuracy  is 0.8793333
#k = 60 accuracy  is 0.874


#Accuracy reported in paper = .8276


################################################################################
#naive Bayes algorithm__________________________________________________________
df4 <- read.csv("creditcards.csv")
library("e1071")
df4$ID <- NULL
str(df4)
#set target variable as factor
df4$default.payment.next.month <- as.factor(df4$default.payment.next.month)

#randomizing
set.seed(12345)
mydatar1 <- df4[sample(nrow(df4)),]

#partitioning
#60% for training, 40% for testing
traindata1 <- mydatar1[1:18000, 1:23]
trainclass1 <- mydatar1[1:18000, 24]
testdata1 <- mydatar1[18001:30000, 1:23]
testclass1 <- mydatar1[18001:30000, 24]

#model
mynb <- naiveBayes(traindata1, trainclass1, laplace = 200)
nbp <-predict(mynb, testdata1)

nbcm <- table(testclass1, nbp)
sum(diag(nbcm))/sum(nbcm)
#____ACCURACY__________________________________________________________________
#accuracy when laplace = 1   is 0.74325
#accuracy when laplace = 5   is 0.74325
#accuracy when laplace = 10  is 0.74325 
#accuracy when laplace = 50  is 0.74325
#accuracy when laplace = 100 is 0.74325
#accuracy when laplace = 200 is 0.74325

#accuracy reported in paper = .6723


