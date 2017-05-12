library(DMwR)
library(randomForest)
library(e1071)
library(rJava)
#library(RWeka)
library(rpart)
#install.packages("PRROC")
library(PRROC)
setwd("~/Documents/capstone/MALSAR")


PRcurve<-function(preds,trues,...){
  require(ROCR,quietly = T)
  pd<-prediction(preds,trues)
  pf<-performance(pd,"prec","rec")
  #rev:reverses a vector. cummax:obtains the cumulative maximum of a set of numbers
  pf@y.values<-lapply(pf@y.values,function(x) rev(cummax(rev(x))))
  plot(pf,xlim = c(0, 1), ylim = c(0,1))
  t = cor(cbind(as.numeric(preds >= 0.5) ,trues))[2]
  legend('topright',c('Matthews: ',  as.character(round(t,4))) )
}




rawdata = read.csv("XD_PCA_P_Simp.csv")



head(rawdata)
dim(rawdata)
# 208559 81
d = dim(rawdata)
row = d[1]
col = d[2]
rawdata = rawdata[,-1]
temp = as.numeric(as.Date(rawdata[,1], "%Y/%m/%d"))
rawdata[,1] = temp
category_variable = rawdata[,c(2,3,4,70)]
numerical_variable = rawdata[,c(1,5:7,9:68,71:75,78)]

temp = as.numeric(as.character(numerical_variable$option3))
numerical_variable$option3 = temp

original_data = cbind(category_variable, numerical_variable)

y = rawdata$option1
y = as.numeric(y);
y[y == 2] = 0

## normalize

idx<-sample(row, row*0.7)
train_x <-original_data[idx,]
col = dim(train_x)[2]

train_y <- y[idx]
test_x <-original_data[-idx,]
test_y <-y[-idx]
temp = train_x[,5:col]

mean = colMeans(temp, na.rm = TRUE)
std = apply(temp, 2, sd, na.rm = TRUE)
norm_train_x = scale(temp, center = mean, scale = std)
#col = 70
norm_train_x[is.na(norm_train_x[,70]), 70] = 0


norm_test_x = scale(test_x[5:col], center = mean, scale = std)
norm_test_x[is.na(norm_test_x[,70]), 70] = 0

#norm_train_x <- scale(train_x[,5:col])

train_data = cbind(train_x[,1:4],norm_train_x)
test_data = cbind(test_x[,1:4],norm_test_x)
  
train_label = factor(train_y)
test_label = factor(test_y)
old_data = cbind(train_label, train_data)
percover = ((length(train_y) / sum(train_y)) - 1)*100
#2032.501
balance_data = SMOTE(train_label~.,old_data,perc.over = 2100,perc.under = 100)
table(balance_data[,1])
################
balance_data <- read.csv(file="trainData.csv")
test = read.csv(file="testData.csv")
test_data = test[, c(-1, -2)]
test_label = test[, 2]
#######################
y1=as.numeric(balance_data[,2])
x1=balance_data[,c(-1, - 2)]
#m<-randomForest(y1~.,x1)
m<-rpart(y1~.,data=x1)
pred=predict(m,test_data)
corr = cor(cbind(as.numeric(pred >= 0.5) ,test_label))
#library(party)
mtree <- ctree(y1~., data = x1)
pred_tree = predict(mtree, test_data)

PRcurve(pred,test_label,main="regression tree")

pr <- pr.curve( test_label,pred, curve = TRUE );
# plot curve
plot(pr);

train = balance_data
write.csv(cbind(pred, test_label), file = "regressionTreeResult.csv")
write.csv(cbind(pred_tree,test_label), file = "decisionTreeResult.csv")

write.csv(train, file = "trainData.csv")
test = cbind(test_label, test_data)
write.csv(test, file = "testData.csv")

