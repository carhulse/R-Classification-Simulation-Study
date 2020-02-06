##STAT 462 Final Exam##
#Question 1

#1.(15) Design and perform a simulation study on comparing the performance of (a) logistic regression models, 
#(b) linear discriminant analysis, (c) quadratic discriminant analysis, (d) treed-based classification.
#Make your own choice of sample size n, the number of predictors, p and the true model.

#let's simulate a linear ##model## y=B0+B1x1+e where e~N(0, 0.25), B0=0.5, B1=2, B2=-3, X1, X2~N(0,1)
#X=rnorm()
#eps= N
#True Model z=0.5 + 2*x1 + eps, y

set.seed(18)
sim.dat = function(n = 1000,beta_0=0.5, beta_1 = 2) {
  x1 = rnorm(n)
  eps = rnorm(1000, sd = sqrt(0.25))
  z = beta_0 + beta_1*x1 + eps
  pr = 1 / (1 + exp(-z))
  y = rbinom(n = n, size = 1, prob = pr)
  data.frame(y, x1)
}

set.seed(18)
sample.data = sim.dat()
head(sample.data)


library(caTools)
set.seed(101) 
sample = sample.split(sample.data, SplitRatio = 0.75)
train = subset(sample.data, sample == TRUE)
test  = subset(sample.data, sample == FALSE)


#(a) logistic regression
set.seed(18)
fit.logit = glm(y ~., data = train, family = binomial(link='logit'))
summary(fit.logit)
str(fit.logit)

prob.logit = predict(fit.logit, test, type = "response")
pred.glm = as.numeric(prob.logit>=0.5)

# Confusion matrix
conf_mat.glm = table(test$y, pred.glm)
conf_mat.glm

##pred.glm
##   0   1
## 0 163  67
## 1  44 226

# Test error rate
mean(pred.glm != test)
#0.611

# Precision: TP/(TP+FP)
conf_mat.glm[2,2]/(conf_mat.glm[2,2]+conf_mat.glm[1,2])
#0.7713311

# Sensitivity: TP/(TP+FN)
conf_mat.glm[2,2]/(conf_mat.glm[2,2]+conf_mat.glm[2,1])
#0.837037

#______________________________________________________________________________
#(b) LDA

library(MASS)
set.seed(18)
fit.lda = lda(y ~ ., data = train)
fit.lda 

model.matrix.lda = model.matrix(fit.lda)
model.matrix.lda

lda.pred = predict(fit.lda, test)
lda.pred

lda.class = lda.pred$class


# Confusion matrix
conf_mat.lda = table(test$y, lda.class)
conf_mat.lda
##    0   1
## 0 160  70
## 1  43 227


# Test error rate
mean(lda.class != test$y)
#0.226

# Precision: TP/(TP+FP)
conf_mat.lda[2,2]/(conf_mat.lda[2,2]+conf_mat.lda[1,2])
#0.7643098

# Sensitivity: TP/(TP+FN)
conf_mat.lda[2,2]/(conf_mat.lda[2,2]+conf_mat.lda[2,1])
#0.8407407


#______________________________________________________________________________
#(c) QDA

set.seed(18)
fit.qda = qda(y ~ ., data = train)
fit.qda 

pred.qda = predict(fit.qda, test)

# Confusion matrix
conf_mat.qda = table(test$y, pred.qda$class)
conf_mat.qda
##    0   1
## 0 159  71
## 1  43 227

qda.class = pred.qda$class

# Test error rate
mean(qda.class != test$y)
#0.228

# Precision: TP/(TP+FP)
conf_mat.qda[2,2]/(conf_mat.qda[2,2]+conf_mat.qda[1,2])
#0.761745

# Sensitivity: TP/(TP+FN)
conf_mat.qda[2,2]/(conf_mat.qda[2,2]+conf_mat.qda[2,1])
#0.8407407

#______________________________________________________________________________
#(d) Tree-based Classification
library(tree)

fit.tree = tree(as.factor(y) ~ ., data = train)
summary(fit.tree)
##Classification tree:
##  tree(formula = as.factor(y) ~ ., data = train)
##Number of terminal nodes:  5 
##Residual mean deviance:  0.8476 = 419.6 / 495 
##Misclassification error rate: 0.192 = 96 / 500 

plot(fit.tree)
text(fit.tree , pretty=0)
fit.tree 

pred.tree <- predict(fit.tree , test, type = "class")
pred.tree

library(caret)

#Confusion Matrix
y.test =test[,1]
table(pred.tree ,y.test)

##         y.test
## pred.tree   0   1
##         0 172  55
##         1  58 215

#Test error rate
mean(pred.tree != y.test) 
#=0.226

# Is a pruned tree better?
cvfit.tree = cv.tree(fit.tree, FUN = prune.misclass)
cvfit.tree
#$size
#[1] 7 2 1

#$dev
#[1] 184 178 296

best.size <- cvfit.tree$size[which(cvfit.tree$dev==min(cvfit.tree$dev))]
best.size #=2

#Prune tree
pruned.tree =prune.misclass(fit.tree ,best = 2)
plot(pruned.tree)
text(pruned.tree ,pretty =0)

summary(pruned.tree)
##Classification tree:
##  snip.tree(tree = fit.tree, nodes = 2:3)
##Number of terminal nodes:  2 
##Residual mean deviance:  0.9739 = 485 / 498 
##Misclassification error rate: 0.192 = 96 / 500 
