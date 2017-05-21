# Team - Optimized Chaos
# Topic - Non-Linear modelling - Splines, Local Regression, and GAMs

#installing required packages 
install.packages("leaps")
install.packages("MASS")
install.packages("ggplot2")
install.packages("dplyr")
install.packages("mgcv")
install.packages("caret")
install.packages("ISLR")

#loading the packages 
library(MASS) 
library(ggplot2)
library(dplyr)
library(ISLR)

#Loading the Data set
View(mcycle)

#plotting data
plot(mcycle,pch=20)

############### REGRESSION SPLINES #####################################################

library(splines)

# demonstarting disconutity in piecewise polynomials without continuity
plot(mcycle$times, mcycle$accel, col = "grey", pch = 19, xlab = "# Times", ylab = "Acceleration", main = "Acceleration vs # Times")
cut_points = c(-1, 15, 20, 32, 40, 1000)
mcycle$label = cut(mcycle$times, breaks = cut_points, labels = c(1,2,3,4,5))

abline(v = cut_points, col = "red", lwd = 2)
lm1 = lm(accel~I(times^3), data = mcycle[mcycle$label ==1, ])
lm2 = lm(accel~I(times^3), data = mcycle[mcycle$label ==2, ])
lm3 = lm(accel~I(times^3), data = mcycle[mcycle$label ==3, ])
lm4 = lm(accel~I(times^3), data = mcycle[mcycle$label ==4, ])
lm5 = lm(accel~I(times^3), data = mcycle[mcycle$label ==5, ])

lines(x = mcycle$times[mcycle$label ==1], y = lm1$fitted.values, col = "blue", lwd = 2)
lines(x = mcycle$times[mcycle$label ==2], y = lm2$fitted.values, col = "green", lwd = 2)
lines(x = mcycle$times[mcycle$label ==3], y = lm3$fitted.values, col = "orange", lwd = 2)
lines(x = mcycle$times[mcycle$label ==4], y = lm4$fitted.values, col = "magenta", lwd = 2)
lines(x = mcycle$times[mcycle$label ==5], y = lm5$fitted.values, col = "maroon", lwd = 2)

mcycle$label=NULL # removing label column

#Case1 (Standard Cubic Polynomial)
plot(mcycle$times, mcycle$accel, col = "grey", pch = 19, xlab = "# Times", ylab = "Acceleration", main = "Acceleration vs # Times")
b=bs(mcycle$times,degree=3) # bs applies b-spline regression spline and outputs a basis matrix. 
# The default function used is a cubic polynomial although the polynomial can be adjusted using the degree parameter.
# The other option is to give the df in which case that many basis functions will be
# created.The number of knots is then df-degree.
lm2=lm(mcycle$accel~b) # applying linear regression on the basis output from b
pr=predict(lm2,newdata=data.frame(mcycle$times))
points(mcycle$times, pr, col = "deepskyblue3",lwd=3.4,type='l')

#Case2 (Cubic Spline) 
# Knots are defined at 3 quantiles - 25%, 50% and 75%
k=quantile(mcycle$times,probs = c(0.25,0.5,0.75))
# Degree = 3 for cubic spline
b=bs(mcycle$times,knots = k,degree=3)
lm2=lm(mcycle$accel~b)
pr=predict(lm2,newdata=data.frame(mcycle$times))
points(mcycle$times, pr, col = "red", lwd = 3.4, type='l')

#Case 3 (Quadratic Spline)
# We can choose different values of knots.
# Consider random knots in the domain at 15,20,33 and 45
# Degree = 2 for quadratic fit
b=bs(mcycle$times,knots = c(15,20,33,45),degree=2)
lm2=lm(mcycle$accel~b)
pr=predict(lm2,newdata=data.frame(mcycle$times))
points(mcycle$times, pr, col = "green", lwd = 3.4, type='l')

#Case 4 (Linear Spline)
# Degree = 1 for linear fit
b=bs(mcycle$times,knots = c(15,20,33,45),degree=1)
lm2=lm(mcycle$accel~b)
pr=predict(lm2,newdata=data.frame(mcycle$times))
points(mcycle$times, pr, col = "magenta", lwd=3.4, type='l')

#Plotting confidence intervals for Cubic Spline and Natural Spline.
b=bs(mcycle$times,knots = c(15,20,33,45),degree=3)
lm2=lm(mcycle$accel~b)
pr=predict(lm2,newdata=list(times1=mcycle$times),se=T)
plot(mcycle$times,mcycle$accel,pch=19,size=2,col="lavender",xlab="times",ylab="accel")
points(mcycle$times, pr$fit, col = "deepskyblue3",type='l',lwd=3.4)
lines(mcycle$times,pr$fit+2*pr$se ,lty="dashed")
lines(mcycle$times,pr$fit-2*pr$se ,lty="dashed")

b=ns(mcycle$times,knots = c(15,20,33,45)) # using natural cubic spline - to force linearity at the edges
lm2=lm(mcycle$accel~b)
pr=predict(lm2,newdata=list(times1=mcycle$times),se=T)
plot(mcycle$times,mcycle$accel,pch=19,size=2,col="lavender",xlab="times",ylab="accel")
points(mcycle$times, pr$fit, col = "green",type='l',lwd=3.4)
lines(mcycle$times,pr$fit+2*pr$se ,lty="dashed",col="red")
lines(mcycle$times,pr$fit-2*pr$se ,lty="dashed",col="red")

#Visualizing different cubic splines by setting different knots

spline_plot = function(k)
{
  ss = lm(mcycle$accel~bs(mcycle$times, knots = k, degree = 3))
  pr=predict(ss,newdata=data.frame(mcycle$times))
  ggplot(data = mcycle, aes(x = times, y = accel)) +
    geom_point(col = "lavender", size = 2) +
    geom_line(data =data.frame("times" = mcycle$times, "accel" = pr), col = "deepskyblue3", lwd = 1.2) +
    labs(title = paste0()) + theme_minimal() +
    theme(panel.grid = element_blank(), 
          panel.grid.major = element_blank(), 
          panel.grid.minor = element_blank(), 
          axis.line.x = element_line(color = "grey"), 
          axis.line.y = element_line(color = "grey")) +
    labs(title=paste0("Knots=",paste(k, collapse = ',')))
}
# Different knots are passed to the function to get the best fit

v = seq(1, max(mcycle$times), by=0.1)
p=rep(0.5,length(v))
require(gridExtra)
# 6 different plots are drawn at different values of knots
plot1 =spline_plot(k=sample(v,3,prob=p))
plot2 =spline_plot(k=sample(v,3,prob=p))
plot3=spline_plot(k=sample(v,3,prob=p))
grid.arrange(plot1,plot2,plot3)

############## SMOOTHING SPLINES ########################################################

ss = smooth.spline(mcycle$times, mcycle$accel) # smoothing spline function
# in the above case no additional parameters are passed and therefore, the 
# optimum smoothing param will be selected using cross validation

# Visualizing the impact of spar(penalty) variable on smoothing spline
spline_plot = function(sp = spar){
  ss = smooth.spline(mcycle$times, mcycle$accel, spar = sp)
  ggplot(data = mcycle, aes(x = times, y = accel)) +
    geom_point(col = "lavender", size = 2) +
    geom_line(data = data.frame("times" = ss$x, "accel" = ss$y), col = "deepskyblue3", lwd = 1.2) +
    labs(title = paste0("Spar = ", sp)) + theme_minimal() +
    theme(panel.grid = element_blank(), 
          panel.grid.major = element_blank(), 
          panel.grid.minor = element_blank(), 
          axis.line.x = element_line(color = "grey"), 
          axis.line.y = element_line(color = "grey")) 
}

spar_range = seq(from = 0, to = 1, by = 0.3) # varying smoothing parameter over a range
plot_list = lapply(spar_range, spline_plot)

# loading gridextra to visualize all impact of spar values
library(gridExtra)
do.call(grid.arrange, c(plot_list, list(ncol=2)))


# How to pick best value of spar - cross validation
spline_spar = function(spar){
  spar_model = smooth.spline(x = mcycle$times, y = mcycle$accel, spar = spar)
  gcv = spar_model$cv.crit
  return(gcv) # generalized cross validation score
}

spar_range = seq(from = 0, to = 1, by = 0.2) 
gcv_range = sapply(spar_range, spline_spar)

plot(spar_range, gcv_range, type = "l", col = "orange", lwd = 2, lty = 2, xlab = "Spar", ylab = "GCV", main = "GCV vs Spar")
points(spar_range, gcv_range, pch = 19, col = "blue")
abline( v= spar_range[which.min(gcv_range)], col = "blue", lwd = 2)

# However smoothness of spline can also be controlled by degrees of freedom. 
# Visualizing the impact of varying df on smoothness of splines
spline_plot = function(df = df){
  ss = smooth.spline(mcycle$times, mcycle$accel, df = df)
  ggplot(data = mcycle, aes(x = times, y = accel)) +
    geom_point(col = "lavender", size = 2) +
    geom_line(data = data.frame("times" = ss$x, "accel" = ss$y), col = "deepskyblue3", lwd = 1.2) +
    labs(title = paste0("df = ", round(df))) + theme_minimal() +
    theme(panel.grid = element_blank(), 
          panel.grid.major = element_blank(), 
          panel.grid.minor = element_blank(), 
          axis.line.x = element_line(color = "grey"), 
          axis.line.y = element_line(color = "grey")) 
  
}

df_range = seq(from = 2, to = 93, length.out = 3)
plot_list = lapply(df_range, spline_plot)
do.call(grid.arrange, c(plot_list, list(ncol=2)))


############## LOESS ####################################################################

#creating Interpolation model
#predictor = times
#target variable = accel
#data set = mcycle
#alpha = 0.1,0.25,0.5,0.75
pd1=loess(accel~times,data=mcycle,span=0.25)
pd2=loess(accel~times,data=mcycle,span=0.50)
pd3=loess(accel~times,data=mcycle,span=0.75)
pd4=loess(accel~times,data=mcycle,span=1)


#Creating prediction
sm1=predict(pd1,se=T)
sm2=predict(pd2,se=T)
sm3=predict(pd3,se=T)
sm4=predict(pd4,se=T)


#plotting results
plot(mcycle,pch=20)
lines(sm1$fit,x=mcycle$times,col="red",lwd=2)
lines(sm2$fit,x=mcycle$times,col="green",lwd=2)
lines(sm3$fit,x=mcycle$times,col="blue",lwd=2)
lines(sm4$fit,x=mcycle$times,col="grey",lwd=2)
legend("topright",c("0.25","0.5","0.75","1.0"),col=c("red","green","blue","grey"),lwd=c(2,2,2,2))


#When the fit was made using surface = "interpolate" (the default),
#predict.loess will not extrapolate - 
#so points outside the original data will have missing (NA) predictions
#and standard errors.


#creating Extrapolation model
#predictor = times
#target variable = accel
#data set = mcycle
#alpha = 0.25,0.50,0.75,1.0
sme1=predict(pde1,data.frame(times=seq(1,100,1)),se=T)
sme2=predict(pde2,data.frame(times=seq(1,100,1)),se=T)
sme3=predict(pde3,data.frame(times=seq(1,100,1)),se=T)
sme4=predict(pde4,data.frame(times=seq(1,100,1)),se=T)

plot(sme1$fit,x=seq(1,100,1),col="red",lwd=2,type="l")
lines(sme2$fit,x=seq(1,100,1),col="green",lwd=2)
lines(sme3$fit,x=seq(1,100,1),col="blue",lwd=2)
lines(sme4$fit,x=seq(1,100,1),col="grey",lwd=2)
legend("topleft",c("0.25","0.5","0.75","1.0"),col=c("red","green","blue","grey"),lwd=c(2,2,2,2))


pde1=loess(accel~times,data=mcycle,span=0.25,control=loess.control(surface="direct"))
pde2=loess(accel~times,data=mcycle,span=0.50,control=loess.control(surface="direct"))
pde3=loess(accel~times,data=mcycle,span=0.75,control=loess.control(surface="direct"))
pde4=loess(accel~times,data=mcycle,span=1.0,control=loess.control(surface="direct"))

# Visualizing the impact of span on LOESS prediction
library(ggplot2)
# Creating function for plotting different span values
loess_plot = function(s=span)
{
  sm=predict(loess(accel~times,data=mcycle,span=s))
  ggplot(data = mcycle, aes(x = times, y = accel)) +
    geom_point(col = "lavender", size = 2) +
    geom_line(aes(x = mcycle$times, y = sm), 
              col = "deepskyblue3", lwd = 1.2) +
    labs(title = paste0("Span = ", s)) + theme_minimal() +
    theme(panel.grid = element_blank(), 
          panel.grid.major = element_blank(), 
          panel.grid.minor = element_blank(), 
          axis.line.x = element_line(color = "grey"), 
          axis.line.y = element_line(color = "grey")) }

# creating a range of span values
span_range = c(0.25,0.50,0.75,1.0)
# applying plotting function to each value in span_range
plot_list = lapply(span_range, loess_plot)
# loading library gridextra to visualize all impact of span values
library(gridExtra)
do.call(grid.arrange, c(plot_list, list(ncol=2)))

#To find out which data points are being used for fitting a polynomial
install.packages("TeachingDemos")
library(TeachingDemos)
#The loess.demo function will interactively demonstrate the 
#window and the weights that was used to fit that particular target point.
loess.demo(mcycle$times,mcycle$accel,span=0.1)
#Click finish to stop the function
#The larger circles depict higher weights and vice versa. 
#Here a span of 0.1 has been use which means 10% of the data has been used near that target data point.

################# GAM #######################################################################

## loading the libraries

library(ISLR)
library(mgcv)


data("College")
str(College) # dataset has 777 observations with 18 variables

## Our aim is to predict out of tuition fee using all the other variables available in the data

## Splitting data into training and testing set
set.seed(111)
split = sample(1:nrow(College), size = round(0.8*nrow(College)), replace = F)

train = College[split, ]
test = College[-split, ]
nrow(train) + nrow(test) == nrow(College)

## to demonstarte the concept of GAM, we will select most relevant predictors. There are multiple ways to do this -
## 1. Using variables with highest correlation to dependent variable
## 2. Using subset selection technique - forward, backward, or stepwise
## 3. Checking out variable importance plot using output of non-parameteric technique like random-forest

### Technique 1
num_var_index = sapply(College, is.numeric)
cor_train = cor(train[num_var_index])
corrplot::corrplot(cor_train, order = "hclust") # Variables including acceptance rate and application rate are highly correlated and can be removed. Checking against Outstate - none of the variable is highly correlated. However to clean the input data, we can remove the identified variables. This can also be established in an orderly fashion using caret package

library(caret)
findCorrelation(cor_train, names = T, cutoff = 0.9) # Output of the function gives out names of variables that have a correlation greater than 0.9 with other variables and must be removed from the train dataframe

train = train[, !(colnames(train) %in% findCorrelation(cor_train, names = T, cutoff = 0.9))]
str(train) # we are now left with 15 variables

### Technique 2

library(leaps)
exhaustive = regsubsets(Outstate~.,data=train,nvmax = 14, method = "exhaustive")
plot_df = NULL
plot_df$MSE = summary(exhaustive)$rss
plot_df$Pred = 1:14
plot_df = as.data.frame(plot_df)

ggplot(data = plot_df, aes(x = Pred, y= MSE)) +
  geom_line(color ="deepskyblue3", lwd = 1.5) +
  geom_point(color = "red", lwd = 3) +
  theme_minimal() +
  labs(title = "MSE vs #Predictors", x = "# Predictors", y ="MSE") +
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        axis.line.x = element_line("grey"), axis.line.y = element_line("grey")) # From the plot it can be inferred that 7-8 predictors will be good enough for prediction

subset_vars = names(coef(exhaustive, 8)) 
subset_vars # these are the 8 best variables as per the subset selection. As private was changed into dummy variables, and only privateYes was marked significant, we must include the Private in GAM model
subset_vars = subset_vars[-1]
subset_vars[1] = "Private"


### Technique 3
#### A quick and dirty way to find most relevant predcitors is to use variable importance plot from random forest
rf = randomForest::randomForest(Outstate~., data = College)
randomForest::varImpPlot(rf, n.var = 7, main = "") # we can see that most significant varibales highlighted by random forest are mostly in sync with what we found from exhaustive subset selection

#### subsetting train data frame with variables selected from exhastive subset selection
train = train[c("Outstate", subset_vars)]

## Fitting GAM Model
### All variables except Private are quantitative, so a smoother can be applied to those

gam_m1 = gam(Outstate ~ Private + s(Top10perc) + s(Room.Board) + s(Personal) + s(Terminal) + s(perc.alumni) + s(Expend) + s(Grad.Rate),  data = train)

plot.gam(gam_m1, col = "deepskyblue3", lwd = 2, se = T, shade = T,pages=2) ## the output shows that room.board, expend, grad.rate, and personal are highly non-linear. This can be verified from the p-value obtained in summary

## From this plot we can make a statement like - Colleges with higher out of state tuition have a higher graduation rate
## or - Out of state fee can  be expected to increase almost linearly for colleges with instructional expenditure between 0-20k, but beyond that an increase in instructional expenditure do not necessarily have an impact on out of state tuition fee

summary(gam_m1) # this model is able to explain about 80% of variance from the dataset

## Prediction
### Predcition from a gam model follow a similar syntax as other linear models in R
gam_m1.predcit = predict(object = gam_m1, newdata = test)
gam_m1.rmse = sqrt(mean((test$Outstate - gam_m1.predcit)^2)) # 1925
gam_m1.mae = mean(abs(test$Outstate - gam_m1.predcit)) #1500
gam_m1.mape = round(mean(abs(test$Outstate - gam_m1.predcit)/abs(test$Outstate)),2) # 0.17

gam_stats = c("rmse" = gam_m1.rmse, "mae" = gam_m1.mae, "mape" = gam_m1.mape)

## Checking the performance against a random forest model
library(randomForest)
rf2 = randomForest(Outstate~., data = train)
rf2.predict = predict(object = rf2, newdata = test)
rf2.rmse = sqrt(mean((test$Outstate - rf2.predict)^2)) #1800
rf2.mae = mean(abs(test$Outstate - rf2.predict))  #1342
rf2.mape = round(mean(abs(test$Outstate - rf2.predict)/abs(test$Outstate)),2) #0.15

rf_stats = c("rmse" = rf2.rmse, "mae" = rf2.mae, "mape" = rf2.mape)

cbind(gam_stats, rf_stats) # there is a difference of 2% points in GAM vs random forest. While the difference may increase in case of tuned RF model, the stats clearly shows that GAM models are capable of modeling non-linear functions and have an added advantage of interpretability over random forest or boosted tree models