rm(list = ls())
setwd("C:/Users/chinna/Desktop/Data Science/Project/Employee Absenteeism")

#libraries
x = c("ggplot2", "corrgram", "DMwR", "caret", "randomForest", "unbalanced", "C50", "dummies", "e1071", "Information",
      "MASS", "rpart", "gbm", "ROSE", 'sampling', 'DataCombine', 'inTrees','readxl')

#load packages(x)
lapply(x, require, character.only = TRUE)
rm(x)

#data
employee = read_excel("Absenteeism.xls")
employee = as.data.frame(employee)

str(employee)

options(warn = -1)

#changing names of the variable.
colnames(employee)[colnames(employee) == 'Absenteeism time in hours'] = 'Absenteeism_time_in_hours'
colnames(employee)[colnames(employee) == 'Reason for absence'] = 'Reason_for_absence'
colnames(employee)[colnames(employee) == 'Month of absence'] = 'Month_of_absence'
colnames(employee)[colnames(employee) == 'Day of the week'] = 'Day_of_week'
colnames(employee)[colnames(employee) == 'Transportation expense'] = 'Transportation_expense'
colnames(employee)[colnames(employee) == 'Distance from Residence to Work'] = 'Distance_from_residence_to_work'
colnames(employee)[colnames(employee) == 'Service time'] = 'Service_time'
colnames(employee)[colnames(employee) == 'Work load Average/day'] = 'Work_load_Average_per_day'
colnames(employee)[colnames(employee) == 'Hit target'] = 'Hit_target'
colnames(employee)[colnames(employee) == 'Disciplinary failure'] = 'Disciplinary_failure'
colnames(employee)[colnames(employee) == 'Social drinker'] = 'Social_drinker'
colnames(employee)[colnames(employee) == 'Social smoker'] = 'Social_smoker'
colnames(employee)[colnames(employee) == 'Body mass index'] = 'Body_mass_index'

#missing value analysis
missing_val = data.frame(apply(employee,2,function(x){sum(is.na(x))}))
missing_val$Columns = row.names(missing_val)
names(missing_val)[1] =  "Missing_percentage"
missing_val$Missing_percentage = (missing_val$Missing_percentage/nrow(employee)) * 100
missing_val = missing_val[order(-missing_val$Missing_percentage),]
row.names(missing_val) = NULL
missing_val = missing_val[,c(2,1)]
cnames = colnames(employee)

#imputing missing vals
for (i in cnames) {
  employee[,i][is.na(employee[,i])] = median(employee[,i], na.rm = T)
}


#converting variables to their types
employee$ID = as.factor(employee$ID)
employee$`Reason_for_absence`=as.factor(employee$`Reason_for_absence`)
employee$`Month_of_absence` = as.factor(employee$`Month_of_absence`)
employee$`Day_of_week` = as.factor(employee$`Day_of_week`)
employee$Seasons = as.factor(employee$Seasons)
employee$`Disciplinary_failure` = as.factor(employee$`Disciplinary_failure`)
employee$Education = as.factor(employee$Education)
employee$`Social_drinker` = as.factor(employee$`Social_drinker`)
employee$`Social_smoker`= as.factor(employee$`Social_smoker`)

str(employee)


#############OUTLIER ANALYSIS##################

numeric_index = sapply(employee,is.numeric) #selecting only numeric
numeric_data = employee[,numeric_index]
cnames = colnames(numeric_data)

for (i in 1:length(cnames))
   {
     assign(paste0("gn",i), ggplot(aes_string(y = (cnames[i]), x = "Absenteeism_time_in_hours"), data = subset(employee))+ 
              stat_boxplot(geom = "errorbar", width = 0.5) +
              geom_boxplot(outlier.colour="red", fill = "blue" ,outlier.shape=18,
                           outlier.size=1, notch=FALSE) +
              theme(legend.position="bottom")+
              labs(y=cnames[i],x="Absenteeism_time_in_hours")+
              ggtitle(paste("Box plot of responded for",cnames[i])))
}

#boxplot of outliers
gridExtra::grid.arrange(gn1,gn2,gn3,ncol=3)
gridExtra::grid.arrange(gn4,gn5,gn6,ncol=3)
gridExtra::grid.arrange(gn7,gn8,gn9,ncol=3)
gridExtra::grid.arrange(gn10,gn11,gn12,ncol=3)

#outlier impute using NA technique
for (i in cnames) {
  val = employee[,i][employee[,i] %in% boxplot.stats(employee[,i])$out]
  employee[,i][(employee[,i] %in% val)] = NA
  employee[,i][is.na(employee[,i])] = median.default(employee[,i], na.rm = T)
}

rm(val)

############FEATURE SELECTION##################

#correlation plot
corrgram(employee, order = F,
         upper.panel=panel.pie, text.panel=panel.txt, main = "Correlation Plot")


str(employee)

df = employee
#employee = df

#ANOVA
result = aov(formula = Absenteeism_time_in_hours ~ ID+Reason_for_absence+Month_of_absence+Day_of_week+Seasons+Disciplinary_failure
             +Education+Social_drinker+Social_smoker , data = employee)
summary(result)


#library(randomForest)
#imp_var = randomForest(Absenteeism_time_in_hours ~ ., data = employee, ntree = 100,
#                       keep.forest = FALSE, importance = TRUE)
#importance(imp_var, type = 1)

#Dimentionality reduction
employee = subset(employee, select = -c(ID, Seasons, Education, Hit_target, Son, Pet, Height, Body_mass_index, 
                                        Age, Disciplinary_failure))
#############FEATURE SCALING###############
#Normality check
hist(employee$Transportation_expense)
hist(employee$Distance_from_residence_to_work)
hist(employee$Service_time)
hist(employee$Work_load_Average_per_day)
hist(employee$Weight)
hist(employee$Absenteeism_time_in_hours)

str(employee)

#Normalization
cnames = c("Transportation_expense","Distance_from_residence_to_work","Service_time","Work_load_Average_per_day",
           "Weight")

for(i in cnames){
  print(i)
  employee[,i] = (employee[,i] - min(employee[,i]))/
    (max(employee[,i] - min(employee[,i])))
}

###############MODEL IMPLEMENTATION

library(rpart)
library(MASS)
#install.packages("rpart.plot")
library(rpart.plot)

rmExcept(c("df","employee"))

train_index = sample(1:nrow(employee), 0.8 * nrow(employee))
train = employee[train_index,]
test = employee[-train_index,]

######DECISION TREE REGRESSION#####

fit = rpart(Absenteeism_time_in_hours ~ ., data = train, method = "anova")
fit
plt = rpart.plot(fit, type = 3, digits = 2, fallen.leaves = TRUE)

prediction_dt = predict(fit, test[-11])

actual = test[,11]
predicted = data.frame(prediction_dt)

error = actual - predicted

rmse <- function(error)
{
  sqrt(mean(error^2))
}

rmse(error)

#ERROR = 3.5147
#Accuracy = 96.4853

##### RANDOM FOREST REGRESSION#######
tree = randomForest(Absenteeism_time_in_hours ~ ., data = train, ntrees = 100)
tree
plot(tree)

prediction_rf = predict(tree, test[-11])

actual = test[,11]
predicted = data.frame(prediction_rf)

error = actual - predicted

rmse <- function(error)
{
  sqrt(mean(error^2))
}

rmse(error)

#ERROR = 2.7376
#Accuracy = 97.3624

library(usdm)
library(car)
library(VIF)

employee$`Reason_for_absence`=as.numeric(employee$`Reason_for_absence`)
employee$`Month_of_absence` = as.numeric(employee$`Month_of_absence`)
employee$`Day_of_week` = as.numeric(employee$`Day_of_week`)
employee$`Social_drinker` = as.numeric(employee$`Social_drinker`)
employee$`Social_smoker`= as.numeric(employee$`Social_smoker`)

str(employee)

vif(employee[,-11])
vifcor(employee[,-11], th = 0.9)

lm_model = lm(Absenteeism_time_in_hours ~ ., data = train)
summary(lm_model)

prediction_lr = predict(lm_model, test[,1:10])

#regr.eval(test[-14], prediction_dt, stats = c('mse','rmse','mape','mae'))


actual = test[,11]
predicted = data.frame(prediction_lr)

error = actual - predicted

rmse <- function(error)
{
  sqrt(mean(error^2))
}

rmse(error)

#ERROR = 3.1516
#Accuracy= 96.8484

str(employee)

########################## 2nd PART PREDICTION OF LOSS FOR THE COMPANY IN EACH MONTH##############################

new = subset(df, select = c(Month_of_absence, Service_time, Absenteeism_time_in_hours, Work_load_Average_per_day))

#Work loss = ((Work load per day/ service time)* Absenteeism hours)

new["loss"]=with(new,((new[,4]*new[,3])/new[,2]))

for(i in 1:12)
{
  d1=new[which(new["Month_of_absence"]==i),]
  print(sum(d1$loss))
  
}
             
a = c(1,2,2)
b = c (1,2)
a*b
b*a
