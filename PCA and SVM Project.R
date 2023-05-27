# Loading Packages
library(ggcorrplot)
library(scatterplot3d)
library(factoextra)
library(plotly)
library(caTools)
library(party)
library(e1071)

# Loading Data
# Using Breast Cancer Wisconsin data set from the UCI Machine Learning Repository as our data
data <- read.csv("E:\\Downloads\\wdbc.data", header = F)

features <- c("radius", "texture", "perimeter", "area", "smoothness", "compactness", "concavity", "concave_points", "symmetry", "fractal_dimension")
# Name all 32 variables. The ID, Diagnosis and ten distinct (30) features.
# The mean, standard error, and “worst” or largest (mean of the three largest values) of these features were computed for each image, resulting in 30 features.
names(data) <- c("ID", "Diagnosis", paste0(features,"_mean"), paste0(features,"_se"), paste0(features,"_worst"))

# Description of the tumor data
str(data)

# Check for null values
sum(is.na(data))

# New data with only numeric values; excluding the 'ID' and 'Diagnosis' variables
numerical_data <- data[,3:32]

# Scatter Plot Matrix of the first 11 features
pairs(numerical_data[,1:11], pch = 20)

# Data Standardization
data_standard <- scale(numerical_data)

# Computing the correlation matrix
corr_matrix <- cor(data_standard)

# Correlation matrix visualization (Heatmap)
ggcorrplot(corr_matrix)

#Applying PCA
pca <- prcomp(numerical_data, center = TRUE, scale = TRUE)
summary(pca)

# Scree Plot of the components
fviz_eig(pca, addlabels = TRUE)

# Kaiser’s rule
# An eigenvalues <1 would mean that the component explains less than a single explanatory variable.
screeplot(pca, type = "l", npcs = 15, main = "Screeplot of the first 15 PCs")
abline(h = 1, col="red", lty=5)
legend("topright", legend=c("Eigenvalue = 1"), 
       col=c("red"), lty=5, cex=0.6)

# Proportion of Variance (PoV)
# Finding Top n principal components which will at least cover 95 % variance of the original data
which(summary(pca)$importance[3,] >= 0.95)[1]

# Cumulative Variance Proportion Plot 
cum_pro= summary(pca)$importance[3, 1:15]
plot(cum_pro, xlab = "PC #", ylab = "Amount of explained variance", main = "Cumulative variance plot")
abline(v = 10, col="blue", lty=5)
abline(h = 0.95157, col="blue", lty=5)
legend("bottomright", legend=c("Cut-off @ PC10"),
       col=c("blue"), lty=5, cex=0.4)

# Biplot of the variables with respect to the principal components
fviz_pca_var(pca)
# Variables that are grouped together are positively correlated to each other.
# Variables that are negatively correlated are displayed to the opposite sides of the biplot’s origin. 
# The higher the distance between the variable and the origin (variable's magnitude), the better represented that variable is.


# Variables’ contribution to principal components (Square Cosine/ Cos2 score)
# determine how much each variable is represented in a given component.
# computing the square cosine value for each variable with respect to the first ten principal components. 
fviz_cos2(pca, choice = "var", axes = 1:10)

# Combination of Biplot and Cos2 score
fviz_pca_var(pca, col.var = "cos2",
             gradient.cols = c("black", "blue", "orange"),
             repel = TRUE)

# We can actually explain more than 60% of variance with just the first two components
# 2D scatter plot 
plot(pca$x[,1], pca$x[,2], xlab="PC1 (44.3%)", ylab = "PC2 (19%)", 
     main = "PC1 / PC2 - plot")

# We can also explain more than 70% of variance with  the first three principal components
# 3D scatter plot
scatterplot3d(pca$x[,1], pca$x[,2], pca$x[,3], color = "blue", pch = 16,
              main = "3D Scatter Plot",
              xlab = "PC1 (44.3%)", ylab = "PC2 (19%)", zlab = "PC3 (9.4%)")

# Notice the clustering going on in the upper/middle-right

# Add the response variable (Diagnosis) to the 2D plot 

fviz_pca_ind(pca, geom = "point", col.ind = as.factor(data$Diagnosis), 
             habillage = as.factor(data$Diagnosis), 
             palette = "jco",
             addEllipses = TRUE,
             legend.title= "Diagnosis") +
  ggtitle("2D PCA-plot from 30 feature dataset") +
  theme(plot.title = element_text(hjust = 0.5))

# Add the response variable (Diagnosis) to the 3D plot (using plotly)

components <- data.frame(pca[["x"]][,1:10])
components = cbind(components, data$Diagnosis)
names(components)[11] <- "Diagnosis"

tot_explained_variance_ratio <- 100* summary(pca)[["importance"]]['Cumulative Proportion',3]

tit = paste('Total Explained Variance = ', tot_explained_variance_ratio, '%')

fig <- plot_ly(components, x = ~PC1, y = ~PC2, z = ~PC3, color = ~Diagnosis, colors = c('#636EFA','#EF553B') ) %>%
  add_markers(size = 12)


fig <- fig %>%
  layout(
    title = tit,
    scene = list(bgcolor = "#e5ecf6")
  )

fig

# Recasting the standardized data along the first ten principal components axes

proj_data= data.frame(predict(pca, data_standard)[, 1:10])
final_data= cbind(proj_data, data$Diagnosis)

# 3D scatter plots

scatterplot3d(final_data$PC1, final_data$PC2, final_data$PC3, color = "blue", pch = 16,
              main = "3D Scatter Plot",
              xlab = "PC1 (44.3%)", ylab = "PC2 (19%)", zlab = "PC3 (9.4%)")


fig <- plot_ly(final_data, x = ~PC1, y = ~PC2, z = ~PC3, color = ~data$Diagnosis, colors = c('#636EFA','#EF553B') ) %>%
  add_markers(size = 12)


fig <- fig %>%
  layout(
    title = tit,
    scene = list(bgcolor = "#e5ecf6")
  )

fig


# Decision Tree Model

# Converting the class variable to be of type 'factor'
components$Diagnosis <- as.factor(components$Diagnosis)

# Splitting the data set into 80% training and 20% testing data sets
set.seed(1234)
sample_data = sample.split(components, SplitRatio = 0.8)
train_data <- subset(components, sample_data == TRUE)
test_data <- subset(components, sample_data == FALSE)
# Checking the dimensions of both train and test data sets
dim(train_data)
dim(test_data)

# Building the model

# First: Training the Decision Tree model  
model<- ctree(Diagnosis ~ ., train_data)

# Visualize the decision tree
plot(model)

#Second: Making predictions on the testing data set
predict_model<- predict(model, test_data)

# Calculating the confusion matrix
confusion_matrix <- table(test_data$Diagnosis, predict_model)
confusion_matrix

# Calculating the accuracy of the model
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)

print(paste('Accuracy of the model is found to be', round(accuracy*100), '%'))

# Support Vector Machines (SVM)

# build svm model, setting required parameters (by default cost (c) = 1)
# Soft-SVM (Linear Kernel trick) 
svm_model_1 <- svm(Diagnosis ~ ., data = train_data, type = "C-classification", 
                    kernel = "linear", scale = FALSE)

#list components of model
names(svm_model_1)

# list values of the SV, index and rho
# Cost (c) parameter 
svm_model_1$cost

# The indices of the support vectors in the original training dataset.
svm_model_1$index
length(svm_model_1$index)

# The support vectors of the SVM model. 
#Support vectors are the data points from the training set that lie closest to the decision boundaries. 
svm_model_1$SV

# The bias term or the intercept in the SVM model. 
# It indicates the offset or the shift of the decision boundary from the origin.
svm_model_1$rho

# compute training accuracy
pred_train <- predict(svm_model_1, train_data)
mean(pred_train == train_data$Diagnosis)

# compute test accuracy
pred_test <- predict(svm_model_1, test_data)
mean(pred_test == test_data$Diagnosis)

# Visualizing Linear SVMs

#build scatter plot of training dataset
scatter_plot <- ggplot(data = train_data, aes(x = PC1, y = PC2, color = Diagnosis)) + 
  geom_point()  

#add plot layer marking out the support vectors 
layered_plot <- 
  scatter_plot + geom_point(data = train_data[svm_model_1$index, ], aes(x = PC1, y = PC2), color = "purple", size = 4, alpha = 0.5)

#display plot
layered_plot

# Visualizing decision & margin bounds for cost = 1

# weight vector 
w_1 = t(svm_model_1$coefs) %*% svm_model_1$SV

#calculate slope and intercept of decision boundary from weight vector and svm model
slope_1 <- -w_1[1]/w_1[2]
intercept_1 <- svm_model_1$rho/w_1[2]

#add decision boundary to the scatter plot of the training dataset
plot_decision <- scatter_plot + geom_abline(slope = slope_1, intercept = intercept_1) 

#display plot of decision boundary
plot_decision

#add margin boundaries
plot_margins <- plot_decision + 
  geom_abline(slope = slope_1, intercept = intercept_1 - 1/w_1[2], linetype = "dashed")+
  geom_abline(slope = slope_1, intercept = intercept_1 + 1/w_1[2], linetype = "dashed")

#display plot of decision and margin boundaries
plot_margins

# Tuning cost (c) parameter

# build another svm model, cost (c) = 100
# Linear Kernel trick
svm_model_100 <- svm(Diagnosis ~ ., data = train_data, type = "C-classification",
                     cost = 100, kernel = "linear", scale = FALSE)

svm_model_100$cost

# The indices of the support vectors in the original training dataset.
svm_model_100$index
length(svm_model_100$index)
# Notice that the number of support vectors decreases as the cost increases


svm_model_100$rho

# compute training accuracy
pred_train <- predict(svm_model_100, train_data)
mean(pred_train == train_data$Diagnosis)

# compute test accuracy
pred_test <- predict(svm_model_100, test_data)
mean(pred_test == test_data$Diagnosis)

#add plot layer marking out the support vectors 
layered_plot_100 <- 
  scatter_plot + geom_point(data = train_data[svm_model_100$index, ], aes(x = PC1, y = PC2), color = "purple", size = 4, alpha = 0.5)

#display plot
layered_plot_100

# Visualizing decision & margin bounds for cost = 100

# Weight vector
w_100=t(svm_model_100$coefs) %*% svm_model_100$SV

#calculate slope and intercept of decision boundary from weight vector and svm model
slope_100 <- -w_100[1]/w_100[2]
intercept_100 <- svm_model_100$rho/w_100[2]

#add decision boundary to scatter plot of the training data
plot_decision_100 <- scatter_plot + geom_abline(slope = slope_100, intercept = intercept_100) 

#display plot of decision boundary
plot_decision_100

#add margin boundaries
plot_margins_100 <- plot_decision_100 + 
  geom_abline(slope = slope_100, intercept = intercept_100 - 1/w_1[2], linetype = "dashed")+
  geom_abline(slope = slope_100, intercept = intercept_100 + 1/w_1[2], linetype = "dashed")

#display plot of decision and margin boundaries
plot_margins_100