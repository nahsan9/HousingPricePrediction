# Project title: House Price Analysis and Prediction with Machine Learning Methods
## Team members: Yandong Luo, Xiaochen Peng, Hongwu Jiang and Panni Wang

---
# 1. Motivaton
House price is highly concerned by people that are looking for places to live or opportunities to invest. Actually,  

<p align="center">
  <img src="https://github.com/yandongluo/HousingPricePrediction/blob/master/Figures/Introduction.png" width="450"/>
</p>

In fact, many factors influence housing price, such as the area of the house, the number of bedrooms, the location et. al. Therefore, in this project, we will focus on the following three aspects related to house price, which is important to help us make a good deal:  
1. Find the main factors that influence house price with feature selection methods such as recursive feature selection (RFE) and RandomForest
2. Build house price prediction model using linear regression and neural network
3. House recommendation based on consumers’ preference with k-neareat neighbor method (K-NN)

---
# 2. Dataset and visulization 

### (1). Dataset: House Sales in King County (from Kaggle)
#### Features in the dataset: 21 features in total
1. id: notation for a house  
2. date: date house was sold  
3. price: the sell price of the house, which is what we need to predict
4. bedrooms: Number of Bedrooms/House
5. bathrooms: Number of bathrooms/House  
6. sqft_livingsquare footage of the home  
7. sqft_lotsquare footage of the lot  
8. floorsTotal: floors (levels) in house  
9. waterfront: House which has a view to a waterfront  
10. view: Has been viewed?  
11. condition: How good the condition is ( Overall )  
12. grade: overall grade given to the housing unit, based on King County grading system  
13. sqft_above: square footage of house apart from basement  
14. sqft_basements: quare footage of the basement  
15. yr_built: Built Year  
16. yr_renovated: Year when house was renovated  
17. zipcode: zip  
18. lat: Latitude coordinate  
19. long: Longitude coordinate  
20. sqft_living15: Living room area in 2015 This might or might not have affected the lotsize area  
21. sqft_lot15: lot size area in 2015  

### (2). dataset visulization
15 features are visulized as below. The features as the following characteristics: 
1. The scale of each feature is quite different. It needs to be normalized
2. There are continuos variables (sqrf_lot et.al), dicreste variables (bedrooms) and categorical variable (yr_renovated)

<p align="center">
  <img src="https://github.com/yandongluo/HousingPricePrediction/blob/master/Figures/Feature_Dist.PNG">
</p>

---
# 3. Data pre-processing
After the dataset is visulized and examined, the data is processed in following ways: 
1. remove irrelevant features: id, date, lat, long, zipcode
2. remove the feature "waterfront" as it is 0 for all the data points
3. normalize the all the features with its mean and sigma as their scale is quite different
4. There is a categorical feature: yr_renovated. It is either 0 or the year that it has been renovated. It is treat as a dummy variable with only two values: "1" if the house has been renovated and "0" if it has not. 

---
# 4. Feature selection 

When the dataset comes to be high dimensional, it could lead to many problems. Firstly, such high dimension will significantly increase the training time of the model, and dramatically increase the complexity of the model. Secondly, the model may make decisions based on the noise, and thus, cause overfitting. What's more, the meaningless redundent data will be very misleading, and could decrease the accuracy.
To achieve better performance, we firstly used two feature selection methods to cancel out the redundent features, one of the methods is to draw the correlation matrix with heatmap, the second method is to calculate the feature importance and select the top-N features according to the feature ranking.

### (1). Dataset Analysis

To better understand the relationship among the features, we have drawn the pairplots for some features, the features we selected are those we considered as very important as our first thought. The pairplots shows how "bathrooms", "bedrooms" and "sqft_living" are distributed vis-a-vis the price as well as the "grade", which means the grading of the houses by the local county. As the pairplot shown below, we could find some linear distribution between price and the features, which could be useful in our linear model.

<p align="center">
  <img src="https://github.com/xiaochen76/CX4240-Project-House-Price-Predict/blob/master/Figures/Feature_Plot.png">
</p>

### (2). Correlation Heat Map

To find out how each feature is correlated to our target variable "price", we have drawn the correation heat map of all the features as shown below:

<p align="center">
  <img src="https://github.com/xiaochen76/CX4240-Project-House-Price-Predict/blob/master/Figures/HeatMap_ALL.PNG" width="500"/>
</p>

It shows that, the correlation value can be positive or negative, when the correlation is positive, it means the increasing of value in such feature will cause the target variable "price" to increase, and vice versa.

We have further drawn the most significant correlated features with target variable "price", as shown below:

<p align="center">
  <img src="https://github.com/xiaochen76/CX4240-Project-House-Price-Predict/blob/master/Figures/HeatMap_select.PNG" width="500"/>
</p>

It is very easy to identify which features are most related to the target variable, as it shown above, the most important features from the heatmap are: bathrooms, bedrooms, floors, grade, sqft_above, sqft_living and sqft_living15.

### (3). Feature Ranking

Of course, using the corelation heap map itself, could be not representive enough, thus, we further used the feature ranking functions in each models and get the mean ranking values, to select the top-N important ones.
We have implemented five representative models to get their scores about the features, and get the mean values of them, which are linear regression, ridge, lasso, recursive feature elimination and random forest model.
As the figures shown below, the first three linear models returned same feature ranking results, the most important features are: grade, view, bathrooms, bedrooms, yr_renovated, floors and conditions.

<img src="https://github.com/xiaochen76/CX4240-Project-House-Price-Predict/blob/master/Figures/FeatureImp_LR.PNG" width="280"/> <img src="https://github.com/xiaochen76/CX4240-Project-House-Price-Predict/blob/master/Figures/FeatureImp_Ridge.PNG" width="280"/><img src="https://github.com/xiaochen76/CX4240-Project-House-Price-Predict/blob/master/Figures/FeatureImp_Lasso.PNG" width="280"/>

However, in recursive feature elimination, we get a quite different feature ranking as shown below:

<p align="center">
  <img src="https://github.com/xiaochen76/CX4240-Project-House-Price-Predict/blob/master/Figures/FeatureImp_RFE.PNG" width="400"/>
</p>

While in random forest model, the feature ranking is give as below:

<p align="center">
  <img src="https://github.com/xiaochen76/CX4240-Project-House-Price-Predict/blob/master/Figures/FeatureImp_RF.PNG" width="400"/>
</p>

We could find that, in recursive feature elimination and random forest feature ranking, there are more features as continuous data on the top ranks, such as the area of living rooms and lots (sqft_living and sqft_lot). While in the linear model, the top ranked features are discrete data.

To get a more balanced feature ranking, we normalized the scores from each model and get the mean values, the final feature ranking is shown as below:

<p align="center">
  <img src="https://github.com/xiaochen76/CX4240-Project-House-Price-Predict/blob/master/Figures/FeatureImp_Average.PNG" width="400"/>
</p>

Where the top ranked features include both continuous and discrete data: sqft_living, sqft_lot as continuous data, grade, view, bedrooms and bathrooms are discrete data.

# 5. Housing price prediction with linear regression

<img src="https://github.com/xiaochen76/CX4240-Project-House-Price-Predict/blob/master/Figures/Price_Bathrooms.PNG" width="280"/> <img src="https://github.com/xiaochen76/CX4240-Project-House-Price-Predict/blob/master/Figures/Price_Bedrooms.PNG" width="280"/><img src="https://github.com/xiaochen76/CX4240-Project-House-Price-Predict/blob/master/Figures/Price_SqLiving.PNG" width="280"/>

As what we have shown in the dataset analysis, some features shows a classical linear relationship, while some do not have very good linear form, thus, we use both of the linear and polynomial regression, to study how the data distribution affect the linear models.

After data pre-processing as we mentioned above, there are 14 features left in the dataset, which is divided in to training and testing set. With linear regression, when the number of features is too low, it could suffer from under-fitting, and get poor performance in both training and testing, while if the features are too many, it is also possible to get over-fitting problem. To learn about the number of features in the linear regression, we have run linear and polynomial regression, based on three models (linear regression, ridge regression and lasso regression), with all the 14 features and only top-10 important features. 

In ridge regression, it shrinks the coefficients (w) by putting constraint on them, and thus, helps to reduce the model complexity and multi-collinearity. Similarly, in lasso regression, the regularization will lead to zero coefficients, which means some of the features are completely neglected for the evaluation of output, thus, lasso regression not only helps in reducing over-fitting, but also helps in feature selection.

The reason why we used three linear models is, the ridge and lasso regression are some of the simple techniques to reduce model complexity and prevent over-fitting which may result from simple linear regression. By comparing the linear regression, ridge regression and lasso regression, we can also get an insight about how the number of features affect the model performance.

### (1). ALL Features Included

As the figure shown below, where red line is the real price value, and the blue dots are the predicted price value, the first row shows the linear, lasso and ridge regression without polynomial, the second row shows when polynomial in introduced with degree equals to 2, and the third is with degree equals to 3. It shows that, with polynomial, the prediction achieves better performance, since it can help to fit in non-linear features.

<p align="center">
  <img src="https://github.com/xiaochen76/CX4240-Project-House-Price-Predict/blob/master/Figures/Predict_All.PNG">
</p>

### (2). Selected Top-10 Features Included

<p align="center">
  <img src="https://github.com/xiaochen76/CX4240-Project-House-Price-Predict/blob/master/Figures/Predict.PNG">
</p>

The figure shown above is the relation between real price and predicted price, when we only introduced the top-10 important features. The first column shows the linear, ridge and lasso regression, and the second column shows the ones with polynomial (degree is set to 2). Similarly as what we have found in the "all features included" method, the linear regression achieves best performance among all the three linear models.

### (3). Comparison and Discussion

It is clearly shown that, no matter in the "all features included" or "selected top-10 features", the linear regression achieves the best performance. While in each method, the ones with polynomial achieves better prediction.

To find out the comparison among all the methods, we have printed out the rmse values of them, as the figure shown below.

<p align="center">
  <img src="https://github.com/xiaochen76/CX4240-Project-House-Price-Predict/blob/master/Figures/RMSE.PNG" width="400"/>
</p>

It further proves our thoughts, the rmse values in linear regression are the lowest ones, no matter for "all features" or "selected top-10 features". While when we fix the regression methods (just linear regression, ridge or lasso regression), we found that, the "all features" always achieves better rmse than "selected top-10 features" ones. It reveals that, the linear regression cannot well fit the dataset and predict the target variable.

# 6. Housing price prediction with neural netwok

### (1). neural network vs linear regression
A 2-layer neural network with fully connected layer is implemented for house price prediction. The hidden layer unit is 64, the activation function at the hidden layer is ReLU and the output is the house price. The prediction is evaluated with root-mean-squred-error (RMSE) of the predicted house price. The neural network is trained with 20 epoch.  
First, the RMSE obtained by neural network method is compared with that of linear regression, as shown in the figure below. Neural network shows lower loss than all the linear regression based methods, which indicates that it can be a good model for house price prediction.

<p align="center">
  <img src="https://github.com/yandongluo/HousingPricePrediction/blob/master/Figures/MRSE_ANN.png" width="400"/>
</p>

## prediction loss vs. number of hidden units
Then we examined the prediction loss with different neural network settings. The prediction loss of the neural network can be decreased by increasing the number of hidden units, as shown in the figure below. It means that a more complicated model is desired for this task

<p align="center">
  <img src="https://github.com/yandongluo/HousingPricePrediction/blob/master/Figures/Loss_vs_hidden_units.png" width="400"/>
</p>

## prediction loss vs. activation function
Three different activation functions are examined ReLU, Sigmoid and Tanh. The results shows that prediction loss is small when "ReLU" is used while the prediction loss is large when the activation is Sigmoid or Tanh. The prediction loss vs. training epoch is plotted. The neural networks with Sigmoid and Tanh shows slow training as the neuron activation value is limited, which is (0,1) for Sigmoid and (-1,1) for Tanh

<p align="center">
  <img src="https://github.com/yandongluo/HousingPricePrediction/blob/master/Figures/MRSE_ANN_activations.png" width="300"/> <img src="https://github.com/yandongluo/HousingPricePrediction/blob/master/Figures/Loss_vs_activation_type" width="300"/>
</p>

## Prediction loss vs. optimizers
The prediction loss of neural network trained with different optimization method are also examined with SGD and RMSprop. RMSprop shows faster convergence and less fluctuations when it is convergent because the learning rate can be varied during the training. 

<p align="center">
  <img src="https://github.com/yandongluo/HousingPricePrediction/blob/master/Figures/Loss_vs_optimizer.png" width="400"/>
</p>
  
# 7. Housing recommendation with K-NN
The house recommendation is conducted with k-neareast neighbor algorithm to find the house that best matches the consumer's preference, which is measured by the Euclidean distance between the house in the dataset and the preference input by consumer. An example is shown in the table below, where 5 recommendations are made. It is noted that house price is an important factor as recommendations are trying to match the price expected by consumers. As consumers are price sensitive, it indicates the K-NN works well for house recommendation. 

<p align="center">
  <img src="https://github.com/yandongluo/HousingPricePrediction/blob/master/Figures/Recommendation.JPG">
<p/>

# 8. Discussions (the questions in proposal) 
a. Do all the feature ranking methods list the same informative features? And do those features ranked in the same order? 
Answer: Yes, almost the same. In both recursive feature elimination (RFE) and random forest feature ranking, categorical features such as grade (the grad evaluated by agency)  

b. With the same set of features, which regression model provides the most accurate prediction. 
Answer: Lasso and Ridge regression with polynomial features (degree = 2) provides the most accurate prediction results as it prevents overfitting. 

c. How to choose the proper methods for prediction
Answer: in this project, neural network shows the smallest rmse loss for prediction. The factors that influence the prediction accuracy are the number of hidden units, activation functions. For house price prediction, hidden units of larger than 64 is preferred and ReLU activation provides faster training and better accuracy 

# 9. Conclusion
a. Obtained features that influence house price the most 
Obtained the features that has the highest impact on house price with two feature selection methods: recursive feature elimination (RFE) and random forest.  
It can be concluded that categorical featuress such as the grade of the house and the number of rooms have the highest impact on house price. The room area is not important for house price. 

b. Build the house prediction model
Both linear regression and neural network are implemented. 
Neural network provides better prediction. 
More hidden units and use 'ReLU' as activation can help improve the prediction

c. House recommendation by K-NN
Recommend house based on consumer's needs
Price is an important factor to match

# 10. Reference
[1]Park, B. and J. K. Bae (2015). "Using machine learning algorithms for housing price prediction: The case of Fairfax County, Virginia housing data." Expert Systems with Applications 42(6): 2928-2934.  
[2]Gür Ali, Ö. et. al (2013). "Selecting rows and columns for training support vector regression models with large retail datasets." European Journal of Operational Research 226(3): 471-480.  
[3]Breiman, L. (2001). "Random Forests." Machine Learning 45(1): 5-32.

