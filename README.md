# Digit Recognizer With Machine Learning:
*Made by:* **Alexander Osokin**  
*Date:* **20.08.2020**  
*Kaggle Test Score:* **98.3% Accuracy**

## Objective:
**Is it possible to classify digit images using machine learning?**

***In this project I joined a competition on Kaggle where the objective is to classify correctly 28,000 digit images. When I explored the notebooks and the submissions of some participants I noticed that most of them if not all used a Convolutional Neural Network which shows high performance when it comes to image based classifications.***

***As a result, I decided to challenge myself and set a goal of achieving a classification accuracy above 95% using only machine learning algorithms (ex: Logistic Regression, Support Vector Machine, Random Forest, etc..)***

## Overview:
***When I joined the competition I got access to 3 .csv files: train.csv, test.csv, sample_submission.csv.***

*train.csv:* contains 48,000 rows (samples) and 785 columns (1 [label] 784 [pixel0 - pixel783])

*test.csv:* contains 28,000 rows (samples) and 784 columns (784 [pixel0 - pixel783])

*sample_submission.csv:* shows how the submission file should look like (1 [ImageId] & 1 [Label])

***Each image in the datasets was 28 pixels in height and 28 pixels in width, for a total of 784 pixels in each image. The images were reshaped into a 1D vector, resulting in 784-pixel columns. Each pixel contains an integer between 0-255 which indicates the lightness or darkness of the pixel, where higher numbers mean darker pixel.***

***The label column is present only in the training set and contains the digit that was drawn by the user.***

## Step 1 - Data Cleaning and Analysis:
***In this step I loaded both train and test sets and explored them a bit.***

***Firstly, I wanted to check if there were missing values which might be a result of a corrupted image. In that case that sample should be removed. Luckily, there were no missing values which means all images were normal.***

***Secondly, by applying panda's .info() function I checked if all the columns were of the correct data type.***

***Lastly, I checked the number of samples for each digit in the label column in order to make sure that the data set was had categorical balance.***

![alt_text][plot1]

[plot1]: https://github.com/AlexOsokin97/Digit_Recognizer/blob/master/Categorical%20Count%20Plot.png "CountPlot"

## Step 2 - Image Representation:
***Because my plan was to use feature engineering and dimension reduction techniques, I took 9 random images for each digit label, transformed them to grayscale and returned them to their original form as 28x28 pixels. Looking at the images gave me ideas for new feature creations and a general intuition on the variance of each digit (because all the digits were drawn by users and each user has his own drawing style).***

![alt_text][plot2]

[plot2]: https://github.com/AlexOsokin97/Digit_Recognizer/blob/master/Digit_Grids/6_grid.jpeg "digit6grid"

## Step 3 - Feature Engineering:

### Creating new features:
**I created 3 new features by using all the pixel columns and their values**

***Feature 1: Average Pixel Used*** - **I constructed this feature by writing a function with time complexity of O(n^2) which iterates through each sample (row), counts the number of pixels (columns) where the pixel's value is bigger than 0, divides the counter by 784 (total number of pixels) and stores the result at the new column with respect to the sample number.**

*method explanation:* *The reason this function has time complexity of O(n^2) is: because for each sample in the data set I had to iterate through every pixel and check the pixel's value. I tried to think of a different way which could achieve the same goal with better time complexity but unfortunately, I could not find it. The reason I chose pixel value > 0 was because a pixel which has a value > 0 means that it is darker and, in my case has a high probability of storing useful information*

***Feature 2: Max Pixel Value*** - **I constructed this feature by creating a function with time complexity of O(n) which reshapes all the samples into 28x28 grayscale image then, it iterates through every image, for every image it creates 4 sub images where every sub image is equal to 1/4 of the whole image and stores the average pixel value for that fraction and finally, it takes maximum average of the 4 sub images and stores it in the new column with respect to the sample.**

*method explanation:* *The reason this function has time complexity of O(n) is because it iterates through every sample in the data set. As a result, the more samples there are the longer it would take to complete (linear relationship). The reason I chose to split each image to 4 sub images, calculate the average pixel value for each sub image and take the maximum average of the 4 images was to resemble the convolution and max pooling methods used in a neural network (Each line of code which does those tasks has time complexity of O(1)).*

***Feature 3: Average Pixel Value*** - **I constructed this feature by creating a function with time complexity of O(n) which reshapes all the samples into 28x28 grayscale image then, it iterates through every image, takes the average pixel value of the image and stores it in the new column.**

*method explanation:* *The reason this function has time complexity of O(n) is because it iterates through every sample in the data set. I decided to create this feature because even though it is very basic it might be a valuable feature when it comes to classification decision because each digit from the same type will use roughly the same amount of pixels no matter how you draw it (as long as their images are on the same scale).*

### Feature Scaling & Applying PCA:
**After I done creating the new features, I ended up with 787/788 features (788-dimensional data). While there are few machine learning algorithm which can run on large dimensional data sets I decided to use Principal Component Analysis (PCA) to reduce the dimensionality of the data.**

***Feature Scaling:*** **In order to apply PCA on the data set I must scale the features. The newly created features were scaled between 0 < n < 1 during creation so there is no need to apply scaling on them. The pixel columns though, had values between 0 - 255 so, I scaled the pixel columns by dividing each value by 255.0. This not only scaled the data between 0 - 1 but also transformed the images into grayscale.**

***PCA:*** **Applied PCA on all the 784 pixels with 45 components. The 45 components together can explain 81% of the variance which means that most of the data stored in the pixels was preserved. I chose explaination of variance above 80% because all of the images mostly store 0 values and do not have a great impact on the model training. Lastly, I removed all the pixel columns and replaced them with all the pca components.**

***To sum up, I managed to reduce the dimensionality of the data from 784 dimensions to 48 dimensions which would reduce the training time of the algorithms greatly. As a result, it gave me the opportunity to use more complex machine learning algorithms by modifying their hyper parameters.***

## Step 4 - Machine Learning:
***Now that I had cleaned my data, got a general intuition on how it looks, engineered new features and reduced dimensionality I was ready to apply Machine Learning Algorithms***

### Functions:
***In order to save time and make the code as clean and short as possible I created 3 functions that would be used by every machine learning algorithm I decided to use***

**Function 1: cross_validation** - *This function takes a machine learning model and applies cross validation on the training label. Cross validation is a method that splits the data into K number of subsets, the model is fitted into each subset and for each subset returns the accuracy of the fit. This method is useful because it allows me to make sure that the dataset has sufficient features in order to predict the label and also it reduces to probability  of an accidental good/bad fit.*

**Function 2: hyper_tuning** - *This function takes a machine learning, and a dictionary of model's hyper parameters and applies it with different hyper parameters combinations on K number of subsets of the original dataset. This method is useful because it allows me to experiment with different hyper parameters in order to find the best combination  that returns the best fit score* 

**Function 3: model_performance** - *This function takes an already trained machine learning model, a list of scoring metrics and returns a dictionary with scores according to every scoring metric. This function is useful because it allows you to see your model's performance from different metrics and make changes/fixes accordingly.*

### Machine Learning Algorithms:
***Now that my Machine Learning enviornment had been set up I was ready to deploy the algorithms. I used 2 algorithms that were in my opinion were the best fit for this problem.***

### Support Vector Machines:
***The Support Vector Machines algorithm takes the given data points and applies them on a higher dimension that makes it possible to seperate them linearly, this is a achieved with the help of kernels. In addition, it uses a regularization parameter (C) that is responsible on the penalization of the algorithm when it misclassifys a sample, which directly affects the complexity of the hyper-plane. Lastly, the hyper-plane is a straight/curved line which is the result of the margin between the closest data points from each class (also called support vectors) which maximize the separabilty space between each class.***

**Why** - *I used this algorithm because according to the machine learning theory, the right data modification and transformation makes it possible to make a linear separation between clusters of data points and that is what the support vector machines does and for this reason, I used dimension reduction techniques and feature engineering. In addition, it is simple to implement and tune to get the best results.*
 
**Implementation** - *The first svm model that I tested was the simplest because, my goal when I featured engineered and used dimension reduction was that even a simple model implementation would achieve a high performance & after all, the simpler the better. The kernel was set to linear with regularization C of 3.5 with the use of cross validation function with k-folds = 10. The result: mean accuracy of 93%. One could argue that 93% is a good accuracy but, imagine we would need to classify 100,000 images with this model, we would misclassify 7,000 images and that is quite alot! As a result, I needed to modify my model in a way that would achieve a much better accuracy. For this task I used the hyper-parameter tunning function and created a hyper-parameter dictionary which consisted of hyper-parameters' names and list of values as their value. I used 3 hyper-parameters: kernel, C and tol. These 3 hyper parameters allowed me to directly control the complexity of the model's fit. When the training and fitting was complete I got the best accuracy of 98.14% with the following hyper-parameters: kernel='rbf', C=3.5, tol=0.1. The reason for those values might be that the data was still not linearly separable enough and as a result it had to use a more complex kernel method. Because our data set is very large and each feature does a good job of generalizing it's label high penalization of the model was not necessary and as a result the model managed to generalize better and making a better fit, same goes with tol=0.1. Now that I got the best model I needed to test it on the testing set.*

#### Test Scores:

***Confusion Matrix***<br/>
0: [417   0   2   0   0   0   1   0   0   0]<br/>
1: [  0 488   0   0   0   0   0   0   0   0]<br/>
2: [  0   0 412   2   2   0   0   3   2   0]<br/>
3: [  0   0   3 411   0   4   0   2   2   1]<br/>
4: [  0   1   2   0 412   0   0   4   0   7]<br/>
5: [  1   1   0   6   0 381   1   0   0   1]<br/>
6: [  1   0   1   0   2   0 399   0   0   0]<br/>
7: [  0   0   3   0   0   0   0 397   0   3]<br/>
8: [  0   3   0   2   2   1   0   0 401   2]<br/>
9: [  1   0   0   0   4   3   0   2   0 404]

***Accuracy Score:*** **98.14%**

#### K Nearest Neighbors:
