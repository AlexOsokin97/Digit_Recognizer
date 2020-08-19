# Digit Recognizer With Machine Learning:
*Made by:* **Alexander Osokin**  
*Date:* **20.08.2020**  
*Kaggle Test Score:* **98% Accuracy**

## Objective:
**Is it possible to classify digit images using machine learning?**

***In this project I joined a competition on kaggle where the objective is to classify correctly 28,000 digit images. When I explored the notebooks and the submissions of some participants I noticed that most of them if not all used a Convolutional Neural Network which shows high performance when it comes to image based classifications.***

***As a result, I decided to challenge myself and set a goal of achieving a classification accuracy above 95% using only machine learning algorithms (ex: Logistic Regression, Support Vector Machine, Random Forest, etc..)***

## Overview:
***When I joined the competition I got access to 3 .csv files: train.csv, test.csv, sample_submission.csv.***

*train.csv:* contains 48,000 rows (samples) and 785 columns (1 [label] 784 [pixel0 - pixel783])
*test.csv:* contains 28,000 rows (samples) and 784 columns (784 [pixel0 - pixel783])
*sample_submission.csv:* shows how the submission file should look like (1 [ImageId] & 1 [Label])

***Each image in the datasets was 28 pixels in height and 28 pixels in width, for a total of 784 pixels in each image. The images were reshaped into a 1D vector, resulting in 784 pixel columns. Each pixel contains an integer between 0-255 which indicates the lightness or darkness of the pixel, where higher numbers mean darker pixel.***

***The label column is present only in the training set and contains the digit that was drawn by the user.***

## Step 1: Data Cleaning and Analysis:
***In this step I loaded both train and test sets and explored them a bit.***

***Firstly, I wanted to check if there were missing values which might be a result of a corrupted image. In that case that sample should be removed. Luckliy, there were no missing values which means all images were normal.***

***Secondly, by applying panda's .info() function I checked if all the columns were of the correct data type.***

***Lastly, I checked the number of samples for each digit in the label column in order to make sure that the data set was had categorical balance.***

![alt_text][plot1]

[plot1]: https://github.com/AlexOsokin97/Digit_Recognizer/blob/master/Categorical%20Count%20Plot.png "CountPlot"

## Step 2: Image Representation:
***Because my plan was to use feature engineering and dimension reduction techniques I took 9 random images for each digit label, transformed them to grayscale and returned them to their original form as 28x28 pixels. Looking at the images gave me ideas for new feature creations and a general intuition on the variance of each digit (because all the digits were drawn by users and each user has his own drawing style).***

![alt_text][plot2]

[plot2]: https://github.com/AlexOsokin97/Digit_Recognizer/blob/master/Digit_Grids/6_grid.jpeg "digit6grid"

## Step 3: Feature Engineering:
***I created 3 new features by using all the pixel columns and their values***

***Feature 1: Average Pixel Used*** - **I constructed this feature by writing a function with time complexity of O(n^2) which iterates through each sample (row), counts the number of pixels (columns) where the pixel's value is bigger than 0, divides the counter by 784 (total number of pixels) and stores the result at the new column with respect to the sample number.**

*method explaination:* *The reason this function has time complexity of O(n^2) is: because for each sample in the data set I had to iterate through every pixel and check the pixel's value. I tried to think of a different way which could achieve the same goal with better time complexity but unfortunatly, I could not find it. The reason I chose pixel value > 0 was because a pixel which has a value > 0 means that it is darker and in my case has a high probability of storing useful information*

***Feature 2: Max Pixel Value*** - **I constructed this feature by creating a function with time complexity of O(n) which reshapes all the samples into 28x28 grayscale image then, it iterates through every image, for every image it creates 4 sub images where every sub image is equal to 1/4 of the whole image and stores the average pixel value for that fraction and finally, it takes maximum average of the 4 sub images and stores it in the new column with respect to the sample.**

*method explaination:* *The reason this function has time complexity of O(n) is because it iterates through every sample in the data set. As a result, the more samples there are the longer it would take to complete (linear relationship). The reason I chose to split each image to 4 sub images, calculate the average pixel value for each sub image and take the maximum average of the 4 images was to resemble the convolution and max pooling methods used in a neural network (Each line of code which does those tasks has time complexity of O(1)).*

***Feature 3: Average Pixel Value*** - 

