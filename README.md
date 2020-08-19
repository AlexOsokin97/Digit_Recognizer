# Digit Recognizer With Machine Learning:
*Made by:* **Alexander Osokin** , *Date:* **20.08.2020**

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
