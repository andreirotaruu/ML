## Logistic Regression 

# What it is 
- Classification model
* this means we are classifying the probability that something lies in a certain category
- Instead of fitting a line to to the data logistic regression fits a S shaped function called a logistic function

# The math
- vector xi with n amount of features
* we have m instances of these features
- theta vector containing n parameters
- z = theta * xi
* we need to use gradient descent to find the parameters such that theta * xi = z