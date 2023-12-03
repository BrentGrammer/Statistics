# Gradient Descent

## The Goal
- The goal is to find what parameter(s) can be plugged into the model so that the derivative of the loss function is or close to zero (no or close to no slope).
  - If input to the model causes loss to increase, then the derivative of the loss fn will move further from zero (either away from or to zero). This will tell you which direction to move the parameter (either increase or decrease it to get the derivative of the loss closer to zero).
- The basic idea is to use the derivative of the loss function (for example mean squared error curve) and get the slope of the loss func curve at given points. We get the slope at a point `b` (random guess) and use it as a signal for whether we increase or decrease `b` to reduce loss (descend on the loss curve).
- "Gradient" is a term that is equivalent to "Derivative", but in the context of many dimensions (i.e., not just a one dimensional derivative). It is a collection of multiple partial derivatives with respect to all the dimensions of that function (see below).
- "Descent" refers to following the derivative down to get to the loss function minimum (least error)
- If the derivative is negative, you move positive (to the right) on the error landscape and vice versa.
- The goal of the derivative in the gradient descent algorithm is to move towards a value of zero. The goal of the equation curve is to move towards the true local minimum of the function curve
- Note that the slope value itself is not related to the parameter at all - only whether it is negative or positive is important as a signal to us to adjust the parameter up or down. The learning rate determines by how much and prevents overshooting (if using the slope directly)

### Loss Function

#### Mean Squared Error

- A standard loss function is the mean squared error for the model function $Lotsize + b = Price$. $${1\over n}\sum_{i=1}^n{(Predicted_i - Actual_i)^2}$$
  - The reasoning behind this is that we take the difference between the predictions and the actual data, and add square to eliminate negative values cancelling out positive values (we want all differences to contribute to the error). So the idea is we compare the result of the model given unknown variables to what the predictions would be given the same unknown variables fed into the model.
  - Take the output of the model given a param (guess the value), take the predictions and get the difference from that output to the actual value squared, and take the mean by dividing by number of data points. Do this for every data point and sum up the results. Then multiply that result by 1 and divide by n data points.
  - see [video](https://www.udemy.com/course/machine-learning-with-javascript/learn/lecture/12279864#questions)
  - example: `m * lot Size + b` (predicting prices of houses based on lot size, some multiplier m of the size and some b that is added to the price): Mean sqaured error would be: $${1\over n}\sum_{i=1}^n((m*x_i + b) - Actual_i)^2$$
- Start with a random guess of a parameter(s) and calculate msqe and then take another guess at the parameter and calculate msqe to compare the result. This is the basic loop to figure out how to reduce loss.
- Since the msqe method involved distance from the actual data points, it's unlikely it will every be 0. This would mean that the trend line is always exactly on the data points (which is unlikely since the data points will usually fall up or below a straight trend line.)

### Derivative of the loss function

See [Derivates](./Derivatives.md) for background and notes.

- For example, to get the slope of the loss function curve for the mean squared error (for $lotsize + b = Price$), the derivative would be: $${2\over n}\sum_{i=1}^n(b - Actual_i)$$
- $b$ is the point or value where we want to get the slope of the loss function (i.e. a random guess to start with). $n$ is the number of values in the dataset.
- We subtract the actual data value (i.e. house price if predicting house prices) from our guess `b` and sum all of those together (each data point subtracted from b), multiply that sum by 2 and divide by `n` data points (number of all points in the set)

#### Additional example (multi-term model function):

- The mean squared error function for $Price = mx + b$ is $${MSE} = {1\over n}\sum_{i=1}^n((m*x_i + b) - Actual_i)^2$$
  - The derivative for the MSE of the multi-term with respect to $m$, would be:
    $${d(MSE)\over dm} = {2\over n}\sum_{i=1}^n{-x_i(Actual = (mx_i + b))}$$
  - The derivative for MSE with respect to $b$ would be:
    $${d(MSE)\over db} = {2\over n}\sum_{i=1}^n((m*x_i + b) - Actual_i)$$

### Local minima

- low points in the error landscape that are "dips" and not the real global minimum loss in gradient descent
- Gradient descent can get "stuck" in these dips that occur (it will move to the right as error loss decreases and move back the left when error loss increases)
  - see [video](https://www.udemy.com/course/deeplearning_x/learn/lecture/27842082) at 4:00
- Gradient descent is gauranteed to go downhill (down the error landscape)
  - This is not a gaurantee that a good solution will be found if parameters are not set right for the particular error landscape

Note: it is impossible to visualize this on a graph beyond 2 dimensions.

- Somehow deep learning overcomes this pitfall, is successful and it is a mystery and not understood how.
  - it is possible that there are many different good solutions (different weights and params result in same high accuracy)
  - it's also possible that there are very few local minima in high-dimensional space (impossible to visualize and confirm). i.e. the local minimum would need to be a local min in all dimensions (unlikely), otherwise gradient descent will go a different direction to continue reducing loss
- Ways of dealing with this
  - (you can never be sure and have to rely on model accuracy) - use random weights and re-train the model many times.
  - increase the dimensionality and complexity of the model so that there will be fewer local minima

### Linear Regression

- Useful for fitting a line to data points that resemble a line (if the data points plotted do not roughly form a line then this may not be an appropriate method to apply)
- Deals with numerical data (the output is a real number)
- Is a form of supervised learning
