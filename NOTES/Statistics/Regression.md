# Regression

## GLM (General Linear Model)

- A framework for doing analyses. Regression is one type of a General Linear Model
- GLMs include ANOVAs, Regression and Correlation - these are all linear models
  - LINEAR means that we only use two operations: Scalar-Multiplication and Addition of the regressors ($\beta_n$)
    - Scalar multiplication means multiplying by a single number, no square roots, logs, trigonometry etc. (these are all nonlinear operations)
    - The data can contain nonlinearities and relationships, but the parameters must be linear in the model.
    - Nonlinear data are often linearized to facilitate interpretation

### The Mathematical Model

- We have data (measurements from the real world) and a Mathematical Model that we want to try to match or fit to the data
- The Mathematical Model has two main parts:
  - **Fixed Features:** defined by you, the engineer a priori based on intuition, theory, what you expect to see in the data, etc.
    - These are fixed and immutable - they are built into the model
  - **Free Parameters:** variables in the model that are allowed to change depending on how well the model fits the data. The goal is to change these to make the model fit the data better (within the constraints of the fixed features)

### When to use an ANOVA vs. a Regression:

- An ANOVA should be used when all the independent variables are discrete (i.e. categories)
  - or if you discretize continuous data (like age) into categorical data (young, old etc.)
- A Regression should be used when at least one of the IVs are continuous.
  - Similar to an extension of a correlation (correlation is between two variables, while a regression can be extended to multiple variables)
  - It's possible to have categorical data in a Regression, but they need to be dummy-coded (converted to numeric)

## Fitting a Model

- Define equations underlying the model
  - Example: $\text{height} = \beta_0 + \beta_1{S} + \beta_2{p} + \beta_3{n}$
    - This model equation would be run on each individual in the dataset for example.
  - s, p, and n (sex parents nutrition) are the fixed features
  - $\beta_n$ when before a parameter are the Beta Coefficients (a.k.a. regression coefficients) which make the fixed parameters flexible (how much weight or how important they are in the equation)
  - $\beta_0$ is the **intercept** term - we need this in every linear model. It captures an average when all the other parameters are 0 (i.e. set all $\beta_n$ to 0 and it results in $\text{height}=\beta_0$).
    - The intercept is where $\beta_0$ crosses the y-axis when all the parameters ($x_i = 0$ i.e. 0 on the x-axis) are set to zero. see [video](https://www.udemy.com/course/statsml_x/learn/lecture/20225688#content) at timestamp 9:11
    - Without an intercept we force the line of fit to go through the 0 origin (see [video](https://www.udemy.com/course/statsml_x/learn/lecture/20225676#content) at timestamp 14:45)
    - EXCEPTION: You do not need an intercept term if all of your data is mean centered, i.e. if average of dependent and all independent variables is 0 then the intercept is gauranteed to be 0.
- Map the data into the model equation
  - Plug in the data into the equation (the parameters)
- Convert equations into matrix-vector equation

  - Note: the computer does this work for you
  - Separate the parts of the equation into a matrix and two vectors (see [video](https://www.udemy.com/course/statsml_x/learn/lecture/20225676#content) at timestamp 9:27)

  ```python
  [
    [1 1 175 8] # each row is an individual in dataset, each col is a Idependent Variable
    [1 1 175 6] # the first col is intercept (set default to 1) and the second col is sex (dummy coded to 0 or 1 for female/male) - make the one you think is more significant in effect 1 and the other 0!
    [1 1 189 7]
  ]

  [
    Beta0 # vector of beta coefficients (unknowns we want to solve for in model fitting)
    Beta1
    Beta2
    Beta3
  ]

  =

  [  # vector of dependent variables/outcomes (actual heights)
    180
    170
    176
  ]
  ```

  - This is the GLM equation and can be shorthanded with: $$X\beta = y$$
    - $X$ is called the Design Matrix (cols are independent variables/predictors/regressors)
    - $\beta$ - Beta/Regression coefficients - things we are trying to fit/unknowns we're solving for
    - $y$ - dependent variable or outcome

## Compute the parameters (statistical analysis)

- Statistical evaluation of the model
  - Is the model a good fit to the data?
  - Are individual $\beta$ coefficients statistically significant? (i.e. a particular beta coefficient or parameter is not important or contributing to the model)

## Least Squares Solution

- A solution to a Regression Model
  - Efficient for a computer to fit - even if you have a very large dataset
  - A closed form solution (we can prove this is correct even without any data)
  - Deterministic and not stochastic - always gives the same solution

$$\beta = (X^TX)^{-1}X^T y$$

- Solving for Beta by multiplying the design matrix by the left inverse and the outcomes $y$

- Solving for $\beta$ in the GLM equation $X\beta = y$
  - Note that $X$ is not a number, it's the Design Matrix, so we can't just divide both sides by X
  - We need to multiply by **X-inverse** (a matrix that cancels out the original matrix)
    - But X will not have an inverse if it is a tall matrix (more rows than columns). Only square matrices have inverses.
    - We can alternatively apply the **One-Sided Inverse** (the Left-Inverse) if assumptions on conditions are made about X.
      $$(X^TX)^{-1}X^T X\beta = (X^TX)^{-1}X^T y$$
    - LEFT INVERSE is $(X^TX)^{-1}X^T$: X transposed (T) times X and then inverted (-1 exp). Though X does not have an inverse, X Transposed DOES have an inverse.
    - We also do the left inverse on both sides of the equation (since we're solving for Beta)
    - Left inverse of X obliterates X and isolates Beta.
- After isolating Beta, Beta is the left inverse of the Design Matrix $X$ times the dependent variable $y$: $\beta = (X^TX)^{-1}X^T y$

### Assumptions needed for Design Matrix

- X needs to be a Tall Matrix (more rows than cols, i.e. more observations or data points than features)
  - For Deep Learning scenarios where wide matrices with more features than observations, this solution will not work. Use gradient descent etc. instead.
- Independent Variables must be "Independent"

  - IVs must not be correlated or be able to be combined in any way
  - Multicollinearity: see [video](https://www.udemy.com/course/statsml_x/learn/lecture/20225678) at timestamp 7:55

    - If you have independent variables in the design matrix measuring the same thing then you can't use least sqaures solution model.
    - Usually multicollinearity results from a mistake (i.e. repeating a column in the Beta Coeficients matrix twice for some reason). Has a Rank-2 matrix.

      - Usually the python function will throw an error about not being able to compute the model, rank deficiency, etc.
      - Example:

      ```python

      [
        182 182 5
        179 179 3
        162 162 1 # note the first two cols are always the same numbers.
      ]
      ```

## Regression Models: R-squared and F

- Epsilon term in the GLM formula:
  $$y = \beta_0 + \beta_1 x_1 ... + \beta_k x_k + \epsilon$$
  - $x_1$... is the regressors (independent variables)
  - $\epsilon$ captures residual variance (error variance or "innovation" term). Everything that is not fit or not predicted about the data by the model.
  - We don't want the $\epsilon$ to be too small! It will be small, but if it is close to or at 0 this means your research question is too simple and not interesting or that the model is too complicated and it should be paired down. The goal is not to explain every single piece of variability, it is to develop simplified models to help us understand complexity in the real world.
  - The epsilon residuals is equal to the actual data minus the predicted data (y hat: $\hat{y}$)
    $$\epsilon = y - \hat{y}$$
    - So if the model is perfect, then $\epsilon$ will be 0.
    - Note: $\hat{y}$ is the predictions output from the model: $\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 ...$

### R-squared

- Used to evaluate the overall fit of the model to the data.
- The total overall fit of the model to the data can be calculated as $R^2$
  $$R^2  = 1-{SS_{\epsilon} \over SS_{\text{Total}}}$$
  - 1 minus the sum of squares for the residual divided by the sum of squares total
    $$1 - {\sum(y_i - \hat{y}_i)^2 \over \sum(y_i - \bar{y})^2}$$
    - Denominator: take each individual data point $y_i$ minus the mean of all the datapoints $\bar{y}$, square that and sum all those up.
    - Numerator: subtract the model prediction for a data point $\hat{y}$ from that data point $y_i$, square that, sum all of those
    - We subtract 1 from all of this to make large $r^2$ correspond to better model fits for convenience.
      - i.e. if there is no error in the denominator, the whole expression evaluates to 0. Adding 1 minus that will give us 1 to indicate a perfect fit.
- An R-squared close to 1 means that the model fits the data well. R-squared closer to 0 means that the model is a poor fit to the data.
  - if R-squared is negative then that means there is likely an error or a mistake was made. It means the model fit is so bad that it's predicting results that are worse than the mean (the numerator is greater than the denominator)
    - the denominator is a model of sorts that every data point is equal to the mean over all data points.

### F-score

- There is no cutoff or threshold for a good R-squared score to indicate we've reached a goal or have a good fit. So we get an F-score that incorporates the R-squared
- Null Hypothesis: $H_0: \beta_{1-k} = 0$
  - Beta coefficients 1 through k are equal to 0. Note: The null hypothesis allows for a non-zero intercept term ($\beta_0$)
  - The null hypothesis translates to: $y = \beta_0 + \epsilon$ (all the beta coefficients are taken out since they are zeros)
- Alternative Hypothesis: At least one Beta Coefficient is not equal to 0
- Similar to an F-test in an ANOVA, this F-score just tells us that at least one Beta Coefficient is not equal to zero (but it doesn't tell us which one or how many are not equal to 0)
  - more inspection and post-hoc analysis is necessary
  - If the F-statistic is greater than what you would expect by chance, then the model fits the data well, and then you can start evaluating each individual beta coefficient: $$t_{N-k} = {\beta \over s_\beta}$$
    - This is a T-distribution with N-k degrees of freedom and the ratio is each individual beta coefiicient $\beta$ (represents one beta coeifficient) divided by the standard deviation of the beta (the square root of the variance in the model scaled by the total variance in the data - this scales each beta coefficient)
      - This T-test tells us if the individual beta coefficient is statistically different from 0.
      - Note that $k$ is the total number of parameters in model including intercept

## Simple Regression

- Has one Dependent Variable and one Independent Variable (technically 2 IVs if you count the intercept)
  - The one IV that is not the intercept is the parameter that we are bringing in from our intuition or theory because we think it's important in the model
    $$y_i = \beta_0 + \beta_1 x_i + \epsilon_i$$
  - The ith data point $y_i$ equals the intercept $\beta_0$ plus the beta coefficient and independent variable for the ith data value $\beta_1 x_i$ plus the i-th error/residual/innovation term $\epsilon_i$
  - Solving for $\epsilon$:
    $$\epsilon_i = y_i - (\beta_0 + \beta_1 x_i)$$
    - The error/residual is the i-th data point minus the model-predicted data

### Example of a simple regression

- Does the amount of sleep effect how much money people spend on food shopping?
- Experiment: Ask 10 random people (N=10) at a food market to report number of hours sleep the previous night and their total shopping bill.
- Collected data in the design matrix

```python
# first col is the intercept, second col is hours slept (IV - Beta_1(x_i))
[
  [1   5]
  [1 5.5]
  [1   6]
  [1   6]
  [1   7]
  [1   7]
  [1 7.5]
  [1   8]
  [1 8.5]
  [1   9]
]

x # matrix multiplication

# beta coefficient vetor
[
    [Beta0] # intercept
    [Beta1] # slope - linear relationship between hours slept and money spent
]

=

# total money spent (DV - y_i)
[
    47
    53
    52
    44
    39
    49
    50
    38
    43
    40
]
```

- A best fit line means that the squared distances from the model predicted points to the actual points are as small as they possibly can be (see [video](https://www.udemy.com/course/statsml_x/learn/lecture/20225688#content) at timestamp 8:11)
  - This is computed by the least squares equation multiplying the left inverse of the design matrix against the outcome data:
- example model values (least squares computation solution): $y_i = 63 - 2.5x_i + \epsilon_i$
  - The intercept is 63 dollars and this means that when all the parameters ($x_i = 0$ i.e. 0 on the x-axis) are set to zero the data crosses the y-axis at 63.
    - In context of example, this would mean that if someone slept 0 hours then they will spend 63 dollars on food.
    - See [video](https://www.udemy.com/course/statsml_x/learn/lecture/20225688#content) at timestamp 9:12 for visual explanation of intercept on y-axis
  - The beta coefficient $- 2.5x_i$ predicts that on average for every notch on the x-axis (every hour a person does sleep), they will spend 2.5 dollars less on food (-2.5 on the y-axis).
  - Example analysis results: R^2 = .36, F(1,8) = 4.54, p = .066
    - Since the p-value is 6.6% (over 5%) it tells us that we reject the alternative hypothesis and cannot conclude that there is a significant effect of sleep on amount of money people spend on food. (it's a marginally significant result)
      - Note this could be because the sample size is too small - increasing sample size to 40 would make a p-value of .39 and reverse this conclusion.
      - **With a too small sample size you're likely to get a non-significant F-statistic** even if there is an actual effect.

## Code: Simple Regression

- use scipy  stats.linregress function for simplest approach:

```python
# input the independent varaible (number hours slept) and dependent variable (dollars spent)
# linregress assumes an intercept for you and you don't have to specify it or pass in a design matrix
slope,intercept,r,p,std_err = stats.linregress(sleepHours,dollars)
```

- [See notebook](../../statsML/regressions/stats_regression_simpleRegression.ipynb)
