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
    Beta0 # vector of beta coefficients (unknowns we want to solve for in model fitting) - each of these line up with a particular IV parameter
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
        162 162 1 # note the first two cols are always the same numbers and therefore highly correlated.
      ]
      ```

#### Be suspicious of high correlations between Independent Variables in the design matrix:

- Very strong correlations in the design matrix indicate that there are redundancies in the model. That will decrease the fit of the model to the data (because the design matrix is less numerically stable). Strong correlations don't necessarily mean that the model is flawed, but it is an indicator that something might have gone wrong with the model specification, and you should look into it.

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

- use scipy stats.linregress function for simplest approach:

```python
# input the independent varaible (number hours slept) and dependent variable (dollars spent)
# linregress assumes an intercept for you and you don't have to specify it or pass in a design matrix
slope,intercept,r,p,std_err = stats.linregress(sleepHours,dollars)
```

- [See notebook](../../statsML/regressions/stats_regression_simpleRegression.ipynb)

## Multiple Regression

- A regression with multiple parameters - multiple independent variables
- Example:
  $$y = \beta_0 + \beta_1 s + \beta_2 h + \beta_3(s * h) + \epsilon$$
- Do students' grade on a stats examp depend on sleep (s), hours studied (h) and the interaction between them (multiply sleep by hours studied) s x h? N=30 students
  - To implement an interaction term in regression you just multiply the two IVs together.
  - Note that you can square IVs if you want them to have more importance, but you cannot square Beta coefficients, Ex: $\beta_2 h^2$ - the square only applies to hours (h), not the beta coeff

### The Regression Table

- Similar to ANOVAs you can make a table for Regressions:
- note the ddof is N-k where k is the total number of parameters (not the same thing as independent variables - the parameters include the intercept/constant)
  - The intercept is sometimes called the Constant

| Source of Variance  | Beta Coef | Std. Error    | t-value | p-value |
| ------------------- | --------- | ------------- | ------- | ------- |
| Constant(Intercept) | 60        | .3 (N-k ddof) | 200     | .000    |
| Study Hrs           | 4.5       | 1.2           | 3.75    | .001    |
| Sleep Hrs           | .8        | .33           | .33     | .373    |
| Study x Sleep       | 5.4       | .8            | 6.75    | .000    |

- Shows that there is a Main Effect of Study Hours (small pval), no main effect of hours slept, but a significant effect of the interaction.
- Note that the above shows that Sleep Hours is not significant (p-value > 0.5), but it is still somehow important because it interacts with Study Hours.
  - This is similar to how in an ANOVA you can have a non significant main effect, but an interaction with it that does have an effect.

### Visualizing the data in Multiple Regression

- We need multiple axes with more than 2 parameters (like a simple regression)
- To avoid plotting multi-dimensional graphs, we can instead discretize one of the variables (the regressors) and bin it into a number of ordinal categories (use a small number, 2 or 3)
  - Example: treat the continuous variable of study hours as ordinal and bin it into 3 parts: small amount of study(2-4 hours), medium amount of study(5-6 hours), large amout of study (7-8 hours).
  - see [video](https://www.udemy.com/course/statsml_x/learn/lecture/20225704#content) at timestamp 8:10 for visual example.
  - In the above example with no main effect, after we average the three bins together and get a flat line with a slope of 0

### Interaction Terms in Multiple Regressions

- The more parameters you have, the more higher-order interaction terms you can have
  - i.e. with 3 IVs, getting all the possible interactions would be 4 permutations resulting in 8 parameters total
  - Avoid introducing multicolinearity
  - Avoid having too many higher-order interaction terms and aim to have as few as possible unless necessary.
  - If at least one variable is binary this does help when you have multipole interaction terms.

### Interpreting a Beta Coefficient

- The beta coefficient ($\beta$) measures the effect of a change of one unit of the independent variable it is associated with ($x$) in $\beta x$, on the dependent variable $y$.
  - IOW, the effect of a change in parameter x on the y-axis
  - Example: If $\beta_1 s$ where $\beta_1$ is .167 that means that for every change of 1 unit in sleep (for example 1 unit is 1 hour), the y-axis or dv is changed by .167
- In a Multiple Regression: the beta coefficient represents the above **when all other variables are held constant**.
  - When we change parameter $x$ we fix all the other parameters in the model (the other parameters do not vary and we shift the model by one unit in $x$ for example)

## Standardizing Regression Coefficients

- Unstandardized Beta Regressions: The raw output (the beta coefficients you get) of the regression model are unstandardized.
- Beta coefficients are scale dependent
  - Example: $\beta_1 s$ measuring the parameter of amount of sleep in test score performance could be measured in terms of hours (1 unit of change is 1 hour) or minutes etc.
  - If $\beta_1$ is .167 that means that for every change of 1 unit in sleep (for example 1 unit is 1 hour), the y-axis or dv is changed by .167
    - If the unit of $s$ was minutes however, then $\beta_1$ would become 10 (because to keep the same effect calculated with hours each minute needs to have an effect of 10x, i.e. 60 \* .167 = 10), if in seconds, then $\beta_1$ becomes 600.
- It's difficult to interpret beta coefficients when they are unstandardized across variables
  - If you have one beta coeff that is in hours or seconds and another that is in calories, you're trying to compare apples to oranges. There's no good way to look at the raw coeffs and see quickly if one is stronger in effect relative to the other.

#### CAUTION: The first thing you need to think about when looking at unstandardized Beta Coefficients (the default in a Regression table for ex.) is to be careful about comparing them

- If they are in the same scale then it is fine, but since they are unstandardized in tables by default, the first thing you should do is confirm this, and standardize them if needed if you want to do a direct comparison between beta coefficients.

### Standardization of Beta Coefficients

- Standardization means that we have the same beta coefficients regardless of the original units of the data. i.e. the units of the parameters do not matter.
- Standardizing coefficients is specifically useful when wanting to compare the effects relative across multiple indepentent variables or parameters or models. Otherwise using unstandardized coefficients helps interpret the effect of a single parameter.
- Standardized Coefficients are in standard deviation units which are unrelated to the scale of the original data.
- You can use unstandardized or standardized beta coefficients depending on what you want to interpret (single params vs. comparing across params/models), and using either has no effect on the underlying statistics - does not change p-vals, t-values, r squared etc.

#### How to standardize Betas:

- Noramilze the Beta Coefficients so that they have a variance equal to 1. (similar to Z-score normalization)
- Two methods:
  - z-normalize the entire dataset (the dependent and independent variables) BEFORE you run the regression.
    - standardized and unstandardized betas will both be in the same scale with this method
  - Recompute Beta values standardized by the standard deviations of both the independent variable and the dependent variable, but leave the original data alone.
    $$b_k = \beta_k{s_{x_k} \over s_y}$$
    - Where $\beta_k$ is the unstandardized raw beta coefficient multiplied by the standard deviation $s$ of the independent variable for the corresponding parameter ($x_k$ is data point $x$ and $k$ is the independent variable) divided by the standard deviation $s$ if the data $y$.
    - We normalize by the standard deviation of the Dependent variable and the relevant independent variable.
    - stats software handle this formula for you and include both standardized and unstandardized regression coefficients in a table for you.

### Interpreting a standardized Beta Coefficient:

- A standardized beta coefficient reflects the effect of one standatd deviation change in $h$ (DV) on how much standard deviation changes occur in $y$ (the IV) **when all other variables are held constant** (as in all multiple regressions)
  - No longer are we reflecting a change in one unit as in unstandardized betas
  - The "units" of change now depend on the data distribution or variance
  - Example if $\beta_2$ for food intake is .8, that means that for every standard deviation of increase in food eaten, that would give you .8 standard deviations higher score on the dependent var $y$.

## Code: Multiple Regressions

- Can use the LinearRegression fn from scikit learn

```python
from sklearn.linear_model import LinearRegression

```

- see [Notebook](../../statsML/regressions/stats_regression_multipleRegression.ipynb)

## Polynomial Regressions

- A polynomial has the form of the data $x$ to increasing exponents
  $$a_0 + a_1x + a_2x^2 + ... + a_nx^n$$
  - starts at $a_0$ which is like multiplying x to the zeroth power. Then exponents on x continually increase.
  - x has coefficients to give each term some weight.
  - Examples: $3 - 5x + 1.5x^2 + 12.4x^3$ or $2x + x^3$ (you leave out the x terms multiplied by 0)

### Order of a polynomial

- The order of a polynomial expression is the largest exponent - i.e. the n-th order polynomial
  - $3 - 5x + 1.5x^2 + 12.4x^3$ is a 3rd order polynomial since the largest exponent is 3

### Formula for a Polynomial Regression

$$y = \beta_0 + \beta_1x + \epsilon$$

- This is a first order polynomial regression ( largest exp of x is 1)
- $y = \beta_0 + \beta_1x^1 + \beta_2x^2 + \epsilon$ is a second order polynomial regression

### Fitting nonlinearity in Polynomial regressions

- Since x is exponentially increasing there is nonlinearity involved, but the Beta coefficients are all linear (regular multiplication).
- Polynomial regression is actually a linear model due to the above (doing nothing nonlinear with the coefficients, just the data is being raised to higher powers)

### When to use a Polynomial Regression

- When a straight line is not a good fit to a dataset and there are curves to the fitting needed.
- See [video](https://www.udemy.com/course/statsml_x/learn/lecture/20225716#content) at timestamp 4:38 for visual

### Determining Appropriate Order - Bayes Information Criteria (BIC)

$$BIC_k = n{\ln}(SS_\epsilon) + k{\ln}(n)$$

- Used to determine how many orders to use for polynomial regression for best fit.
- Can compute this for every model order (up to $k$ model orders)
- $n$ is the number of data points you have, multiplied by natural log of the Sum of Squares of the Residuals ($\epsilon$) plus $k$ number of parameters times natural log of datapoints $n$
- Run through a bunch of model orders and fit them all. Then look at a plot of the BICs (plot the BIC (on y-axis) as a function of the polynomial order (on x-axis))
  - Look for the minimum point (smallest BIC on y-axis) in that plot and that is the optimal model order to use. (look at the x-axis for the order to use)
- We try to find the smallest BIC to get a good fit to the data without overfitting (as orders/params increase, so does the R-squared which can be misleading)

### Code: Polynimial Regressions

- See [notebook](../../statsML/regressions/stats_regression_polynomialRegression.ipynb)
- Use numpy functions `polyfit` and `polyval`

```python
# polyfit: first arg is IV, then dependent variable(y dataset), then specify the order of polynomial
  # we know the order is 2 since we made the data, but you can use the BIC to calculate it (see further below cells)
pterms = np.polyfit(x,y1,2)
print(pterms) # we get 3 nums: x^2 term, x^1 term, x^0 (intercept term) - NOTE THE ORDER!!!

# We now use the polyfit terms in the polyval function to evaluate the polynomial
   # input the coefficients and the independent variable
# this gives us the predicted data
yHat1 = np.polyval(pterms,x) # evaluate a polynomial
```

## Logistic Regression

- a.k.a. Binary Logistic Regression
- Has a dependent variable that is logical (i.e. a boolean) and binary (2 possible outcomes)
  - True/False
  - Male/Female
  - Has tumor/ Does not have tumor
  - Win/Lose
- **Does NOT classify**: The output is not True/False or Pass/Fail etc. It returns probabilities of category membership (i.e. of different events occuring, i.e. a probability of passing or failing)
- After running a logistic regression to get the probabilities we then need to do some classification by implementing a threshold (i.e. if regression says probability is greater than 50%, r>0.5, then we consider that to be a pass or true result etc.).

### Multinomial Logistic Regression

- Extended to any number of categorical outcomes (i.e. more than 2 like the logistic regression)
  - cat / penguin / truck
- What Deep Neural Networks implement for things like image recognition/categprization

### Formula

$$\ln{p \over 1-p} = \beta_0 + \beta_1x_1 + ... + \beta_kx_k$$

- The right side is a normal regression
- The left side is the different part: The natural log of the ratio of the probability of the event happening to the probability of the event not happening.

### Solve for p to get the probability

- We really want to solve for $p$ to get the probability.

  - use the natural exponent on both sides (natural exp cancels out ln natural logarithm)
  - Fuller explanation in [video](https://www.udemy.com/course/statsml_x/learn/lecture/20225732#content) at timestamp 6:25
    $$p = {1 \over {1 + e^{-{\hat{y}}}}}$$
  - Where $\hat{y}$ y-hat is the right side of the regression formula above and $e$ is the natural exponent

- We use the log because it has a larger dynamic range when dealing with small values (i.e. $1 \over 1-p$).
  - See [video](https://www.udemy.com/course/statsml_x/learn/lecture/20225732#content) at timestamp 7:23
  - Makes it easier to work with these small values in optimization problems. Increasing the dynamic range helps the model find the right parameters to get the best fit for the data.

### Finding the best regression coefficients

- Because there are nonlinearities in the formula (division, betas in the exponent, stuff other than just multiplication and addition etc.), it's not possible to use the left-inverse and linear algebra as we would in other General Linear Models.
- Instead we need to use iterative methods such as **Gradient Descent** to find the set of parameters (Beta Coefficients) that make the probabilities output (predictions) best match the Dependent Variable (actuals)
  - Start off with random Beta Coefficients, and see how well does the equation produce probabilities that match the outcomes. Keep repeating different parameters to find a good fit.

### Example of a Logistic Regression

- Research Question: Does the amount of sleep and number of study hours predict PASSING an exam?
  - The main difference here that makes this appropriate for a logistic regression is that we change the Dependent Variable from being the exact score that people get, we just want to predict a binary outcome of pass or fail.
- Experiment: Ask 20 students (N=20) to report the number of hours slept and number of hours studied. Then look at their grades and label it as a pass or fail outcome.

### Visualizing a Logistic Regression

- Since the model is outputing probabilities, we can set up a plot in the following way:
  - see [video](https://www.udemy.com/course/statsml_x/learn/lecture/20225732#content) at timestamp 15:00
  - The individuals are on the X-axis (i.e. N=20, so 20 ticks)
  - Group the individuals into the categorization of the test - i.e. 10 students who passed are in one group and the 10 students who failed are in another group (the first 10 ticks are pass, second 10 ticks on x-axis are failed)
  - On the y-axis plot the output of the logistic regression model (the probability values for each individual whether they passed the exam)
  - You need to establish a threshold (i.e. 0.5 probability above is passing, below which is failing)
  - Determine concordances (correctly predicted outcomes) and errors (incorrect predictions) and determine an accuracy, i.e. 0.75 or 75%.
  - This framework is called **PREDICTIVE MODELING**
    - Use a logistic regression and make a threshold and count the accuracy rate

## Code: Logistic Regression
- see [notebook](../../statsML/regressions/stats_regression_logisticRegression.ipynb)
- use the function that comes from scikitlearn:
```python
from sklearn.linear_model import LogisticRegression
```