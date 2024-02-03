# Overfitting

- models with too many parameters (i.e. in the limit if the number of parameters equal the number of data points in the actuals)

### Why Overfitting is bad

- Violates the spirit of statistics - the goal is to fit data with a model that is as simple as possible. Models that overfit are over complicated
- Models that over fit will be capturing and fitting to noise (some of the data points will be noise)
  - overly sensitive to noise
- Generalizability - overfitted models will not generalize to new data.

### Good about overfitting

- increased subtle effects

## Researcher Degrees of Freedom (Hidden overfitting)

- A hidden way of overfitting
- The researcher has many choices for how to clean, organize and select the data and which models and how many models to run.
  - Each time you make a choice you're testing a new model of the data

# Underfitting

- models with too few parameters that fail to capture the complexity of the data
- Underfitting is less sensitive to noise (more robust to noise)
- Underfitting can result in good results with less data
  - with too little data better to have too few than too many parameters (still undesirable)
  - lean on the side of underfitting with small datasets

### Example of Researcher degrees of freedom

- Models A,B and C are being tested. The researcher is disatisfied and goes back and cleans the data and re-runs the tests.
- This results in 6 tests (a,b,c for each set of cleaned data)
- The actual p-threshold or alpha threshold is: .05 < a < .05\*6 = .3
  - The reason the alpha is less than .3 is because the 6 tests are not independent of each other. We are working with the same data and the models are probably related with dependencies.
  - We don't know the exact alpha threshold, but due to the above it is probably greater than the assumed 0.5
  - So if a t-statistic or test result had a 5% probability of falling in the null hypothesis distribution, then it could still be statistically significant due to an actual threshold that is higher than .5 due to dependencies in the more number of tests
- if you look at the results, then go back and change how you select the data to try to change the result, then that is researcher overfitting.

## How to know what the correct number of parameters are

- Use a combination of intuition, critical thinking and formal model comparisons
- With 1 to 2 dimensions in a dataset: Visualize the data and make a reasonable guess
- With 3+ dimensions: You will need to use Formal model comparisons and things like the Bayes Information Criteria etc.

## How to Avoid Overfitting

- Decide the complete analysis pipeline in advance (before data collection). Any deviations from this plan must be clearly stated (ex: pre-registered report)
  - Make decisions before looking at the data: this is how I'm going to clean the data, this how I am going to sort and filter the data, this is the analysis I'm going to run, these are the parameters we're going to use, this is our statistical threshold. (this is like a Pre-registered report where you write the research paper before you collect the data)
    - You are allowed to deviate but you must document what and why so others can evaluate if the change will result in reducing generalizability to new datasets.
- You adopt a exploration/experimentation phase to develop and try out different analyses and models on a **sample** of data (i.e 5 or 10% of the data). Then once decisions on which parameters to use etc. apply that pipeline to the rest of the data. (similar to training/testing sets from cross validation)
  - This is a common way to avoid overfitting in practice. Trying a model pipeline on a sample and then trying it on the rest of the data without deviations.

## Comparing Models

- Determining whether a model is worth having more parameters and being more complex than a simpler model.

### Nested Models

- A model that contains some identical predictors to another, and the Dependent Variable is the same.

- A model that is nested inside another:
  $$y = \beta_0 + \beta_1s + \epsilon$$
  $$y = \beta_0 + \beta_1s + \beta_3(s*h) + \epsilon$$

- The top model is the "Reduced Model" (nested)
- The bottom model containing the nested is the "Full model" (extended)

- It is best practice to have nested models when doing comparisons to differ by only one parameter.
  - with more differences it gets very difficult to determine which parameter is causing a difference in comparisons

### NOTE On Fitting

- The Full Model will always fit the data better than the reduced model. This is the nature of adding more parameters to a model - it will always make for a better fit
- The question is whether it is justified to add extra parameters.

- When testing/comparing we want to penalize the models with more parameters and conclude that adding more parameters is justified only when the additions REALLY improve the model and fit to the data.

## F-test for model comparison

- The quantity we use is going to be the Sum of Squared Errors
  $$SS_\epsilon = \sum(y_i-\hat{y}_i)^2$$
  - sum of (actual individual data point minus the prediction for it) squared

### Formula for F-test in comparisons

$$F(p-k,n-p-1) = {{(SS_\epsilon^R - SS_\epsilon^F)/(p-k)} \over SS_\epsilon^F/(n-p-1)}$$

- This is an F ratio with numerator and denominator degrees of freedom ($(p-k,n-p-1)$)
- $p$ is the Full model number of parameters (number of beta params not counting epsilon)
- $k$ is the reduced(nested) model number of parameters (i.e. the number of beta coefficient parameters)
- $SS_\epsilon^F$ is the Full model sum of squared errors
- $SS_\epsilon^R$ is the Reduced model sum of squared errors
- For understanding the formula, ignore the $p-k$ in the numerator and ignore the entire denominator (it is just a normalization factor to bring the result to a ratio in order to evaluate the F value in the scale of the data)
  - The important part is the subtraction $(SS_\epsilon^R - SS_\epsilon^F)$ in the numerator
  - Note that when the model fits the data better, the Sum of Squares gets smaller. We expect the above term in the numerator to be positive because the full model will fit better and be a smaller number than the reduced model with less parameters and worse fit (larger SS_e)
- When the Sum of Squares Errors for the Full and Reduced models are similar the F ratio will be closer and go towards 0
- When the Full Model Sum of Squares Errors is large (really good fit to the data), then the F ratio will get larger
  - The SS errors for the Full model is small relative to the SS errors for the Reduced model

### Interpreting the F statistic

- When we have a large F statistic (how large depends on the degrees of freedom p-k and n-p-1), if the F ratio is statistically significant IOW, that means that more parameters improved the model and we should prefer the more complex model.
- If the F ratio is not significant, then the model with fewer parameters fits as well as the more complicated one. Prefer the simpler model.

### Null and Alternative Hypotheses for the F test in comparisons

$$H_0 = \beta_{k+1} = ... = \beta_p = 0$$
$$H_A = \text{At least one } \beta_{k+1:p} \ne = 0$$

- The null hypothesis is that all the extra parameters contained in the full model that are not contained in the nested model are all zero.
- The alternative hypothesis is that at least one of those extra parameters in the full model is not equal to zero
  - For this reason it is difficult to know which parameter is driving the model improvement if you have more than one parameter difference

### Running comparisons in practice

- You can compute the ANOVA or Regression tables with statistics libraries (i.e. how it's done in python) and get the Sum of Squares Errors from those tables (that way you don't have to compute those yourself and you just plug them into the formula above for the F-test)
