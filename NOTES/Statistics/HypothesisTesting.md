# Hypothesis Testing

- Hypothesis: a model comparison, comparing different models
  - Goal is to determine which model is better than the other.
  - Technical definition: A falsifiable claim that requires verification, from experimental or observational data, and allows for predictions about future observations.
- We compare a null hypothesis (representing a control, no effect) and a alternative hypothesis (representing our model and containing one sample from it's population). The comparison tells us if our model is right.

### A good hypothesis:

- Clear
- Specific
- Falsifiable
- Based on prior data or theory
- Leads to a statistical test
- a statement, not a question
- prediction about the direction of an effect (is specific about what should happen)
  - Example:
    - bad: Studying improves grades
    - good: A combination of self study and group study will improve final exam grades by at least 10% (include types of study and specific outcome)
- Relevant for unobserved data or phenomena
- Relevant for understanding nature.

### NULL HYPOTHESIS:

- the hypothesis that nothing interesting is happening in the data. Your manipulation had no effect on the data.
  - In research, you specify the "alternative hypothesis" (can be thought of as a "Effect Hypothesis")
    - Alternative Hypothesis distribution is a repeated sampling of data after applying an effect.
  - In Statistics, on the other hand, you test the Null Hypothesis
    - Does the data support the null hypothesis or not? You hope to find that it does not support it, you reject the null hypothesis in favor of the alternative hypothesis.
  - referred to as "H nought" ($H_0$)
- Example:
  - Alternative Hypothesis: People will buy more widgets after seeing advertisement X compared to advertisement Y.
  - Null Hypothesis: The advertisement type has no effect on widget purchases.
    - In statistics you gather evidence to determine the strength of this null hypothesis, in which case you reject the null hypothesis and TENTATIVELY favor the alternative hypothesis. (the reason is it's not possible to actually prove a hypothesis.)
    - You either reject or fail a null hypothesis or not.
- The Null Hypothesis is analytically derived based on formulas etc., it is theoretical (The alternative Hypothesis is generated from data and is empirical)

### SIGNAL VS. NOISE - The basis of Inferential Statistics

- The goal of hypothesis testing is to quantify some Signal to Noise ratio.
  - The goal of inferential statistics is to determine Signal/Noise ratio (some effect vs. some null hypothesis distribution)
- see [video](https://www.udemy.com/course/statsml_x/learn/lecture/20023396) around 6 minutes
- In Hypothesis Testing, we take the mean of the null hypothesis distribution and the mean of the Alternative Hypothesis distribution and measure the difference.
  - This requires that the distributions be scaled and normalized properly to get a meaningful signal/noise result.
  - Note: In practice, the H0 (null hypothesis) distribution comes from a formula (it is analytical/theoretical) or from data shuffling (in permutation testing, which makes it empirical then), and the HA (alternative hypothesis) is measured in the data.
- We take the null hypothesis distribution (what distribution we expect if the null hypothesis is true)
  - If the null hypothesis is false and the alternative hypothesis is true, we take that distribution that we would expect and compare it to the null hypothesis distribution. To do this we compare the Centers or MEANS of the distributions.
    - **Because we can end up with an alternative hypothesis distribution that is almost the same as or looks slightly different than the null hypothesis distribution due to sampling variability or NOISE, we compare the means**
    - We need the difference of centers of the distributions to be normalized or scaled by some feature of the widths of the distribution.
      $${Difference \: of \: centers \over Widths \: of \: distributions} = {Signal \over Noise}$$
    - The difference of centers could be called the signal, and the dispersion/width could be called noise.

## P-Values

- [video](https://www.udemy.com/course/statsml_x/learn/lecture/20023398#content)
- A p-value (Probability Value) is the probability of observing a data point from your sample data given that the null hypothesis is actually true.
  - If you observe a value in your data that is unlikely to occur given the null hypothesis then you say that is statistically significant effect.
  - a low p-value means there is a low probability of the statistic being observed given the null hypothesis (so it is significant and is evidence of your alternative hypothesis)
- In practice, you will have one value for the Alternative Hypothesis that you will compare with the Null Hypothesis distribution (i.e. where does it land on it?)
- How likely is the $H_A$ value occuring if the Null Hypothesis is true? Or, What is the probability of observing a paremeter estimate of $H_A$ or larger, given that there is no true effect?
  $$p-value = P(H_A|H_0)$$
  - For example, a $H_A$ value occuring within the shallow part of the null hypothesis distribution would be unlikely.
  - We cannot prove $H_A$ is true. We only compute the probability of whether it could be observed GIVEN (assuming) the Null Hypothesis is true!
- P-values are probabilities and range from 0 to 1
  - a p-value of 0 indicates low probability of $H_A|H_0$
  - a p-value of 1 means there is high probability of $H_A$ being observed given that $H_0$ is true.
  - H_A vals that land closer to the center of a null distribution have a higher p-value than those that fall on the outer edges. They increase to 0.5 as they get to the peak of the null hypothesis distribution.

### Statistical Significance

- The probability of observing a test statistic this large given that the null hypothesis is true.
  - Note that the term statistical significance can refer specifically and technically to this approach!
- A finding is statistically significant if the probability of a test statistic is greater than a threshold. If $p(H_A) < p(a)$ where $a$ is the sigficance threshold on the x-axis of the null hypothesis distribution plot.
  - The thresholds are arbitrary
  - sometimes called alpha $a$
  - Common thresholds used are .05 or .01 (probability of 5 or 1 percent of alpha, or confidence level of 95% or 99%)
    - The higher confidence level means we want to be more sure that our findings indicate the null hypothesis is false. i.e. if our CL is 99%, it means we expect to make an error in judging the conclusion of the hypothesis 1% of the time (Type I or II errors)
  - for a result to be considered statistically significant, the p-value should be smaller than the alpha level
- Note the TAILS (each side) of the Null Hypothesis distribution is unlikely.
  - If the Null Hypothesis is true, you're unlikely to get test statistic values ($H_A$ value) in the tails of the $H_0$ distribution.
  #### One-tailed vs. two-tailed thresholds:
  - For example if your threshold is .05 (5%), then you have a one-tailed threshold (only vals that fall on the right 5% of the null hypothesis distribution, i.e. the right tail, are statistically significant)
  - If your threshold is two-tailed, and you want 5%, you need to divide by two to get 0.25 on each tail (to total .5 or 5%)
  - **Whenever possible you want your hypotheses to be one-tailed because that is more specific**
    - Note outside of hypotheses, in statistics you usually do two-tailed tests.
- note that the probability distribution is not always 'normal'. It can be a t-distribution, F-distribution, chi-square distribution, etc. We often visualize normal distributions because they look nice and symmetric and for example.
- Note the term significance can mean different things and not related to pvalues or statistical significance.
  - Theoretical Significance: A finding is relevant for a theory or it leads to new experiments and discoveries - it is not statistical significance or related to p value threshold etc.
  - Clinical Significance: a finding is relevant for diagnosing or treating a disease.

### Mis-interpreting P-values

- The p-value is not an indication of the percent of data in the population that your alternative hypothesis has an effect on:
  - Wrong: My p-value is .02, so the effect is present for 2% of the population.
  - Wrong: My p-value is .02, so there is 98% chance that my sample statistic equals the population parameter.
  - Correct: My p-value is .02, so there is a 2% chance that there is no effect (iow, the null hypothesis is true) and my large sample statistic was due to sampling variability, noise, small sample size, or systematic bias (something other than a real effect actually being present).
    - IOW, there is 2% chance that your $H_A$ result was the result of no effect on the population and you got your observed statistic by chance. So the Alternative Hypothesis is most likely better than the null hypothesis.
  - Wrong: My p-value is smaller than the threshold, so therefore the effect is real.
    - The p-value is not telling us if the effect is real, it is just telling us whether it is likely or unlikely to observe a sample statistic as large as we found in our data given that the null hypothesis is true.
  - Correct: My p-value is smaller than the threshold, so it is unlikely that the effect in the sample would have been observed given the null hypothesis is true, assuming that the sample is representative of the population.
  - Wrong: My p-value is larger than the threshold, so therefore the null hypothesis is true.
    - The p value is a probability and it does not hold that a high p-value means that the null hypothesis is necessarily true.
  - Correct: My p-value is larger than the threshold, so it is likely that the effect in the sample would have been observed given the null hypothesis, assuming a representative sample. There could be other explanations for the p-value though other than $H_0$
    - The data observed is consistent with the null hypothesis, but there could be other explanations for the p-value.

### Common P and Z values, or P-Z Pairs (good to memorize)

- Poportions For Gaussian Distributions:
  - 68.3% percent of the data is contained in two std devs around the mean (one on either side -1 to +1 std dev).
  - 95.5% of data falls within 2 std devs on either side of the mean
  - 99.7% of data falls with 3 std devs (very rare to be more than 3 std devs from the mean)

#### P-Z pairs for a one-tailed test:

- `p = .05 <=> z = 1.64`
  - if we take .05 as the threshold, finding a p-value of .05, then the observed data is 1.64 std dev from the mean of the null hypothesis distribution
- `p = .01 <=> z = 2.32`
  - if we have a p-value threshold of .01, then the observed test statistic must be more than 2.32 std dev away from the mean of the null hypothesis distribution in order for it to be considered statistically significant.
- `p = .001 <=> 3.09`

#### P-Z pairs for two-tailed tests:

- Note: generally in tests, you want to do a two tailed test unless special reason to do a one tailed test (they are more conservative and do not assume directionality).
- `p=0.5 <=> z=+/-1.96`
  - the statistic observed must be 1.96 std dev from the mean of the H_0 distribution on the right or left of it.
- `p=.01 <=> z=+/-2.58`
- `p=0.001 <=> z=+/-3.29`
  - to reiterate, if you choose a 0.001 p-value threshold to determine statistical significance, then the observed data point must be more than 3.29 std dev on either side of the mean of the null hypothesis distribution in order to be considered statistically significant.

## Statistical Errors

- Errors occur when we have false negatives or false positives when doing hypothesis testing for example.
- True/False positives: A positive result is when the alternative hypothesis tests positive and is favored, negative is when alt hypothesis is not favored and we accept the null hypothesis

### Types of Errors

- Type I Errors: False Positive - occurs when the null hypothesis is actually true, but we get a very large test statistic (i.e. an outlier for ex.) sample by random chance, so it's past the threshold and we incorrectly favor the alternative hypothesis mistaking it for signal. We fail to identify a real effect.
- Type II Errors: False Negative - the test statistic was smaller than the significance threshold, so we don't reject the null hypothesis in favor of the alternative hypothesis. But, this can be observed when the alt hypothesis is actually true.

  - Can occur when the null and alt hypothesis distributions overlap significantly and a test value is selected which falls in the overlap (so it doesn't make the threshold to prove the alt hypothesis, but it still falls within relative likely ranges of both the null and alt hypothesis distribuition curves.)
  - We just didn't have enough evidence for the alt

### Avoiding errors

- To reduce Type I errors: increase the threshold (alpha) so that more results will fall under it and the null hypothesis will be favored. This reduces the chance of identifying the alternative hypothesis as what's true.
  - example set threshold pvalue to .001 instead of .05
  - The tradeoff is you're reducing the possibility of having true positives (when the alt hypothesis is actually real and true, but you don't accept it due to the shifted threshold)
- To reduce Type II Errors: Move the threshold to favor the alternative hypothesis more over to the left on the x axis for the null hypothesis to prevent more False Negatives.
  - Tradeoff: you increase the likelihood of Type I errors and could get False Positives that incorrectly identify the alt hypothesis as true when it's not.

#### How to generally decrease chances for error:

- Minimize the overlap between the null hypothesis and alternative hypothesis distributions.
- Two ways to do this:
  - Increase the distance between the distributions, moving the alternative hypothesis distribution further to the right or left (null hypothesis is generally fixed in position)
    - Look for very large or experimental values and bigger effects that you expect to be very different from the null hypothesis value
    - Compare samples that are very different - i.e. giraffe heights vs. monkey heights instead of north african giraffes vs. south african giraffe heights.
  - Decrease the width of the distributions
    - Have less variability in the data so the dispersion is tighter and the distributions are less wide.
    - You can make the null hypothesis distribution narrower by changing the degrees of freedom. The larger the sample size, the narrow the distributions become.

## Parametric Statistics vs. nonparametric statistics

- Should prefer parametric statistics over nonparametric statistics
- Nonparametric statistics allow us to relax assumptions: i.e. that distributions are Gaussian, or the assumption of the absence of outliers.
- Nonparametric statistics also include statistical inference methods that generate the Null Hypothesis distribution from actual data, instead of from an equation.
- nonparametric does not mean no parameters.

### Parametric Methods:

- Widely used and standard
- Based on assumptions (guassian distrib., no outliers, etc. violations can effect the results negatively)
- Can be incorrect when assumptions are violated
- Computationally fast
- Analytically proven

### Nonparametric Methods:

- Some are nonstandard methods
- No assumptions necessary
- Can be slow, inefficient
- can be algorithms over proven methods
- Appropriate for non-numeric data
- Appropriate for small sample sizes
- Some methods can sometimes give different results each time

## Multiple Comparisons and Bonferroni Correction

### Family Wise Error Rate (FWE or FWER)

- The probability of at least one hypothesis or test in a set of multiple hypothesis tests is a false alarm (a type I error false positive).
- FWE is impossible to calculate accurately because it assumes that the tests are 100% independent of each other (in order to add up the probability/pvalues of each to get the collective type I error potential). If the tests are dependent on each other, you can't do this.
- Because it's not possible to know the true FWE, we estimate the MAXIMUM FWE:
  $$FWE \le 1 - (1-\alpha)^n \le n\alpha$$
  - This formula states that the true FWE is probably less than or equal to the terms $n\alpha$ (alpha is the probability threshold in use (ex 5%/.05) and n is the number of tests), or $1 - (1-\alpha)^n$ (1 minus the leftover from alpha (ex. .95 if alpha is .05) ot the power of the number of tests in the set)
  - Note that, given an alpha of .05, if you are running 20 or more tests you are gauranteed to get at least one false alarm. (this is how to interpret the last term $n\alpha$)
  - as you increase powers of n then the first term $1-(1-\alpha)$ approaches 1.

### Bonferroni Correction

- A more conservative formula/solution to the problem that arises when you run more and more tests (the FWE potential increases).
- Most commonly used correction for multiple comparisons
- Set a new threshold to be the $\alpha$ (current threshold of the test) divided by the number of tests in the set:
  $$\alpha_{new} = \alpha/N$$
- where alpha is the threshold and N is the number of tests
  Example:
  $$p(H_1|H_0) = .05/3$$
  $$p(H_2|H_0) = .05/3$$
  $$p(H_3|H_0) = .05/3$$
  $$.05/3+.05/3+.05/3=.15/3=.05$$
  This would say that the probability of a false alarm (type I error) for all these tests is 5%. Each test is about the test statistic given the null hypothesis.
  The individual probability for each individual false alarm test for error would be .0167 (This makes the individual test more stringent)

## Cross Validation

- Dividing data into a training and test set (In sample or out of sample data) and validate the model you used with the training set on the test set to compare results.
- Comparing training and testing data results is done in one iteration. You continue to iterate choosing different sets of data for the training or testing datasets each time.
  - Example: take 10% of the data for testing and 90% for the training data. Keep shifting/choosing the next 10% for testing data on each run until all of your data has been used for both training and testing.
  - **K-fold cross validation**: K is the number of iterations you do to cover all the data. 10 to 20 is a typical number for K - this is arbitrary and can change based on tinkering. (this would mean using 5-10% of data for testing on each run)
- Avoid Biases by making sure that the data for the testing set is selected randomly from the entire dataset.
  - Note: it can be tricky to make sure that the test set is independent of the training set in some scenarios even if you randomly select and separte the data (ex: timeseries where there is strong correlation between points builtin - even if you select random samples, they will by nature be correlated due to the builtin correlation in the dataset type)

### Uses for Cross Validation

- Useful for use in confidence intervals and Jack-Knifing
- good for avoiding biases in results and checking for overfitting (If model is overfitting in the training data, you will see reduced performance in the test data set - models will do better on the test set when they are not overfitting the training set.)
- Used to compute Classification Accuracy (esp in Machine learning and Deep Learning validating test sets against training sets)

## Classification Accuracy

- Classification is an alternative approach to hypothesis testing from p-value approach (using null hypotheses and alternate ones etc.) - we compare predictions with real outcomes and validate against training/test data.
- Classification is mainly used on categorical data (nominam or ordinal data)
  - It's possible on discrete data as well, though less common
  - It is not done on interval or ratio data due to arbitrary precision and to many fine grained categorization (continuous values means there could be infinite classifications)
- Classification Accuracy: compare model predictions against actual outcomes.
  - If a model got 3 out of 4 predictions correct against the actual data, then we say the accuracy is 75% for example.
  - This measuring is robust to sample size.
  - As opposed to p-value approach, where you can take each parameter in the model and compute their probability of being correct/test them separately, with classification accuracy usually the entire model is considered, not individual params.
