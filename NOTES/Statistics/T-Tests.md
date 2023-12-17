# T-Tests

Used very commonly in Statistics.

## What is a T-Test?

- The main idea is to compare two means from two different groups
  - Example: The alternative hypothesis would be that the mean of group A is different from the mean for group B. (The null hypothesis of a T-test would be that the means of two groups are equal/not different. In other words, you could shuffle the data from each group into the other and still get the same means afterwards.)
  - It's possible to test a group of data against an empty group (implicitly it means we are testing for a mean of 0). i.e. we test whether a vitamin makes people happier (yes or no) - we test that the group with the vitamin has an effect/mean greater than zero.

### General Formula

$$t_k = {\bar{x} - \bar{y} \over s / \sqrt{n}}$$
or you can rewrite this to eliminate division in the denominator:
$$t_k = {{(\bar{x} - \bar{y}) * \sqrt{n}} \over s}$$

- Most t-test formulas will look similar to this
- $\bar{x}$ is the average of group 1 and $\bar{y}$ is the average of group 2
- $s$ is the standard deviation of the sample
- $\sqrt{n}$ is the square root of the number of samples $n$
- What this generally measures is the diff of the means scaled or normalized by standard deviation: $\text{Difference of Means} \over \text{Standard Deviations}$

### Testing

- Example test output for t-value and graph in [video](https://www.udemy.com/course/statsml_x/learn/lecture/20025050) at timestamp 6:40
- Because of sampling variability, there will be a range of T values with a probability distribution that you could expect even if the null hypothesis were true. (on the y axis of a plot you'll see: $P(t|H_0)$, i.e. the probability of a t value occuring given that the null hypothesis is true)
  - If the null hypothesis is true, then we expect $\bar{x} - \bar{y}$ to be zero.
  - The distribution of range of T-values represents our null hypothesis range/probability (that it's true there is no difference in the group means)
- The T-value is derived from the above formula and is represented on the x-axis of a plot (example, a T-value of zero means that the null hypothesis is true since the difference in means of the two groups is zero.)
  - Because there is random noise and variability in the data groups, the t-value could be a range of values - this range is a distribution that we can expect given that the null hypothesis is true (0 would be the most probable t-value/peak in this distribution)
  - To test, we compare the t-value from an actual measurement against the real data (the distribution above is just analytical) against the range of T-values distribution given the null hypothesis is true.
  - P-value: When plotting the t-value $t_k$ on the x-axis, we ask "what is the probability of observing this t-statistic or greater, given that the null hypothesis is true?" - if not statistically significant based on a threshold, then it means that the t-value does not indicate or provide sufficient evidence that the means of the two groups are different from each other.
    - The larger the t-values are (closer to left/right edges on x axis), the smaller the p-value will be.

### 3 ways of getting a large T-statistic vals

- If we have large group mean differences
- If there is a reduction of variability in the data set (small standard deviation $s$)
- Increasing the sample size/collecting more data is going to increase the value of the t-statistic (if we increase $n$ we get a larger $t_k$)

#### Why do we want to try to get larger T-statistics?

- We usually want the t-value $t_k$ to be large in order to find a statistically significant effect.
- If we're dealing with noisy dataset groups with high variability, then we want to look for effects that are large in magnitude so we don't mistake the noise for indications that the alternative hypothesis is correct when it's not.
  - Example: If we're studying a group of patients between the ages of 60 and 70 with a certain condition, the sample size $n$ is going to be small and hard to find people fitting the requirement. So we need to make sure that either the effect size is large ($\bar{x}-\bar{y}$), or the variability is small (smaller $s$), since we can't increase our T-value based on $n$.

# Types of T-tests

## One-sample t-test

- Testing whether a set of numbers could have been drawn from a distribution with a specific mean.
- We do not test multiple groups - there is just one group under test with no alterations to the samples of group, it is a one shot experiment.
  - Example application: We have **one** group of students and we want to test whether the IQ of the group is significantly different from 100. If the sample has a IQ higher than the population avg of 100, then the sample can be said to be above avg intelligence, and if lower than 100, the sample is said to be below avg intelligence.
  - Formally stated: Estimate the probability (of the null hypothesis) that a certain group of students has an average IQ of 100.

### Formula for one sample t-test:

$$t_n-1 = {\bar{x}-\mu \over s/\sqrt{n}}$$

- The t-value with $n-1$ degrees of freedom is equal to the observed mean of the sample $\bar{x}$ minus the null hypothesis mean (the population mean $\mu$) we're testing against, divided by the standard deviation of the sample data (the std dev of IQ tests across the sample group of students). $n$ is the number of sample data points.
- The reason we have $n-1$ degrees of freedom in the one sample t-test is because we are testing for the mean ($\bar{x}$) and because the mean of a sample, once we compute it, will have n-1 degrees of freedom. (see [degrees of freedom](./DegreesOfFreedom.md))
  - For example, Imagine we have a sample with N=10 and we are testing the hypothesis that the mean is 100. So you don't need to know all 10 values; you only need to know 9, and the final value you can compute based on the 9 values and the predicted mean.
- Note that $\mu$ is the population mean that you are testing against. So for example, if the average IQ of the pop. in the example above was set by parents, then it will likely be set higher than 100 due to bias. So if it is set to an expected value of 120 instead of 100, then the one sample test would reveal that the samples of kids are not as high IQ as originally thought.

### Assumptions of a one-sample t-tests (and two-sample t-test)

- Data is assumed to be numeric (ideally interval or ratio scale data, discrete is usually okay as well at the cost of a little precision, but alright if enough values are used.)
  - One sample t-tests are not appropriate for categorical data
- Data is independent from each other (there should not be dependencies in the sample)
  - Data is randomly drawn from the population to which the generalization should be made.
- The data under test is roughly normally distributed (the mean and std dev are valid descriptors of central tendency and dispersion)

### Code: One Sample T-test

[see Code in Notebook](statsML\ttest\stats_ttest_oneSampleT.ipynb)

## Two-samples t-test

- Purpose is to test whether two sets of numbers vould have been drawn from the same distribution (the null hypothesis we are testing against).
- Example
  - Informal statement (what you would say to non-statisticians): Test whether self-reported stress levels changed after 6 weeks of social distancing.
  - Formal statement (precise statement to statisticians): Estimate the probability that self-reported stress levels before and after 6 weeks of social distancing were drawn from the same distribution. (note: the same distribution means the null hypthesis distribution)

### Formula

- The formula can change depending on:
  - whether the groups are paired or unpaired: whether two groups of data are drawn from the same or different individuals
    - Paired example: Always testing against the same individuals: the same individuals self-report their stress levels before and after social disancing
    - Unpaired example: testing against different individuals in the two groups: looking at change in stress from social distancing between two different countries or two different people/individuals
  - have equal or unequal variance: whether the two groups have roughly equal variance. (if groups are similar or different significantly from each other in variability)
  - have matched or different sample sizes: whether the number of data points in both groups are the same (applies only to unpaired groups, obviously paired samples will have the same number of data points/same individuals)
- Note: in practice these formulas are taken care of by the python libraries you use. The important thing you need to know is the characteristics of your data (i.e. are they paired/unpaired, variance differences, etc.)
