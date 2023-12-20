# Confidence Intervals

- The probability that an unknown population parameter falls within a range of values in repeated samples.
  - Tells you how confident you can be that a true population value of the parameter lies within your confidence interval
  - More specifically it is how confident you are that the population parameter value in future independent experiments (that replicate yours) will fall within the same confidence intervals in those future experiments.
  - It is an interval that covers a range of collected sample means from a population
- Useful to determine what is statistically signficant - i.e. if a confidence interval is 95% it means that the p-value is 5%, so if any result occurs outside of the confidence interval then it is significant (indicates that the probability that the true mean of a population is outside of the 95% range of sample means is less than 5%)
- Typical confidence interval values are 95%,99%,or 90%
- Can think of the interval as a horizontal range or line in a distribution under which a population parameter would fall.
- Principle: if we have large enough samples from two different populations and we plot the difference of sample means as a distribution (becomes normal with large enough sampling, i.e. size >=30), then we can say that the mean of that sample diff distribution is equal to the difference between the means of the two populations.
  - As long as either the sample size is at least 30 or the original populations are normally distributed, then the sample mean diff distribution will also be normally distributed
  - $\mu_{\bar{x_1}-{\bar{x_2}} = \mu_{1} - \mu_{2}}$
- **NOTE**: A confidence interval around a sample mean does NOT gaurantee that the true mean of the population will lie within that range. Instead, the Confidence Interval tells us that if we repeat the experiment a large number of times, then 95% of those times, the true mean of the pop. will fall within the confidence interval range of the sample mean.

### Interpreting Confidence Intervals

- see [video](https://www.udemy.com/course/statsml_x/learn/lecture/20025094) and Q&A
- Confidence intervals are NOT about confidence of a true population parameter, but about the sample estimate.

  - Example: say that there is a 95% probability that the mean of the repeated sample will be in the range of 1 to 5, then the Confidence Interval calculated for the sample parameter is about the reliability of the parameter estimate and about the prediction of parameter values when taking a repeated samples.

- The thing to keep in mind is that the interval is your confidence about the reproducibility of the estimate of the population parameter, not about the population parameter itself. So, how confident can you be about whether a replication experiment will obtain the same estimate that you have found?
- INCORRECT STATEMENTS (COMMON MISINTERPRETATIONS):
  - "I am 95% confident that the population mean is the sample mean"
    - Correct: "95% of confidence intervals in repeated samples will contain the true population mean". 95% of future experiments will contain confidence intervals that contain the true population mean.
    - The incorrect statement indicates the pop mean will equal the sample mean which is wrong - there is an interval or range that the true pop mean can fall in.
    - The statement is also wrong because it focuses on a single sample mean collected - the confidence interval is about many repeated experiments and sample means and how often the true pop mean will fall within a range over many experiments.
  - "I am 95% confident that the population mean is within the confidence interval in my dataset".
    - This is incorrect because the confidence interval refers to the estimate and not the population parameter (the mean itself). It's about the confidence you can have about the estimation procedure of your sample estimate. i.e. how confident are you about the mean in your dataset, not the population parameter.
    - 95% of the repititions of experiments will have confidence intervals that include the population mean. The dataset being used and sampled might not contain the population parameter, 95% of repeated experiments will contain the population parameter, but yours might not be one of those.
    - The confidence interval tells us about our confidence about the precision of our estimate of the sample parameters.
  - "Confidence intervals for two parameters overlap; therefore, they cannot be significantly different"
    - whether confidence intervals overlap tells us nothing about statistical significance. They are the estimate of a parameter (i.e. mean etc.), and not about the relationship between parameters.

### Formula

$$P(L < \mu < U) = c$$

- L is the lower boundary of the confidence interval, U is the upper boundary, and c is the proportion of the confidence interval, i.e. 95% etc., mu is the population mean
- The probability that the population param is between the lower and upper bound of the confidence interval is "c" percent in a large number of future experiements that replicate the current experiment
  - The proportion of a large number of samples that will include the population parameter within its confidence interval.

### Factors influencing the width of the confidence interval

- Sample size: larger sample sizes make smaller confidence interval widths/boundaries get closer to each other.
- Variance: As variance in the data gets smaller, the confidence interval boundaries are closer together/more narrow.

### Computing Confidence Intervals

#### Via a Formula with 1 Sample dataset

- see [video](https://www.udemy.com/course/statsml_x/learn/lecture/20025080)
- Use a formula when you can assume that the underlying population distribution is roughly gaussian
- Usually you will not be able to get large number of sample sizes to iterate on, so you will have to compute a confidence interval with one sample set:
  $$\text{C.I.} = \bar{x} \underline{+} t^*(k){s \over \sqrt{n}} $$
- A confidence interval around the mean is the mean plus or minus (upper and lower bound) t-star by k degrees of freedom times s over square root on n, where:
  - x bar is the sample mean
  - the $\underline{+}$ represents an upper and lower bound (representing two tail ends of the T distribution)
  - t-star: $t^*$ is a t-value, but not one associated with a t statistic on the data
    - t-star comes from a T-distribution, but it is NOT a t statistic from a t-test. The T-distribution is an analytic (theoretical) distribution (i.e. of Gaussian shape)
    - it is the t-value associated with one tail of the confidence interval (i.e. a 5% value would be 2.5% on each tail, left and right)
    - $tinv({(1-.95)\over 2}, n-1)$: tinv is t-inverse fn where n-1 is degrees of freedom. A t-value takes two parameters: the probability and the degrees of freedom. The probability here is .025, corresponding to a two-tailed 95% confidence interval (2.5% on each side of the distribution). The expression (1-.95)/2 is making it slightly more general. So if you want the 90% confidence interval, replace .95 with .9. Or if you want the 99% confidence interval, replace .95 with .99.
  - s is the sample standard deviation (empirical)
    - Note that as s gets smaller, the confidence interval also gets smaller (which is better)
  - n is the sample size
    - Note that as n gets larger, the confidence interval will get smaller (smaller is better)
- Ideally we want to have a large n and a small s (variability) in order to get a small confidence interval C.I.

#### Assumptions of the formula:

- s is an appropriate measure of variability in the dataset

### Coding Confidence Intervals

- see [notebook](/statsML/confint/stats_confint_analytic.ipynb)
- see for [single formula bootstrapped](/statsML/confint/stats_confint_bootstrap.ipynb)

## Bootstrapping Confidence Intervals (Resampling)

- Nonparametric method, does not use a formula which requires assumptions, no assumptions about the underlying distribution being guassian for example.
- We compute the confidence interval based on the data itself by repeatedly sampling from the dataset (resampling)
  - Resampling: We treat a sample as a population and take samples from that sample

### Resample with Replacement

- method of resampling by taking random vals from a sample, recording them and putting them back in the sample
  - The resampled set is commonly the same size as the original sample size, but that is not a requirement - bootstrapped sets can be smaller than the sample drawn from (i.e. if the sample size is very large and it's expensive to make resampling sets the same size).
  - Example: given a set {1,2,3,4,5}, we randomly take out elements and collect them into a resampled bootstrap dataset of the same size. Note that it's possible to get the same value on random selection and leave out other values altogether: example bootstrapped set: {1,2,2,3,5}
  - We do this resampling and record the mean, iterate and repeat this process many times(i.e. 500-1000 times)
  - By getting the means of all the iterations of the bootstrapped sets, we can make a distribution which will have a peak or mean that should be similar to the mean of the original sample the bootstrapped sets were taken from

### Pros and cons of Bootstrapping/resampling

- Works with any kind of parameter (mean, variance, correlation, median etc.)
  - You can get confidence intervals around the variance, median of a population/set etc., not just the mean.
- Useful when your dataset size is limited or smaller
- Con: gives slightly different results each time the analysis is run
- Con: is time consuming for large ssets (you can mitigate this by having the bootstrapped samples be smaller than the original sample size)
- Con: The data you are working with needs to be representive of the population - with parametric method (analytic and not empircical like this), any outliers in data will have smaller effect, but with this nonparametric method, if your sample is not representative then you will not get a good confidence interval (weird characteristics of the sample used will have a negative effect).

### Code

- see [bootstrapping notebook](../../statsML/confint/stats_confint_bootstrap.ipynb)
