# Statistical Power

$$\text{Statistical Power} = 1-\beta$$

- i.e. a True Positive in the Type Error table (A true positive means that we can reject the null hypothesis and it is not a Type I error)

### What is statistical power?

- Formal definition:The probability of rejecting the null hypothesis given that the null hypothesis is really false.
  - IOW, the probability of rejecting H_0 given that the right thing to do is to reject it.
- Colloqial definition: The probability of finding an effect when there really exists an effect in the real world.
  - Example: imagine you have a small sample size with high individual variability and the effect you're looking for is relatively small. The null hypothesis distribution will be wide and the alternative hypothesis distribution will largely overlap with the null hypothesis.
    - So it can be difficult to distinguish and find a true effect if it is really there - see [video](https://www.udemy.com/course/statsml_x/learn/lecture/20249432#content) at timestamp 3:15
- Expressed as a probability - can be referred to as a proportion between 0 and 1 as well or from 0% to 100%.

### Interpreting the Statistical Power

- You want the statistical power to be as high as possible.
  - The higher the power, the more likely you are to identify an actual effect in your data when that effect is present in the real world.
- Statistical Power can be different for different analyses of the same dataset
- Caution: Computing the true Statistical Power requires knowing the true effect size (this is unknowable).
- Use Statistical Power as a guideline, but not as a trustworthy numerical value
- See [video](https://www.udemy.com/course/statsml_x/learn/lecture/20249434#content) at timestamp 7:00 for visuals and explanation of how variability, standard deviation and sample size interact and relate to affecting statistical power
  - Show how lower Std Dev (variance) and higher samples sizes increases statistical power etc.

### Ways to increase Statistical Power

- Increase Sample Size
- Look for larger effects
- Lower the alpha threshold (ex: p<0.1) - this is not desirable due to increase risk of Type I errors, but technically an option
  - Type I Errors: False Positive - occurs when the null hypothesis is actually true, but we get a very large test statistic (i.e. an outlier for ex.) sample by random chance, so it's past the threshold and we incorrectly favor the alternative hypothesis mistaking it for signal. We fail to identify a real effect.
- Statistical power DECREASES with:
  - Higher variability
  - More stringent alpha thresholds (p<0.01) or very small critical alpha vals

### A Priori Power

- Getting the effect size and standard deviation from published studies or pilot data (gathering initial samples to estimate statistical power in a larger scale study)
- Used to compute sample size before you start running your experiment and collecting the data
- A Priori Statistical Power is widely used and accepted (but wrought with uncertainties for awareness - don't have too much faith in it, but should still be computed)

### Post-hoc Power

- computing statistical power on your data and analyses after you've already run them and collected all the data
- This winds up being proportional to the p-value and provides no new useful information (the p-value and sample size in the data you already have is the same info)
- Asked for sometimes, but it is useless and not providing anything new
  - Post-hoc power is the effect size in the sample you've already collected, which is essentially the same thing as the p-value. So it's a bit circular. The non-circular way to use statistical power is to estimate a sample size of a planned (future) study, based on previous studies.

## Statistical Power Computation in Practice

- You can use an online calculator for computing Statistical Power and Sample Size
- The most accepted one is [G\*Power](https://www.psychologie.hhu.de/arbeitsgruppen/allgemeine-psychologie-und-arbeitspsychologie/gpower)
  - Free to download
  - See [video](https://www.udemy.com/course/statsml_x/learn/lecture/20249440#content) at timestamp 1:06 for demo
- You can compute a desired sample size for an A Priori Statistical Power by plugging in an effect size, the statistical power you want (i.e. 0.9 for ex.) and parameters
  - For suggestions on what paramters to use you can optionally look through the relevant scientific literature to see the effects sizes of similar studies using similar experimental techniques.
