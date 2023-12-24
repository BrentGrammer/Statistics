# Correlation

- Takes into account individual variability in a group and reveals if there are important relationships between individual variability and some other variable
- Comparisons between individuals must be paired - they can be different or the same individuals in different groups can be compared (i.e. different aspects of the same individual can be compared/correlated)
  - When we plot correlations on a graph, each point represents a pair of data points. the parameters for each individual for the pair are on the x/y axis respectively
- Correlation coefficients are computed over groups - it is not possible to compute a correlation between one pair of data points.
- Correlation is just a relationship and does not imply causality

### Formal Definition

- a correlation analysis computes a correlation coefficient (usually termed "r")
- The correlation coefficient "r" is a single number that shows the linear relationship between two variables
- A correlation coefficient is a continuous measure of correlation strength: a correlation efficient value by itself without any context is meaningless (it is not high or low whatever it may be on it's own).
  - A corresponding p-value needs to be computed that accompanies each r value. You use this p-value to interpret the signficance of the coefficient.
    - Ex: If the p-value is very low, i.e. below 5%, then that means that it is significant

### Negative, Positive and Zero Correlations

- A correlation Coefficient is standardized and is between -1 and +1.
  - -1 is a perfect inverse relationship: 1 variable goes up the other does down
  - +1 is a perfect positive relationship: both variables go up and down together
  - 0 means no relationship: one variable tells you nothing about the other.
- On a plot:
  - positive correlation resembles a line (points clustered around it) that slopes up, as one var goes up, the other goes up as well in a pair
  - negative correlation looks like a line going down (points clustered around it), when one variable goes up, the other goes down in a pair
  - Zero correlation looks like points scattered all over with no line formed.
    - Note: It is not accurate to say there is NO Correlation in this case - there is never no correlation. There is always a correlation, it is just 0 or close to 0 or not significant
    - **Points can be strongly related, i.e. non linearly, and still have zero correlation**: Correlation measures a linear relationship, but it is possible to have nonlinear relationships (see [video](https://www.udemy.com/course/statsml_x/learn/lecture/20025098) at 12:10 where plot is a circle shape), however since the relationship is not linear, the correlation is still zero. i.e. it would be the case that as one variable in a pair goes up, the other variable does not go up the same distance etc.
    - Note that it is also possible to get a significant r value for correlation even if the plot does not look clearly linear (points more scattered but still a small correlation of say .3)

## Covariance

- A single number that measures the linear relationship between two variables
- Similar to correlation - to get correlation, you start by computing the covariance and then normalize it to get correlation.
  - A correlation coefficient is just a covariance coefficient that has been scaled to between -1 and +1.
  - covariance is not normalized and in the scale of the original data
- Formula for covariance: $$c = {1 \over n-1}\sum_{i=1}^n(x_i - \bar{x})(y_i - \bar{y})$$

  - this is the covariance c for two variables x and y, it is the sum of each x and y element mean centered.
  - The reason we divide by n-1 is because subtracting the mean from x and y takes away one degree of freedom, so once we take away one element (the last) then it can be determined and is no longer free
  - $1 \over n-1$ is for normalization and is not always critical to have depending on what you're doing with the covariance

  ### Correlation Coefficent (R) formula

  $$r = {\sum_{i=1}^n(x_i - \bar{x})(y_i - \bar{y})} \over {\sqrt{\sum_{i=1}^n(x_i - \bar{x})^2{\sum_{i=1}^n(y_i - \bar{y})^2}}}$$

  - The numerator is basically the covariance forumula above
  - the denominator is the sums of the x and y values squared, multiplied together, and then the square root of that is taken. This equates to the denominator equalling the numerator if x = y which means that the maximum value of r is 1. (it is a normalization to bound the possible solution between -1 and +1)

  ### Computing the P-value for a correlation

  - We need a p-value to determine whether a correlation coefficient is signficant or not (we cannot determine this with a correlation coefficient on it's own)
  - Formula: $$t_{n-2} = {r\sqrt{n-2} \over 1-r^2}$$
    - where $r$ is the correation coefficient, $n$ is the sample size
    - As sample size increases it makes a sublinear increase in r (since we're taking the square root of a measure of n). So as n increases, r increases, but r increases at a smaller rate as n gets larger
    - as $r$ increases and approaches 1, the denominator gets smaller and closer to zero, so it means that the ratio gets large as r increases
    - as $r$ gets closer to -1, the numerator becomes negative, but the denominator still approaches zero
    - as the t-statistic $t_{n-2}$ gets larger and larger, the p-value will get smaller

### Code (Correlation Coefficient)

- See [notebook](../../statsML/correlation/stats_corr_corrcoef.ipynb)
- use numpy method for covariance: `np.cov(np.vstack((x,y)))`
- Correlation: `np.corrcoef(np.vstack((x,y)))`
- Correlation with p-value:

```python
# use this in practice: the stats module from scipy has a pearsonr function that returns both the correlation r and the p value p (the other builtin correlation method does not return p-value)
r,p = stats.pearsonr(x,y)
```

### Coding data with a specified correlation

- NOTE: When using this method to generate data with a specific r, it will not match exactly the target you specify, but will get close to it (esp. with larger sample sizes)
- See [notebook](../../statsML/correlation/stats_corr_generateCorrelation.ipynb)

## Correlation Matrix

- Used for comparing correlations from multiple variables (i.e. 3 or more vars)
- Correlation matrices are symmetric - the halves mirror each other along a diagonal boundary (consisting of the self same r values)
  Ex:

```
  1 .67 .51
.67   1 .41
.51 .41   1
```

### Covariance Matrix

- The first step to getting the correlation matrix is to compute the covariance matrix

```
3.16 4.13  2.79
4.13 11.79 4.32
2.79 4.32  9.31
```

- Note how the diagonals are not all 1s like in the correlation matrix.
  - **Computing the covariance of a variable against itself is simply the variance of that variable.**

### Coloring Matrices

- You can color the squares of a correlation or covariance matrix with lower numbers being cooler colors (closer to blue) and higher numbers being closer to yellow/reds
- Useful for larger matrices with lots of variables - helps to visualize and saves space since you do not have room to print out the numbers if there are lots of them.

### Coding Correlation/Covariance Matrices
- See [notebook](../../statsML/correlation/stats_corr_corrMatrix.ipynb)

