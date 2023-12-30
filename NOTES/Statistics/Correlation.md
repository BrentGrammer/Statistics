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

## Partial Correlations

- Getting the correlation between two variables in a multi-variate scenario (i.e. variables is 3 or more) where you exclude the other variables.
- In a multivariate scenario there could be indirect correlations between variables. A partial correlation could isolate a more direct relationship of a set of variables within the multi-variate scenario
- The greek character rho is used for partial correlation notation ($\rho$)
- When we isolate partial correlations, we say that we partialize out another variable relationship.
  - A partial correlation of a and x while partialializing out z: ${\rho_{ax|z} = 0.5}$
  - If we have a partial correlation that can be computed but is irrelevant to our study then the result could be written as **NA** instead of a decimal.

### Code: Partial Correlations

- Use the `pingouin` package.
  - (scipy,numpy and pandas does not have a built in partial correlation function)
  - `pc = pg.partial_corr(df,x='x3',y='x2',covar='x1')`
- see [notebook](../../statsML/correlation/stats_corr_partialCorrs.ipynb)

## Types of Correlations

### Pearson Correlations

- The correlations computed up to this point above are all Pearson correlations.
- Only appropriate measure for the linear relationship between two (roughly) normally distributed variables/data that do not contain outliers. (small outliers might be okay)
- Potential problem with Pearson correlation is that it can over or underrepresent relationships if they contain nonlinearities or outliers:
  - illustrated in the **Anscobe's quartet**
    - See [video](https://www.udemy.com/course/statsml_x/learn/lecture/20025118#content) at around 1:15
    - The Pearson correlation may not be optimal when dealing with nonlinear relationships and data (i.e. if the data has a nonlinear curve to it)
    - Another problem scenario is when a set of data is very highly correlated except for one outlier very far away which skews and pulls the Pearson correlation badly.

### Spearman (rank) Correlation (a.k.a. Spearman's rho)

- Dominant nonparametric correlation method altervative to the Pearson correlation
- Robust to and useful for dealing with data with outliers
- Looks identical to Pearson correlation when data are linear and normally distributed.
- Tests for a monotonic relationship regardless if the true relationship is linear or nonlinear.
  - Monotonic relationship: tests for increasing or decreasing numbers regardless of the spacing between the numbers.
    - Example: monotonic increase - Every point is higher than the previous point
    - montonic decrease - every point is lower than the previous one
    - a relationship is non-monotonic if the consecutive data points go up and down and are not consistently greater or less than in longer sequences.

### How Spearman Correlation works:

- Transform both variables to rank
  - i.e. [33233,-40,1,0] transforms to [4,1,3,2]
  - Transforming the data to rank eliminates the effect of outliers (i.e. 33233 distance from other data points no longer matters)
  - Note that in practice you might just remove the 33233 data point since it is such an extreme outlier and might not be useful or meaningful. But, if you have situations with outliers that may still be meaningful then you opt for the Spearman method.
- Then compute the Pearson Correlation Coefficient on the ranks
- P value for statistical significance is the same as computing it for the Pearson Correlation coefficient.

## Fisher-Z Transformation for Correlations

- Used when you are pooling a collection of multiple correlations together to run analysis on them.
  - You then want to run subsequent analysis that makes assumptions about normal distributed data like a t-test or ANOVAs etc.
  - Fisher-z transform is usually used when testing whether a sample of correlation coefficients is significantly different from r=zero.
- Correlations may need to be transformed for further use in some situations
  - i.e. when you are doing analysis on a collection of correlation coefficients
    - A distribution of correlation coefficients is uniformly distributed (not normally distributed)
    - If the end goal of the analysis is to just determine the correlation coefficient, then no transformations are necessary...
    - But, there are many statistical methods that assume normally or Gaussian distributed data, so if you are goin analysis on a collection or distribution of correlation coefficients, you can transform them to a gaussian scale

### Formula for the Fisher-Z transformation:

$$z_r = {1 \over 2}ln({1+r \over 1-r})$$

- z-tranform of correlation coefficient `r` is 1/2 of the natural log of 1+r divided by 1-r
  - note that the log of a negative number does not exist in the real number world
  - the log of a number between 0 and 1 is going to be a negative number
  - the log of a number that's greater than 1 is going to be a positive number
  - So, **when $({1+r \over 1-r})$ is greater than one, then the result of the fisher-z transform will be positive, and when that part is between 0 and 1, then the result will be negative**
- Note that this formula is the inverse hyperbolic tangent (inverse of the arc) of the correlation coefficient ($\text{arctanh}(r)$)

### Code: Fisher-Z transformations (and Spearman correlation method)

- see [notebook](../../statsML/correlation/stats_corr_Spearman.ipynb)
- use scipy stats package for spearman correlation: `corr_s = stats.spearmanr(anscombe[:,i*2],anscombe[:,i*2+1])[0]`
  - this automatically rank transforms the data for you and gets the correlation
- Fisher-Z transform:

```python
# Fisher-Z - use inverse hyperbolic tangent built into numpy which is equivalent to fisher z transform, pass in correlation vals
fz = np.arctanh(r)
```

## Kendall's Correlation

- Specifically used only if you have ordinal data (categorical data that is not intrinsically numeric with no fixed relationship across levels)
  - i.e. for education levels (Middle School, High School, Bachelor's degree, Masters degree)
  - movie ratings (i.e. 1 to 5 stars), the difference between one and two stars is not the same as the diff between 4 and 5 stars. You also cannot sum them together (multiple one star ratings cannot sum them together to get a 5 star rating.)
- We transform the data to rank concordances (this means "agreement" between two things)
  - relative signs across the values per variable
- Two versions of the Kendall Correlation:
  - Kendall tau-b: includes an adjustment for ties and is most often used.
  - Kendall tau-a: uses a simplified normalization factor which can underestimate the true correlation in the data.
  - The R when using Kendall correlation method is called the Kendall Tau ("Taow")
    - the value is interpreted same as Pearson correlation - -1 is negative correlation, 0 is no correlation and 1 is positive correleation
    - same characteristics of other correlations apply: i.e. if you have modest correlation that is not statistically significant given small N number of samples, but if you add more and more data sampled the p-value gets smaller and smaller.

### Formula

$$t = K^{-1}\sum \text{sgn}(x_i-x_{i:})\text{sgn}(y_i-y_{i:})$$

- $\text{sgn}$ is the signum or sine function - returns -1 for any negative num, 0 for 0 and +1 for any positive number
- x and y are the two variables (the rank version of them)
- We compute the sine value of data point i relative to the points in the rest of the data ($x_{i:}$ means index i up until the end of the vector)
- The idea is we want to know if x at point i is relatively large or small compared to the other ranked transformed elements of x.
  - so when the points x or y at i are relatively small to the rest, the multiplication of the two sine functions for them will yield a positive number
  - if x/y at the i-th point in the data is relatively large compared to the other points then the two sine functions operate on a positive number.
  - So when x/y is either relatively large or small to the rest of the distribution, the sum $\sum$ grows and gets larger and is positive.
  - If x is relatively high and y is relatively low alternatively, we'll have the sine functions taking in positive and negative yield results - this gives us negative numbers to sum up which will decrease $\sum$ overall
    - Note that $\sum$ is not bound by 1, so it will be some integer value that could be arbitrarily large. This is why we use the normalization $K^{-1}$. K is designed to be the maximum possible value that everything to the right of it in the formula (the whole rest of formula result) could be if there were 100% concordances (this scales it down to 1). K is also adjusted for the ties.
      - see [video](https://www.udemy.com/course/statsml_x/learn/lecture/20025130) at 9:40

### Code: Kendall Correlation

- use scipy stats function: `cr[0] = stats.kendalltau(eduLevel,docuRatings)[0]`
- see [notebook](../../statsML/correlation/stats_corr_Kendall.ipynb)

## Simpson's Paradox (Sub-group correlation paradox)

- If you take a group of data that is highly correlated and split it into groups and look at the correlation inside each group separately, you can find that there is no correlation.
  - But, paradoxically if you look at the groups together in aggregate you can find a strong linear relationship with high correlation.
  - You can also find instances where each group has a negative correlation
- This can lead to misleading conclusions that are incorrect.
- If you encounter this in your data, you need to make a decision about whether the sub groups are qualitatively distinct from each other.
  - If they really are distinct, then maybe the sub groups come from different samples/populations or have some underlying fundamental difference
  - If it is determined they really are separate categorically then it is more appropriate to interpret the correlations of the sub groups individually and ignore the aggregate correlation.
  - If the groups are actually closely related and there is no justification for splitting, then you need to ignore the sub group correlation and use the aggregate correlation.
- See [notebook](../../statsML/correlation/stats_corr_subgroups.ipynb) for examples of demonstrating this in code.

# Cosine Similarity

- Another way of looking at relationship between data and closely related to the Pearson correlation coefficient.
- This method is interpreted the same way that the Pearson Correlation is. (just an alternative way to get the same thing)
- The cosine similarity and correlation can differ (one uses mean centered data the other does not mean center), but neither are wrong if so. they just tell us different things based on different assumptions
  - With Pearson r, for example we care about whether the data values are generally going up and down together and whether large or small jumps are seen in both vars at the same time.
  - The relationships between the values are what is important and whether the variables generally go up and down together is what's important.
  - With Cosine Similarity, on the other hand, you might see a diff if there are scale differences between the variables.

### Formula

$$\text{cos}() = {{\sum^n_{i=1} x_i y_i} \over \sqrt{\sum^n_{i=1} x_i^2} \sqrt{\sum_{i=1}^n y_i^2}}$$

- The cosine of the angle between two variables/lines (vectors in $n$ dimensional space, iow there are n elements in the vectors)
  - $x_i y_i$ are like lines in space: x is one line in $n$ dimensional space, y is another line in $n$ dim space.
  - The cosine of an angle varies between -1 and +1.
    - The cosine of the angle is -1 when the two lines meet at 180 degrees (going in opposite directions)
    - The cosine of the angle is +1 when the lines are exactly on top of each other, when they're parallel
    - The cosine of the angle is 0 when the angle between the vectors is 90 degrees (orthogonal)
- Note the similarity to the Pearson R formula. The difference is here we do not subract the means of x and y.
  - so the pearson formula is the same if the data is mean centered or the mean is 0

### Code

```python
# implementing the formula for data x and y
cs_num = sum(x*y) # numerator: each element in x times each element in y summed up
cs_den = np.sqrt(sum(x*x)) * np.sqrt(sum(y*y)) # denominator: in algebra the norm of vector x and norm of vector y in technical terms
corrs[ri,1] = cs_num / cs_den # the ratio is the cosine similarity

# you can use the spatial module instead as well for a built in function:
from scipy import spatial
corrs[ri,1] = 1-spatial.distance.cosine(x,y)
```

- See [notebook](../../statsML/correlation/stats_corr_cosine.ipynb)
