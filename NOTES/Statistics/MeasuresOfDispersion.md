# Measures of Dispersion

- [Lecture Video](https://www.udemy.com/course/statsml_x/learn/lecture/20009398#content)

- measurements of how spread out different distributions are.
  - A single number that tells you how dispersed the data are around the central tendency.
  - Ex: a wide curve will have a higher dispersion number - data is more widely distributed, while a narrow curve will have a smaller dispersion measure.

## Variance

- The measure of the average degree to which each number is different to the mean
  - Higher variance = wider range of data/numbers
  - Lower variance = lower range of data/numbers
- Formula for Sample Variance is: $$\sigma^2 = {1\over{n-1}}\sum_{i=1}^n{(x_i - \bar{x})^2}$$
  - Formula is a little similar to the mean - we sum up the data (${\sum_{i=1}^n}$) and divide by some kind of "n"
  - $n - 1$ is the number of degrees of freedom ([see discussion](https://www.udemy.com/course/statsml_x/learn/lecture/20009398#questions/11167339)). There's always going to be 1 value that we don't need to know in advance because it is dependent on the other values and on the mean.
  - uses **"Mean Centering"** ${(x_i - \bar{x})^2}$: Take difference of each individual data point ($x_i$) and the mean ($\bar{x}$) and square that difference.
    - Why mean center? We want to represent dispersion around the Average instead of the magnitude of the values themselves. Goal is to measure distances to the average.
    - Ex: [1,2,3,3,2,1] vs. [101,102,103,103,102,101] - if you did not mean center on the second the outcome would be a huge number for dispersion while it would be small for the first set - even though both sets are dispersed the same spread.
  - Take the result of mean centered data points and sum them all up together
  - Divide that by n-1 where n is the number of data points in the sample
  - Why square the mean centering? If we don't then the outcome is always zero. squaring translates to large values and is easier for working with spread out datasets among other optimizations (you can alternatively use mean absolute difference using the absolute value instead of squaring, but you lose the sqauring benefits, it is useful and robust against outliers, though)
- Indicated as 'sigma squared' ($\sigma^2$)
  - note: sometimes sigma squared $\sigma^2$ indicates population variance while the letter `S` ($s^2$) indicates sample vairance.
  - ex: [8,0,4,1,-2,7] - the variance is about 16, for [2,3,4,2,4,4] the variance is 0.67 (much smaller spread)

#### Code

```python
import numpy as np

# Calculate the variance
variance = np.var(data)

# By default, ddof is set to 1 in np.var, which calculates the sample variance (dividing by N - 1, where N is the number of data points). If you want to calculate the population variance (dividing by N), set ddof to 0.

# Calculate the population variance
population_variance = np.var(data, ddof=0)

## can also use builtin standard statistics lib
import statistics

# Calculate the variance using statistics module
variance = statistics.variance(data)
```

### Sample vs. Population Variance

- In Sample variance we divide by $n - 1$
  - uses sample mean, which is an empirical quantity - it will differ for every single sample
  - a sample taken from a population of all italians height.
  - Ex: The sample mean of a die roll of 4 times can be 3 (different from 3.5 - the population mean below)
- in Population Variance we just divide by $n$
  - uses population mean which is a theoretical quality - you know what it is in advance and it does not change.
  - Ex: avg. height of all Italians
  - Ex: The population mean of a die (dice) is 3.5 (if you roll it inifinite times)
- Biased Variance: Dividing by N instead of N-1 on a sample gives you what's called the "biased" variance. (using the same division of N you would use for a population). dividing by N-1 gives you the unbiased variance. The biased variance on a sample gives you in general a smaller value than the unbiased variance.
  - The difference gets greater if your sample size is smaller (larger does not make as much of a difference)

### When to use variance

- Suitable for any distribution, but most easy to interpret for roughly gaussian unimodal distributions (one peak)
- Suitable data types:
  - Numerical
  - Ordinal(requires a computable mean..) - can be a little tricky with ordinal data

## Standard Deviation

- Measure of how spread out a group of numbers is from the mean
- Usually represented by sigma: $\sigma$
- The square root of Variance: $\sigma = \sqrt{\sigma^2}$
  - note: The square root is equivalent to taking a value to the power of 1/2: $\sqrt{x} = x^{1/2}$

## Fano Factor and Coefficient of variation

- Show up in physics, neuroscience etc.
- Normalized measures of variability
- **Only suitable for datasets with only positive values**
  - don't want to divide by zero

### Fano Factor

- variance divided by the mean: $$F = {\sigma^2\over\mu}$$

### Coefficient of variation

- standard deviation divided by the mean: $$CV = {\sigma\over\mu}$$

# Coding variance measures

## Variance

```python
high_var_array = np.array([1,100,200,300,4000,5000])
low_var_array = np.array([2,4,6,8,10])

# use numpy's np.var() to get the variance
np.var(high_var_array)
np.var(low_var_array)

```

## Standard Deviation

- Different ways of making a normal distribution with mean and median and std deviation vals
- **NOTE**: The numpy standard deviation and variance functions by default compute the Population variances (the biased measures)!
  - You need to remember to use the second argument, `ddof=1`. ddof is degrees of freedom and is zero by default (this is technically incorrect)

left off at 15:27 - need to figure out what the heck the code is doing there.

```python
mean = 10.2
std_dev = 7.5
num_sample = 123

# can use random.normal to pass in values directly from numpy
np.random.normal(mean, std_dev, num_sample)
# this line is equivalent to the above and does the same thing.
np.random.randn(num_sample) * std_dev + mean
```

```python
## create some data distributions

# the distributions
N = 10001   # number of data points
nbins = 30  # number of histogram bins

data = np.random.randn(N) - 1

# need their histograms
y1,x1 = np.histogram(data,nbins)
x1 = (x1[1:]+x1[:-1])/2


# plot them
plt.plot(x1,y1,'b')

plt.xlabel('Data values')
plt.ylabel('Data counts')
plt.show()

######## Calculate Standard Deviations:

# initialize sets (only 1 set here)
stds = np.zeros(1)

# compute the standard deviations of datasets
# **NOTE**: The numpy standard deviation and variance functions by default compute the Population variances (the biased measures)!
# You need to remember to use the second argument, `ddof=1`. ddof is denominator degrees of freedom and is zero by default (this is technically incorrect for sample data since you need to divide by N-1). If you leave ddof=1 out then numpy will implement dividing by N not N-1 (N = number of data points)
stds[0] = np.std(data, ddof=1)

plt.plot(x1,y1,'b')

# add std deviation to plot
plt.plot()
```

### Fano Factor and Coefficient

```python
# The function linspace stands for "linearly spaced."
# generates an array of 15 evenly spaced values over the interval from 1 to 12 (inclusive).
lambdas = np.linspace(1,12,15)

fano = np.zeros(len(lambdas))
cv = np.zeroes(len(lambdas))

for li in range(len(lambdas)):
  data = np.random.poisson(lambdas[li],1000)

  #Fano - divide std deviation by mean
  cv[li] = np.std(data) / np.mean(data)
  #Coefficient Variance - divide variance by mean
  fano[li] = np.var(data) / np.mean(data)

```
