# Distributions

- The shape of a histogram (i.e. when finely grained). (the shape of the line following a histogram graph)
- Statistical procedures chosen are based on the assumptions about what kind of distribution the data is
  - determines what procedure to apply
- Nature and Biological systems use distributions to try to understand these systems.

### Qualities of distributions (what to take note of)

- Peak: where the distributions have the highest point
- Width
- Unimodal Distributions: Has one peak or high point
- BiModal or Multimodal Distributions: Have more than one peak (more than one mode)

## Common distributions:

### Power Law Distributions (1 over F Distribution)

- characteristic of a dataset where the number/strength of occurances (say on the x axis) is inversely related to the count/magnitude of the occurances (say on the y axis).
- Larger values are less frequent in the data, and conversely smaller values are more frequent in the data.
- Example: Earthquakes (magnitude vs. frequency). There is a large number of weak earthquakes and as you get stronger earthquakes there are fewer and fewer of them.

### Gaussian

- A bell curve
- negative or positive values
- Types:

  - Analytical Normal Distribution is based on a formula, i.e. a Gaussian distribution is theoretical with predetermined characteristics. A.K.A. a theoretical distribution.
  - Empirical Normal Distribution: representative of the actual data - not predetermined characteristics - may not be perfectly Gaussian, just roughly gaussian.

- The core simplified and distrilled formula of a Gaussian Distribution is: $e^{-x^2}$
  - You can plot $e$ on the x axis and $-x^2$ (also called "t") on the y axis
  - Almost everything in statistics can be related back to this simple expression

### Uniform Distribution

- all values within a range are equally likely to occur, values outside that range are not at all likely to occur.
  - On a histogram this distribution has no definite peak and is "flat"- all values are roughly equally likely to appear.

### t Distribution

- measures statistical significance of t values.
- negative or positive values

### F Distribution

- Important for general linear models, like regressions
- only defined for positive values

### Chi-sqaure ($\chi^2$)

- Greek letter Chi (pronounced "Kie"): ($\chi$)
- only defined for positive values

### QQ Plots (quintile-quintile plots)

- [video](https://www.udemy.com/course/statsml_x/learn/lecture/20009414#content)
- Used to determine if your data is roughly normally distributed.
- Useful because many statistical procedures (t-tests, regressions, etc) assume that data is normally distributed.
- Answers the question: how do you know if data points given came from or resemble a guassian distributed system?
- In a graph on the x axis is the expected values in a normal distribution, and on the y-axis are the actual values from a dataset
  - If the actual values land close or on to the line of unity from left to right in the graph then the data can be verified as being normally distributed.
  ```python
  # use stats from scipy, pass in emprical data and it will make a qq plot against a gaussian line and show it
  qq = stats.probplot(data, plot=plt) # plt comes from mapplotlib
  ```

## Code

- Can use the stats module in the `scipy.stats` library

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

######## GAUSSIAN DISTRIBUTION
N = 1001 # symmetric number - nicer to look at than just 1000
# create a vector x (i.e. the x axis) from -4 to +4 in 1001 steps/ticks
x = np.linspace(-4,4,N)

# create a normal distribution using stats.norm, use the probability density function (pdf) of the normal distribution
gausdist = stats.norm.pdf(x)

plt.plot(x, gausdist)
plt.title('Analytic Normal Distribution')
plt.show()
```

### Probability distributions

- The y axis needs to add up to 1 or 100%
- Check the sum of the distribution: `print(sum(gausdist))`
- two ways to convert into a probability distribution

Divide the distribution by the sum of the distribution values: `plt.plot(x, gausdist/sum(gausdist))`

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

N = 1001
x = np.linspace(-4,4,N)

gausdist = stats.norm.pdf(x) # float[] of 1001 values

# check the sum to determine if it is a probability distribution (should not add up to more than allowed (up to 1 total))
print(sum(gausdist)) # prints the sum of all the numbers - is about 125
# divide the sum by itself to get a total of 1:
print(sum(gausdist)/sum(gausdist))

### Convert to a probability distribution
plt.plot(x, gausdist/sum(gausdist)) # divide by the sum of the values
plt.title('Analytic Normal Distribution')
plt.show()
```

- The second way to convert into a probability distribution is to normalize the function. We want the total area within and under the curve to be equal to 1.
- From Calculus you can use a remonsum - a method for computing the area under a curve (you take the height of each bar/point and multiply by the width on the x axis (height x width) - sum up all areas.)
  - to get the widths of the points along the x axis, just take the difference between any two successive values on the x axis.

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

N = 1001
x = np.linspace(-4,4,N)

gausdist = stats.norm.pdf(x) # float[] of 1001 values

# treat the values of the vector as the heights of the bars
height = sum(gausdist)
# multiply by the width values (take difference of first two points)
width = np.diff(x[:2]) # get the first two points with :2
print(height * width) # this will get you very close to 1, .9999... if you extend x axis to -6 and 6 for ex. then you get even closer.

### Convert to a probability distribution
plt.plot(x, gausdist/sum(gausdist)) # divide by the sum of the values
plt.title('Analytic Normal Distribution')
plt.show()
```

#### Emprical Normal Distribution

- Analytical Normal Distribution is based on a formula

```python
import matplotlib.pyplot as plt
import numpy as np

stretch = 1 # optionally you can stretch to give distribution a different variance (square of standard deviation or is this standard deviation?)
shift = 5 # optionally give the distribution a different mean
n = 1000

# generate random numbers. `randn` methods means make normally distributed random numbers
data = stretch * np.random.randn(n) + shift

# plot data
plt.hist(data, 25)
plt.title('Empirical normal distribution')
plt.show()
```

### Uniform Distribution

- `np.random.randn(n)` generates uniformly distributed numbers

```python
n = 10000
# rand (not randn) generates uniformly distributed random numbers
data = np.random.rand(n)

# setup layout for plots (see other examples)
fig, ax = ply.subplots(2, 1, figsize=(5,6))

# build a graph with dots
ax[0].plot(data,'.',markersize=1)
ax[0].set_title('Uniform data values')
# build a histogram
ax[1].hist(data,25)
ax[1].set_title('Uniform data histogram')

plt.show()
```

### Log normal distribution

- use the `stats.lognorm` method from scipy package

```python
from scipy import stats

N = 1001
# linspace generates evenly spaced values over a range (N) - i.e. from 0 to 10 with 1001 values spread evenly between those numbers
x = np.linspace(0,10,N)

lognormdist = stats.lognorm.pdf(x,1)

plt.plot(x, lognormdist)

```

### Binomial Distribution

- What is the probability of getting a certain number of a result given a certain number of repititions
- "Binomial" means there has to be two possible outcomes - i.e. heads or tails in the case of a coin toss, or win or lose if in some kind of game, or pass or fail for an exam
- A graph of a binomial distribution only deals with integer values (you can't have 2.3 heads in x number of coin tosses, for example)

```python
# probability of K heads in N coin tosses given a probability of p heads (.5 is a fair coin not rigged)
# how many times would you get heads given you played the coin toss 10 times
n = 10 # num coin tosses
p = .5 # probability of heads

x = range(n+2) # When creating a binomial distribution, you specify the number of bins for the distribution. It's basically the same thing as specifying the number of bins when creating a histogram. explanation for this is confusing and doesn't make sense.
bindhist = stats.binom.pmf(x,n,p)

plt.bar(x, bindhist)
plt.title('Binomial dist)')
plt.show()
```

### T Distribution

- used to evaluate the statistical significance of a T test
- looks similar to a guassian distribution, the key parameter is degrees of freedom

```python
from
x = np.linspace(-4,4,1001)
degrees_of_freedom = 200 # when smaller, looks less like a guassian (tails higher and thinner in middle), as it gets larger, starts looking more like a gaussian
t = stats.t.pdf(x, degrees_of_freedom)

plt.plot(x, t)
plt.xlabel('t-value')
plt.ylabel('P(t | H$_0$)')
plt.title('t(%g) distribution'%degrees_of_freedom)
plt.show()
```

### F Distribution

- Used to evaluate statistical significance of novas, regressions and general linear model
- Strictly positive - no negative values
- A little like a log normal distribution - a lot of values clustered on the left and fewer values towards the right.
- Has two degrees of freedom parameters (numerator and denominator)
  - these change the shape of the distribution - how far out it is from 0 and what it looks like on the left side.
    - numerator keeps left side of distribution high (starts high value)

```python
numerator_df = 2 # numerator degrees of freedom
denominator_df = 100 # denominator degrees of freedom

# values to evaluate
x = np.linspace(0,10,1000)

# distribution
fdistribution = stats.f.pdf(x, numerator_df, denominator_df)

plt.plot(x, fdistribution)
plt.title(f'F({numerator_df},{denominator_df}) distribution')
plt.xlabel('F value')
plt.show()
```
