# Histograms

- https://www.udemy.com/course/statsml_x/learn/lecture/20009322#content
- Have binned continuous data (interval,ratio,descrete) on the x-axis. (Bar plots have categories on the x-axis)
- How to tell if you need a bar plot: Can you swap/re-arrange or re-order the locations of the x-axis bins? If yes, you are dealing with a bar plot
- The shape of a histogram is important, the shape of a bar plot is not.
- You can move categories on the x-axis around and the shape of the higher/lower bars doesn't matter - their order.
- But, if you move the values/bins on the x-axis of a histogram around, then it doesn't make sense. You'll have a bin that is greater values coming the left of a bin with smaller values etc. Ordering on the X-axis is meaningful in a histogram.

### How many bins to use?

- [video](https://www.udemy.com/course/statsml_x/learn/lecture/20009434#content)
- The goal of how many bins is to show enough to get the shape of the data, but not be overly detailed.
- too few bins does not give an impression of the distribution or shape characteristic of the data.
- There are mathematical guidelines for determing number of bins.

#### Rules for determining number of bins:

- **Freedman-Diaconis Rule (FD)**: Best and most recommended rule to use! Specify h (the width of bins) instead of k (the number of bins). As data increases, the bin size/width gets larger and the number of bins start to decrease. Depends on both the data count and the data spread. It is also guided by the features and characteristics of the data set. Given the width (h) you can apply the below formula to get the number of bins you should use: $$k = {\lceil{{max(x) - min(x)} \over h}\rceil}$$ - k is number of bins, h is the width of the bins (how wide they are, range of values for one bin), where x is the data values - The square brackets mean apply the ceiling function (always round up, i.e. 2.01 => 3.0)
- Sturges guideline: specify k directly, the number of bins increase as the number of data points you have increases
  - Is a logarithmic function - increase starts steep at the beginning and the increase slows down as you get further to the right - as n gets larger, the number of bins does not increase linearly (slows down)
- Arbirtrary Rule: just directly assigns a number of k bins. advantage is it's easy to use, based on intuition. generally 30,40,50 tends to be good number.
- Do NOT vary bin widths - make them all the same size and width.

```python
# use matplotlib to get FD formulated bins:
plt.hist(data, bins='fd')
plt.show()

# Alternatively, seabourn presents a nicer plot with a probability distribution curve
import seaborn as sns

sns.distplot(data) # uses FD by default
```

### Using Proportion vs. Count measurements

- On the y-axis you can have a count or a percentage (proportion) scale measurement.

#### Counts

- More meaningful for interpreting raw numbers
- unbounded, does not need to sum to 1 or 100%
- more difficult to compare across datasets
- Better for Qaulatative inspection (for one particular dataset or research study)

#### Proportion (%)

- formula for getting proprotion from a raw bin data = $100 * (bin / sumOfAllBins)$
- Easy to compare across dataset (doesn't make sense to compare raw numbers since population sums will probably be different)
- adds extra step to get the raw number data
- Better for Quantitative analysis (comparing across different studies)

### Lines vs. Bars

- When you are comparing historgrams overlapping you can switch to plot them as lines to easier visualize the comparison on a graph plot
- Lines are never suitable for bar plots with categorical data.

## Code

```python
import matplotlib.pyplot as plt
import numpy as np

# creating data (follows log normal distribution)
n = 1000
# put the random numbers into the natural exponential (exp) to get log normal distribution. note that the natural exponent cannot yield a negative value even if the numbers are negative it operates on.
data = np.exp(np.random.randn(n)/2) # 2 is just a random normalization number to affect histogram shape.

# show a histogram of the data using the .hist() method

# number of bins
k = 40

plt.hist(data, bins=k)
plt.show()


#### You can show a line graph representation of the histogram data as follows #####

#### Extract the data (y,x) from the histogram with numpy
y, x = np.histogram(data, bins=k)

# you need to get the midpoint of each bin, otherwise you'll just have the boundaries of the bins which add up to one more than the x axis
# note that x is not a list it is a numpy ndarray type
xx = (x[1:]+x[:-1])/2 # you have to average the bin values together in order to get the plot to work and not throw errors that x does not equal y dimensions.

plt.plot(xx,y) # plot the line graph repr of the histogram using the center of the bins to make x line up with y counts
plt.show()
```

### Showing a line graph for a histogram

- useful when we want to plot multiple histograms for comparison over each other.

```python
import matplotlib.pyplot as plt
import numpy as np

n = 1000
# generate some data with different distributions
data1 = np.exp(np.random.randn(n)/2)
data2 = np.exp(np.random.randn(n)/10)
data3 = np.exp(np.random.randn(n)/2 + 1)

# histogram discretization for the datasets
y1,x1 = np.histogram(data1,bins=k)
xx1 = (x1[0:-1] + x1[1:]) / 2 # get the midpoints of bins to prevent x/y dimensions error (# you have to average the bin values together in order to get the plot to work and not throw errors that x does not equal y dimensions.)
y1 = y1 / sum(y1) # convert to proportion - amount over the total sum of all y axis values (total count of data)

y2,x2 = np.histogram(data2,bins=k)
xx2 = (x2[0:-1] + x2[1:]) / 2 # get midpoint for dimensions error
y2 = y2 / sum(y2) # convert to proportion - amount over the total sum of all y axis values

y3,x3 = np.histogram(data3,bins=k)
xx3 = (x3[0:-1] + x3[1:]) / 2 # get midpoint for dimensions error
y3 = y3 / sum(y3) # convert to proportion - amount over the total sum of all y axis values



# show the line plots which overlap each other
plt.plot(xx1,y1,'s-',label='data1') # s- means plot points as sqaures
plt.plot(xx2,y2,'o-',label='data2') # o- circles
plt.plot(xx3,y3,'^-',label='data3') # ^- means triangles

plt.legend()
plt.xlabel('Value')
plt.ylabel('Probability')
plt.show()
```
