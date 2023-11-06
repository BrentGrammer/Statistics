# Measures of Central Tendency

- Similar to the "center of gravity"
- The value that you would expect the data points to merge towards
  - or where the data is **most strongly clustered**
- In a Gaussian Distribution, the mean, median and mode are all the same thing.
- Central Tendency is NOT necessarily the same as "expected value" (the data value times it's probability of occurance)

## Mean

- a.k.a. the average, or "arithmetic mean" (there are other types of means like a geometric mean, etc.)
- formula: $$\bar{x} = n^{-1}\sum_{i=1}^n x_i$$
  - take all the data values ($x_i$) and sum them up ( $\sum_{i=1}^n$ ) and divide by `n` the number of data points in the set ($n^{-1}$)
    - note that $n^{-1}$ is the same as 1/n (the reciprocal of the sum of data points - multiplying the sum by this is the same as dividing the sum by n)
- Notation: indicated by bar over the data sum variable ($\bar{x}$) or mu: $\mu$ or $\mu_{x}$

### When to use the mean:

- **The mean is suitable for data that is roughly normally distributed (does not have to be perfectly gaussian)**
  - The mean is the peak or center of the distribution curve
  - a "failure" scenario would be in a bimodal distribution for example (with two peaks) or a skewed distribution - here the mean is not representative of the tendency of the distribution.
- **Most suitable for interval and ratio data**
  - Not suitable for discrete data (the average number of kids in a family), the mean can be a decimal which might require careful interpretation (1.9 children etc.) - the division results in decimals which go against discrete data (whole integers)
  - or ordinal data like course ratings (out of five stars) or level of education. i.e. the average is awkward because the difference between 3 and 4 stars is not the same as the difference between 4 and 5 stars (2 stars is higher than 1 star. But, 2 stars isn't "twice as good" as 1 star), or the difference between an associate's and master's degree is not the same as that of a bachelor's and master's etc.
  - nominal data: flavors of ice cream (assign to numbers), an average of 1.7

## Median

- The middle value of the data: half of the data is below that value and half of the data is above it.
- formula: $$x_i, i = {{n+1}\over{2}}$$
  - i is the `index` and the median is the data point value at that index ($x_i$)
  - To get the median take a set and sort it ([0,4,1,-2,7] > [-2,0,1,4,7]) and take the middle (1)
    - If you have an even number of points, take the average of the two middle values [-2,0,1,4,7.10] - the med(x) is 2.5 (avg of 1 and 4)
- notation: `med(x) = n`

### When to use the median:

- Suitable for any Unimodal distribution (mot more than one peak in the distribution)
  - should tell you where the data is most strongly clustered (in bimodal distributions for example, it does not as there are large number of points on either side of it)
  - Better than the mean for skewed distributions for example.
- Suitable for interval or ratio data
- Example: income distributions - the median will not be pulled in the direction of outliers whereas the mean will (if outliers earn much more income than most people). The difference of the mean and median of net incomes in a country can be a measure of income inequality (income inequality was 0 then the mean and median would be the same).

## Mode

- The most common value in a dataset
  - [0,0,1,1,1,1,2,3] > the mode is 1
  - it is possible to have more than one mode! [0,0,0,1,1,1,2,3] > mode(x) = 0,1
  - notation: mode(x) = 1
-

### When to use Mode:

- Suitable for any distribution
- Suitable for any data (numerical data needs to be converted to discrete values)
- Most useful for nominal data types
  - Example mediums of news - 'TV, internet, radio' - the mode would be internet

# Coding in Python

## Showing the Mean

```python
N = 10001 # num of data points
nbins = # num histogram bins

# create normal distribution of data with numpy
d = np.random.randn(N)

y, x = np.histogram(d, nbins)
x = (x[1:]+x[:-1])/2 # (# you have to average the bin values together in order to get the plot to work and not throw errors that x does not equal y dimensions. note x is not a list, but a special numpy type so the addition is adding all vals)

plt.plot(x,y)

plt.xlabel('Data vals')
plt.ylabel('Data counts')
plt.show()

## Overlay the mean on top of the distribution

# can use numpy if not using sum() and len() for the formula:
mean = np.mean(d)

plt.plot(x,y)

###  to draw a line you specify two X points and two Y points.
### In this case, both X points at the same, which means the line will be vertical (it doesn't move on the x-axis).
### The two Y points are 0 and max(y1), which means a line that goes from 0 to the largest value.
plt.plot([mean, mean], [0,max(y)])

plt.show()
```

## Showing the Median (against a mean for comparison)

```python
# create a log normal distribution skewed to the right
shift = 0
stretch = .7
n = 2000
nbins = 50

# generate data skewed to right (stretch)
data = stretch * np.random.randn(n) + shift
data = np.exp(data) # generates log normal distribution (raises e to the value of the data)

# generate historgram
y,x = np.histogram(data, nbins)
x = (x[:-1] + x[1:])/2

mean = np.mean(data)
# generate the median
median = np.median(data)

plt.plot(x,y)
# show lines of mean and median on the plot:
plt.plot([mean,mean],[0,max(y)], 'r--')
plt.plot([median,median],[0,max(y)], 'b--')
plt.set_title('Log-normal data histogram')
plt.show()
```

## Showing the Mode

```python
data = np.round(np.random.randn(10)) # rounds all data to integers i.e. [0,1,0,1,0,0,1,0,0,0] to get discrete data - the point is that there is a small number of values data can be to get the most common
# note - if you get values like -0.0 or -1.0, it represents a tiny rounding error from np.round.

uniq_data = np.unique(data)
for i in range(len(uniq_data)):
    print(f'{uniq_data[i]} appears {sum(data==uniq_data[i])} times.')

# stats.mode comes from the scipy package
mode = stats.mode(data)[0][0] # the value that occurs most often


```
