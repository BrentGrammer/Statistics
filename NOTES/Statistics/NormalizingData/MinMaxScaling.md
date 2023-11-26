# Min-max scaling

- Transform data to a range (the most typical range is unity-scaling which is between 0 and 1).
  - The smallest value becomes 0, and the largest value becomes 1.
- Example: a range of data between [-200,1000], the lowest value -200 becomes zero, and the largest value 1000 becomes 1. The other data points are distributed relatively within 0,1 range.
- When scaling to a range of 0 to 1, the term is **Unity-normed data**. Note: The units are arbitrary.
- NOT lossy - we don't lose relative information, the relation between the values in the dataset is preserved, we just shift the scale of the y-axis.
- This is useful if you need to have a restricted range in your analysis, i.e. if anaylsis does not tolerate negative values or extreme values, you can use min-max scaling.

### Formula

#### scaling to 0/1:

- Subtract the minimum value of the entire dataset from each (the i-th) data point and divide by the total range of the data (the smallest value taken away from the largest value). The result gives you the new dataset scaled between 0 and 1.
  - $x$ represents the entire dataset, and $x_i$ represents a single data point in that set.
    $$\tilde{x}_i = {x_i - \min x \over \max {x} - \min x}$$

#### Scaling to other ranges:

- First scale to 0/1 as above to get unity-scaled data: $\tilde{x}$
- Then take that data and add $a$ which is the lowest boundary you want, and multiply each individual element by $b - a$
  $$x^* = a + {\tilde{x}(b - a)}$$
- The range of the new data is $a - b$
- $a$ is equated to $\min x$ and $b$ is equated to $\max x$
  - so in contrast to the unity scaling, we are adding $\min x$ and multiplying by the range (instead of dividing by the range)

## Code

```python
# create some data
N = 42
# take the natural log of 42 uniformly distributed random nums
# using np.random.rand() gives you a uniformly distributed range of numbers between 0 and 1. Note: the log gives you negative numbers. We make them even more negative by multiplying by 234, but then make most of them positive by adding 934. (these are arbitrary values)
data = np.log(np.random.rand(N))*234 + 934

dataMin = min(data)
dataMax = max(data)

# min-max scale - this is the formula in python for min-max
# note that data is numpy structure and so subtracting dataMin/Max does that for each individual element in the numpy data structure.
dataS = (data - dataMin) / (dataMax - dataMin)

# plot the data against the newly scaled data (comparison visualization)
fig,ax = plt.subplots(1,2,figsize=(8,4))
# we add noise with np.random.randn(N)/20 because there are a lot of overlapping data points on the plot for the scaled data (makes it easier to see distribution, without it we just see a bar of data). This helps us see how many data points are clustered in the distribution. 20 is an arbitrary value.
ax[0].plot(1+np.random.randn(N)/20,data,'ks')
ax[0].set_xlim([0,2])
ax[0].set_xticks([])
ax[0].set_ylabel('Original data scale')
ax[0].set_title('Original data')

ax[1].plot(1+np.random.randn(N)/20,dataS,'ks')
ax[1].set_xlim([0,2])
ax[1].set_xticks([])
ax[1].set_ylabel('Unity-normed data scale')
ax[1].set_title('Scaled data')

plt.show()


### Normalize to any arbitrary range

# scale data from 4 to 8.7
newMin = 4
newMax = 8.7

# data Scale-scale - we take the unity scaled data and apply the formula for arbitrary range scaling
dataSS = dataS*(newMax-newMin) + newMin

print([min(dataSS), max(dataSS)]) # [4.0, 8.7]
```
