# Outliers

- Ambiguous - hard to determine sometimes what is and is not an outlier
- Alternative terms:
  - Anomaly
  - Extreme(deviant) data
  - Non-representative data
  - Noise

### The problem with outliers

- Many statistical methods use squared terms. This can result in producing massively magnified outliers that can skew results more badly.
- When sample sizes are small, outliers have a much higher impact - should be aware of this when working with small sets of data.

### Leverage

- The effect that an outlier has on the slope of a fitting line.
  - Outliers that do not have a large effect on the slope of a line (fitting some data points, for example, and you have one outlier on the y axis among many other in range data points) have Low Leverage.
  - If an outlier is high on the x and y axis, for example then it will affect the slope of the fitting line more and has High Leverage.
    - With large datasets the effect will not be as bad as with smaller data sets.
  - see [video](https://www.udemy.com/course/statsml_x/learn/lecture/20011194#content) at 9:00 timestamp around there
- Outliers can be more or less deleterious (bad) for your results depending
  - If they are near the edges of the data then the effect is greater
  - If they are in the middle dimension of data (scaling) then they are less effective

## Dealing with outliers

- One approach is to remove them from your data prior to analysis (assumes they are noise)
- Another approach is to leave them in the data, but use robust methods that remove the negative impact on the results (assumes that outliers are unusual but valid data)
  - You can use non-parametric T-tests, Spearmon correlations, permutation testing, weighted regression and iteratively re-weighted regression for example
- Always assume there are outliers in your data - check the data carefully.

### Identifying Outliers

#### Using the Z-score method

- Appropriate for roughly Guassian distributions

Steps:

- Convert data to z-score (scaled to standard deviations)
- Pick a standard deviation threshold (usually of 3 std devs above or below, for example) and look for datapoints that exceed that threshold (are outside of 3 std devs for ex.) - these are outliers
  - Note that the threshold is subjective and somewhat arbitrary - it's arrived at after inspection of the data.
  - There can also be edge cases like values that just straddle the threshold for example, so something to watch out for.

#### Using the iterative Z-score method

- same as z-score method, but you repeat the steps until there are no more outliers

Steps:

- Convert data to z-score (scaled to standard deviations)
- Pick a standard deviation threshold (of above or below 3 std devs, for example) and look for datapoints that exceed that threshold - these are outliers
- Remove the outliers from the data and repeat the steps on the cleaned data until there are no more outliers.
  - The Z-score translation will be different since the previous outliers were removed (mean pulls up and std dev gets larger)
- This iterative approach should be used with caution - it tends to remove data points that are still valid but just close to the edges of normal data ranges.
- Generally not recommended to use this, but there are cases where it might work well.

#### The Modified Z-score method

- Used when the data is not similar to a normal or gaussian distribution (Non-gaussian distributions)
- Can be useful for long-tailed distributions

Steps:

- Replace the regular z-score scaling with the modified z-score scaling
  - based on medians instead of means (instead of subtracing mean from data points, we subtract the median $\tilde{x}$ from each data point in the set)
  - Formula: $$M_i = {.6745(x_i-\tilde{x})\over MAD}$$
    - $MAD$ is the **Median Absolute Deviation**: $median(\left|x_i-\tilde{x}\right|)$
      - With MAD: take each data point $x_i$ and subtract the median $\tilde{x}$, then take the absolute value of the set of those numbers, and take the median of that set of numbers.
    - $.6475$ is the standard deviation unit corresponding to the 75th percentile or third quartile of the gaussian distribution. This is done to normalize the modified z-scores to be more similar to the regular z-scores. see [answer to question](https://www.udemy.com/course/statsml_x/learn/lecture/20011208#questions/14750956) for more info. the value is arbitrary.
- Repeat previous z-score methods above
  - Convert data to z-score (scaled to standard deviations)
  - Pick a standard deviation threshold (usually of 3 std devs above or below, for example) and look for datapoints that exceed that threshold (are outside of 3 std devs for ex.) - these are outliers

## Code: Removing and dealing with outliers

```python
import numpy as np
import matplotlib.pyplot as plt
from statsmodel import robust
import scipy.stats as stats

# generate some data and force some outliers
N = 40
data = np.random.randn(N)
# force outliers
data[data<-1] = data[data<-1] + 2 # [data<-1] is a boolean mask and selects all the data points that match the condition.
data[data>2] = data[data>2]**2 # square to increase changes of outliers
data = data*200 + 50 # stretch/change scale for the distribution to make it more interesting to comparison with z

#convert data to z scale
dataZ = (data-np.mean(data) / np.std(data)) # z-score formula

# Specify z-score threshold (how man std dev away from the mean which represents an outlier)
zscorethresh = 3

fig, ax = plt.subplots(2,1,figsize=(8,6))

ax[0].plot(data,'k^',markerfacecolor='w',markersize=12)
ax[0].set_xticks([])
ax[0].set_xlabel('Data index')
ax[0].set_ylabel('Orig. scale')

# plot z scores
ax[1].plot(dataZ,'k^',markerfacecolor='w',markersize=12)
ax[1].plot([0,1],[zscorethresh,zscorethresh],'r--') # show threshhold as red dotted line on the graph (should also draw this on the negative side if applicable)
ax[1].set_xlabel('Data index')
ax[1].set_ylabel('Z distance')
plt.show()

## Find the outliers
# any point that has an absolute value of the z normalization (we flip the negative signs to positive here to capture outliers above and below the 0 line)
# The value of the dataZ point is in z normalized scale (how many std dev from mean)
outliers = np.where(abs(dataZ) > zscorethresh)[0]

# plot the outlier removals
ax[0].plot(outliers,data[outliers],'x',color='r',markersize=20) # these put a red x over the outliers previously plotted to show they are to be removed.
ax[1].plot(outliers,data[outliers],'x',color='r',markersize=20)

fig

## At this point we could remove the data or replace with NaN
#### NOTE: it is often more convenient to replace the outliers with NaN - you still have the same number of data points that way and if you're aligning them with other features/visualization then that can help.
```

### Code: Iterative Z-score method

```python
# note: this is a lenient threshold and only used for demonstration
zscorethresh = 2 # normally 2 std devs is too lenient for removing data, should be higher
dataZ = (data=np.mean(data)) / np.std(data)

colorz = 'brkm'
numiters = 0 # num of iterations/counter

# use a while loop as we don't know how many iterations we'll need
while True:
  #convert to z scale
  datamean = np.nanmean(dataZ)
  datastd = np.nanstd(dataZ)
  dataZ = (dataZ-datamean) / datastd

  # collect values that exceed the threshold (again use absolute value to capture values that fall outside of 2 std dev in the negative direction)
  toremove = np.abs(dataZ)>zscorethresh

  # break if no points to remove (the sum of the empty vector toremove will be zero)
  if sum(roremove)==0:
    break
  else:
    #mark outliers on plot
    plt.plot(np.where(toremove)[0],dataZ[toremove],'%sx'%colorz[numiters],markersize=12)
    dataZ[toremove] = np.nan # replace data point with NaN

  # replot - this will show which points were deleted as color coded Xs on the graph
  plt.plot(dataZ,'k^',markersize=12,markerfacecolor=colorz[numiters],label='iteration %g'%numiters)
  numiters = numiters + 1

plt.xticks([])
plt.ylabel('Z-score')
plt.xlabel('Data index')
plt.legend()
plt.show()

# print removed:
removeFromOriginal = np.where(np.isnan(dataZ))[0]
print(removeFromOriginal)
```

### Code: Modified Z-score

```python
import statsmodels.robust as robust

# compute modified z (based on medians instead of mean)
dataMed = np.median(data)
dataMAD = robust.mad(data) # not sure where robust comes from - what library?

# subtract median instead of mean and instead of dividing by std dev divide by the Median Absolute Difference/Deviation
dataMz = stats.norm.ppf(.75)*(data-dataMed) / dataMAD # .ppf computes the probability density (ppf stands for percent point function). It returns a z-value for a given p-value.

#plot it
fig,ax = plt.subplots(2,1,figsize=(8,6))

ax[0].plot(data,'k^',markerfacecolor='w',markersize=12)
ax[0].set_xticks([])
ax[0].set_xlabel('Data index')
ax[0].set_ylabel('Orig. scale')

# plotting z scores
ax[1].plot(dataZ,'k^',markerfacecolor='w',markersize=12)
ax[1].plot([0,1],[zscorethresh,zscorethresh],'r--') # show threshhold as red dotted line on the graph (should also draw this on the negative side if applicable)
ax[1].set_xlabel('Data index')
ax[1].set_ylabel('Median dev. units (Mz)')
plt.show()
```

## Z-score method for Multi-variate data

- Multivariate means that you have a space that plots more than one variable (note: the space is not thought of as a traditional plane)
- Note that outliers in multivariate space could have properties such as they are not extreme in one dimension but extreme in others, and so still considered an outlier

### Euclidian Distance

- Distance between two points in a space - generalizes to any number of dimensions (you sum x number of squared distances)
- For 2 dimensions: distance between two points (take the difference of x coords of first and second point, square it and add it to the difference of y between 2 points squared, then take the square root of that):
  $$d_{a,b} = \sqrt{(a_x - b_x)^2 + (a_y - b_y)^2}$$
- For 3 dimensions:
  $$d_{a,b} = \sqrt{(a_x - b_x)^2 + (a_y - b_y)^2 + (a_z - b_z)^2}$$

### Multivariate Mean

- The mean on all dimensions (for example the x,y position where x is the mean of all x axis values and y is the mean of all y axis values)
- In addition to computing the euclidian distance of points on a multivariate plane, you can also compute the distance from any data point to this multivariate mean point

### Z-score method (for multivariate data)

Steps:

- Compute the multivariate mean
- Compute the distance from each data point to that mean
- Convert the distances to Z-scores
- Pick a standard deviation threshold and look for datapoints that exceed that threshold - these are outliers
  - points that are close to the mean, for example, will have a z-score distance that is smaller

### Code: Multivariate data

```python
# two dimensional data
d1 = np.exp(-abs(np.random.randn(N)*3)) # arbitrary values to vary the data
d1 = np.exp(-abs(np.random.randn(N)*5))
# multivariate mean (the mean along each dimension, feature dimension 1 and feature dimension 2):
datamean = [np.mean(d1),np.mean(d2)] # list of each mean for it's dimension

# compute distance of each point to the mean (of all the data)
distance = np.zeros(N)
for i in range(N):
  distance[i] = np.sqrt( (d1[i]-datamean[0])**2 + (d2[i]-datamean[1])**2 ) # formula for euclidian distance

# convert to z scale (Note: we're overwriting the original data in original units here - often you'll want another variable to keep the original scaling if needed, here we overwrite the distances with the z normalized units)
distance = (distance-np.mean(distance) / np.std(distance))

# plot data
fig,ax = plt.subplots(1,2,figsize=(8,6))

ax[0].plot(d1,d2,'ko',markerfacecolor='k')
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[0].set_xlabel('Variable x')
ax[0].set_ylabel('Variable y')

# This plots the multivariate mean (of all data) as a green mark
ax[0].plot(datamean[0],datamean[1],'kp',markerfacecolor='g',markersize=15)

# then plot those distances - the points plotted correspond to the z-score distance for each data point
ax[1].plot(ds,'ko',markerfacecolor=[.7,.5,.3],markersize=12)
ax[1].set_xlabel('Data index')
ax[1].set_ylabel('Z distance')

# Set a threshold for outliers
distanceThresh = 2.5

# find the outlier points:
oidx = np.where(ds>distanceThresh)[0] # oidx is offending index
print(oidx)

# cross out these points on the plot
ax[1].plot(oidx,ds[oidx],'x',color='r',markersize=20)
ax[0].plot(d1[oidx],d2[oidx],'x',color='r',markersize=20)

fig
```

### Data Trimming to remove outliers

Steps:

- Mean center the data first
  - Note: This may or may not make sense to do depending on your data set - i.e. if you expect all outliers to be positive, then you don't need to do this (it turns negative into positive values)
- Sort the mean-centered data
  - We sort the data to make it easier to identify outliers regardless if they are positive or negative (to the right or left of the distribution) - all the values will be positive due to mean centering
- Remove the ost extreme k values, or the most extreme k%
  - You pick and set the parameter k - remove 2 values, or remove extreme 5% of data etc.

NOTE: With Data trimming the thresholds are subjective and require thought and judgement. There is potential to remove valid data and misidentifying outliers.

### Code: Trimming data (removing outliers)

```python
N = 40
# create gaussian distributed random noise (randn)
data = np.random.randn(N,1) # expected mean is zero and the expected variance is 1
data[data<-2] = -data[data<-2]**2 # these lines just increase the probability of finding outliers in the dataset (both on the left and right side of the distribution)
data[data>2] = data[data>2]**2

# mean centered data
dataMC = data - np.mean(data)

# plot the data (this will show points as yellow triangles)
fig,ax = plt.subplots(1,1)
ax.plot(data,'k^',markerfacecolor='y',markersize=12)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel('Data index')
ax.set_ylabel('Data value')
plt.show()

###### Removing the k% percentile

# percent to remove
trimPct = 5

# identify the cutoff - note we use abs() to turn negatives into positives
datacutoff = np.percentile(abs(dataMC),100-trimPct) # this computes 95th percentile

# find data that exceeds cutoff
data2cut = np.where(abs(dataMC)>datacutoff)[0]
# mark outliers with red x
ax.plot(data2cut,data[data2cut],'rx',markersize=15)
fig

##### Removing the extreme k values
k2remove = 3
# sort the data
datasortIdx = np.argsort(abs(dataMC),axis=0)[::-1]
# remove the most extreme k values (first three values)
data2cut = np.squeeze(datasortIdx[:k2remove])

# mark with red xs
ax.plot(data2cut,data[data2cut],'go',markersize=15,alpha=.5)
ax.legend(('All data','%g%% threshold'%(100-trimPct),'%g-value threshold'%k2remove))
fig
```

## Using Alternative statistical methods robust to Outliers

- Outliers may be considered valid, but unusual data in some cases.
- non-parametric analyses are robust to outliers and are not as affected by them. (parametric tests are more sensitive to outliers)

### Types of tests

- Many parametric tests often have a corresponding nonparametric test
  - 1-sample t-test => Wilcoxon sign-rank test
  - 2-sample t-test => Mann-Whitney U test
  - Pearson correlation => Spearman correlation
  - ANOVA => Kruskal-Wallis test
- Nonparametric tests are usually based on medians or ranks, which are insensitive to outliers (as opposed to mean based methods for example)

## Nonlinear data transformations (for normalization)

- Nonlinear transformation can make outliers that are incorrectly identified as such less extreme
- The goal is to transform the data to make linear methods (std dev, mean, assuming gaussian data etc.) valid, or to make the data distribution approach Gaussian.

### Most common nonlinear transformations:

- Rank-transform (values translated into rankings based on their relative value i.e. ranking 1 through 5 etc.)
- Square root - similar to logarithm transformation. The main idea is that smaller numbers (on y axis, output for ex) are spread further apart and larger numbers (on the y axis) are compressed more. See [video](https://www.udemy.com/course/statsml_x/learn/lecture/24008986#content) at 6:15 timestamp. Useful if transforming a power distribution, for example.
- Logarithm - similar to square root method - transforms nongaussian data to a more gaussian distribution scale. Used only for positive values/non-negative data (same a square root) - would be useful if dealing with a log normal distribution for example.
- Fisher-z: useful for transforming a uniform distribution to a gaussian like distribution

### Pitfalls/cautions with nonlinear transformations

- How transformation affects spacing in data points is not knowable in advance all the time (it might depend on the data). Ex: rank-transform will bury the spacing between the original data (large spreads between ranks will not be visible)
- There is a nonlinear relationship between the spacing of values in the transform and the spacing of the values in the original data.
  - Most statistical models are linear, so the results must be interpreted interms of the transformed data, not the original!
  - Always be careful to interpret your results in the context of the transformation you've applied (not in terms of the context of the original data).
