# Z-score

- ubiquitous method of normalizing data
- Addresses a way to compare values across different scales or units (i.e. height to weight (cm vs. kg)).
  - We normalize measurements in comparison to a unit-less scale
- We transform the data to a number that is relative to its distribution or sample (we don't use about the actual number of the individual data point for comparisons)
  - ex: our weight is average (near the mean), and the height is one standard deviation below the average.
  - It's easier to interpret a value relative to it's distribution than on it's own!

### Formula: Z-transform

$$z_i = {x_i - \bar{x}}\over\sigma_x$$

- Subtract the mean from the data point (mean-center)
- normalize by std dev: divide by the standard deviation
- This results in a unit of standard deviations (unit-less practically speaking)
  - the z-score is for the individual data point: z of point i.
- Z-transform shifts and stretches the data, but does not change it's shape (distribution)
  - For example if you z-transformed all data points and plotted them in histogram the distribution overall shape would be the same as the original data shape.
  - It only changes the scaling on the y axis in a plot for example.
- The nature of this formula results in the mean of the transformed data being zero and the std deviation of the z transformed data being 1.

### Assumptions when using the Z-Score

- The mean and standard deviation are valid descriptions of the distribution's central tendency and dispersion (roughly gaussian distributions).
  - In other words, it assumes we're dealing with a roughly Gaussian distribution (not perfect) but generally so that these measures are meaningful
  - Datasets that are bimodal for example will yield hard to interpret z-scores since the mean and std dev are not as meaningful. Technically you could still apply the z-score to non gaussian shaped distributions in special cases.

## Code

- z-score function is in the `scipy.stats` module

```python
from scipy import stats

# poisson noise of a 1000 data points with a param of 3, and square all of them (no reason, just arbitrary data)
data = np.random.poisson(3,1000)**2

# get the mean and std dev for the data to compare to the z-score
datamean = np.mean(data)
# remember to use the correct degrees of freedom (ddof=1 the "biased" std dev if using sample data - almost always need this) if using numpy std or variance function!
datastd = np.std(data,ddof=1)

# Note you can also call these directly on numpy data (same result as lines above)
# mean = data.mean()
# std = data.std(ddof=1)

# calculate a z-score manually (sometimes you need to do this, i.e. permeutation testing)
dataz = (data-datamean) / datastd

# alternatively can use the scipy function
dataz = stats.zscore(data)

# compute mean and std deviation of the z-score data
dataZmean = np.mean(dataz) # mean of z-transformed data is 0
dataZstd = np.std(dataz,ddof=1) # std dev of z-transformed data is 1

plt.plot(dataz,'s',markersize=3)
plt.xlabel('Data index')
plt.ylabel('Data value')
# note we round only to make the title less long
plt.title(f'Mean = {np.round(dataZmean,2)}; std = {np.round(dataZstd,2)}')

plt.show()
```
