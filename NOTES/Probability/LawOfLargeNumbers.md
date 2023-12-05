# Law of Large Numbers

- As we repeat the same experiment over and over, the average of the sample means collected better approximates the population mean.
- As we get more and more repititions we can approximate more accurately what we don't know - the population mean (sometimes you cannot know the population mean feasibly)
  $$\lim_{n\to \infty}P(\left|{\bar{x}_n - \mu}\right| > \epsilon) = 0$$
  - Limit: what happens to this equation as $n$ gets really large (approaching infinity) where $n$ is the number of repititions
  - $\bar{x}_n$ is the average of all the sample means averaged together minus the population mean $\mu$ (which is something we don't know and want to do the experiments to better approximate it). We take the absolute value ($\left| ... \right|$), we only care how far off we are from the mean (not over or underestimating)
  - $\epsilon$ - Epsilon is used to indicate some arbitratily small number that is greater than 0
    - Note that epsilon represents the smallest number your computer/OS can represent in python as well.
    - Use in the formula is stating that: "The probability of an error between the sample mean average and the population mean being greater than 0 is approaching 0 ($... = 0$)." $\bar{x}_n - \mu$ can be thought of as an error (diff between samples mean and pop. mean)
- One sample is unlikely to represent a good estimate of the true population mean, but sampling many times (independent replications) can provide a accurate estimate of the population mean.

## Code: Law of Large Numbers Demonstration (Rolling die experiment)

```python
import matplotlib.pyplot as plt
import numpy as np

# Create an unfair die with unequal probabilities (die number = probability)
f1 = 2/8
f2 = 2/8
f3 = 1/8
f4 = 1/8
f5 = 1/8
f6 = 1/8
# these all sum to 1

# expected value - sum up all the probabilities of possible outcomes
expval = 1*f1 + 2*f2 + 3*f3 + 4*f4 + 5*f5 + 6*f6 # 3.0

# generate seed of outcomes (rolling 1s and 2s are twice as likely)
population = [1,1,2,2,3,4,5,6]
# stack this back to back 20 times
for i in range(20):
     population = np.hstack((population,population)) # this results in 8 million die rolls in the population

# random sample of 8 rolls:
sample = np.random.choice(population,8) # the mean will vary from the expected value being only one sample

nPop = len(population) # 8 million rolls

# experiment on large number of sample sizes
k = 5000 # max num of samples
sampleAve = np.zeros(k) # init sample averages

for i in range(k):
    # random index number, as i increases you draw more and more samples here. (next loop would be 2, then 3 samples etc. to 5000 samples)
    idx = np.floor(np.random.rand(i+1)*nPop) # [23.], next run: [54.,26.] etc. up to array of len 5000
    # draw one random sample from pop (will increase in number of points drawn on each iteration) and store the average of the sample.
    sampleAve[i] = np.mean(population[idx.astype(int)])

# plot sample averages
plt.plot(sampleAve,'k') # 'k' sets the plot lines to black color
# plot expected val as a red line
plt.plot([1,k],[expval,expval],'r',linewidth=4)
plt.xlabel('Number of samples')
plt.ylabel('Value')
plt.ylim([expval-1,expval+1])
plt.legend(('Sample avg','expected val'))
# what you'll see is that as you increase the sample sizes towards 5000, the average (blake lines) starts to waver less, but still never converges on the population mean (red line)

# if you take the mean of all the sample averages, then you get very close to the true pop mean of 3.0:
print(np.mean(sampleAve)) # 2.9999...

# Another way of computing law of large numbers using cumsum
populationN = 1000000
population = np.random.randn(populationN)
population = population - np.mean(population) # force mean to be zero

samplesize = 30 # take consistently small samples of 30 points
numberOfExps = 500
samplemeans = np.zeros(numberOfExps)

for expi in range(numberOfExps):
    sampleidx = np.random.randint(0,populationN,sampleSizes) # 30 random sample points
    # collect means of samples
    samplemeans[expi] = np.mean(population[sampleidx])

# plot the sample averages
fig,ax = plt.subplots(2,1,figsize=(4,6))
ax[0].plot(samplemeans,'s-') # blue sqaures show sample means infividually
ax[0].plot([0,numberOfExps],[np.mean(population),np.mean(population)],'r',linewidth=3) # plot true mean as red line
ax[0].set_xlabel('Experiment number')
ax[0].set_ylabel('mean value')
ax[0].legend('Sample means','Population mean')

# plot the average of the sample averages showing it converges to true mean

# Use cumsum to add the means together and divide by iterations
ax[1].plot(cp.cumsum(samplemeans) / np.arange(1,numberOfExps+1),'s-') # apparently np.arange gives array and is used with cumsum? Should just be num iterations not sure why this is used.
ax[1].plot([0,numberOfExps],[np.mean(population),np.mean(population)],'r',linewidth=3) # plot true mean as red line
ax[1].set_xlabel('Experiment Number')
ax[1].set_ylabel('mean value')
ax[1].legend('Sample means','Population mean')
```

# Central Limit Theorem

- The distribution of sample means approaches a Gaussian distribution regardless of the shape of the population/original distribution.
  - If we look at the distribution of all the sample means (before averaging them together a la Law of Large Numbers), then the distribution on a plot will look gaussian.
- Random samples from independent variables will tend towards a normal distribution even if the variables are non-normally distributed.

  - Random linear mixtures of signals that have non gaussian distributions will tend towards a combined distribution that is more gaussian, but only if they are roughly in the same scale.
  - You can take samples from different distributions and add them together (i.e. convolutions: A "convolution" (X+Y) of independent random variables from different distributions approaches a gaussian distribution)
  - Note: this depends on the sample count, scaling of the data or normalization and other factors.
  - For example if you add together distributions that are not scaled similarly, you don't get a guassian like distribution result. (see [video](https://www.udemy.com/course/statsml_x/learn/lecture/20014690#content) at timestamp 8:00-8:15)

- Note: the CLT is defined only for the mean, but in practice it holds for any sample statistic.

### Coding: CLT

```python
# create power law like data distribution
N = 1000000
data = np.random.randn(N)**2

plt.hist(data,40)
plt.show()

samplesize = 30
numberOfExps = 500
samplemeans = np.zeros(numberOfExps)

for expi in range(numberOfExps):
    sampleidx = np.random.randint(0,N,samplesize)
    samplemeans[expi] = np.mean(data[sampleidx])

# create a histogram of the sample means - this will show a gaussian like distribution
plt.hist(samplemeans,30)
plt.xlabel('Mean estimate')
plt.ylabel('Count')
plt.show()


### Demo of how combinations of non gaussian samples/signals can combine to form gaussian distribution/signal
# two datasets with non gaussian distributions
x = np.linspace(0,6*np.pi,10001) # create x axis
s = np.sin(x) # signal 1 (sin wave)
u = 2*np.random.rand(len(x))-1 # signal 2 (uniform noise), note the scaling (mult by 2 to get 0-2 and then subtract 1 to get -1 and +1 range). This puts this signal into the same y axis scale as the sin wave signal data.
### It is very important to scale different sets of data and normalize them before combining in order to get a gaussian distribution.

# plot sine wave data (signal 1)
fig,ax = plt.subplots(2,3,figsize=(10,6))
ax[0,0].plot(x,s,'b')
ax[0,0].set_title('Signal')
#plot sin wave distribution
y,xx = np.histogram(s,200)
ax[1,0].plot(y,'b')
ax[1,0].set_title('Distribution')
# plot noise data (signal 2)
ax[0,1].plot(x,u,'m')
ax[0,1].set_title('Signal')
# plot noise distribution
y,xx = np.histogram(u,200)
ax[1,1].plot(y,'m')
ax[1,1].set_title('Distribution')
# plot combined data
ax[0,2].plot(x,s+u,'k') # add/combine the sin wave and noise signals together
ax[0,2].set_title('Combined signal')
# plot distribution of combined data (will tend towards a Gaussian)
y,xx = np.histogram(s+u,200)
ax[1,2].plot(y,'k')
ax[1,2].set_title('Combined distribution')

plt.show()
```
