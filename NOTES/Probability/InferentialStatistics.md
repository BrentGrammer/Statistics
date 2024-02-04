# Inferential Statistics (Probability)

- The goal in Inferential Statistics is to determine the Signal to Noise Ratio.
- Inferential Statistics is all about probability
- **Probability**: numerical description of how likely an event is to occur or how likely it is that a proposition is true.
  - A number between 0 and 1.
  - The sum of all probabilities in a set must sum to 1.
  - Note confidence is not the same as probability. 20% chance of rain means there is a 1 in 5 chance of it raining, not that we have 20% confidence it will rain.
- Developed by Bernoulli family and Bayes (originally as a way to predict gambling outcomes possibly)
- Probability is needed when there is some kind of uncertainty

### Probability vs. Proportion

- Proportion is a fraction of the whole (while probability is the likelihood of an event occurring or that a statement is true)
- These can be easily confused sometimes.

#### Examples:

- Eating chocolate on Friday:
  - The proportion of eating chocolate during the week is 1/7
  - The probability that a randomly selected day is a day that you eat chocolate is 1/7 (incidently the same as proportion, but different meaning)
  - The probability that you eat chocolate during the week is 1 (100%)
- Some illness mortality rate is 3.4%
  - False statement: you have a 3.4% chance of dying from illness complications
    - The key here is that the statement refers to me. The probability of death depends on age (and other factors like health status, co-morbidities, etc.). I am young and healthy, so the probability for me to die of covid is smaller. That number 3.4% is based on the entire population.
  - True: The probability of a randomly selected infected patient will die from illness is 3.4%
    - Note from above that this is because you are sampling from the entire population. The probability changes for the individual.
  - True: The proportion of infected people who die from illness is .034 (3.4%)
    - Involves **Conditional Probability** in addition as the higher the age, the higher the fatality rate.
- 10 coin flips (6 heads and 4 tails):
  - The proportion of heads is 60% (6/10)
  - The probability of getting heads is 50% (each flip has the same probability of .5)
  - Slightly different: the probability of choosing a random coin that is heads out of the ten is 60%.

## Computing Probability

### Valid data types for computing probability:

- Valid: Discrete, Ordinal and Nominal data
- Invalid: Interval and Ratio data are not valid for computing probability (unless converted into discrete numeric/range data)
- The reason you can't compute probability on this type of data is that they have arbitrary precision and aren't countable things.
  - Example it doesn't make sense to measure probability of a penguin height being 65.23288929873483 cm. (you would instead get the probability of a height range, i.e. between 60 and 70 cm)
- Data must have mutually exclusive labels or bins (the total of all probabilities must sum to 1 or 100%)
  - i.e. a coin cannot be tails and heads at the same time, a card cannot be queen of hearts and ace of hearts at the same time, a dice has six unique sides, etc.
  - You cannot compute probability for something like survey of where people get news sources (people can get their news from more than one source.)

### Probability Formula

$$p_i = {(100 * )c_i \over {\sum c}}$$

- The probability of event $i$ is the count $c$ of that event divided by the total number of all events $\sum c$
- Optionally you can multiply by 100 $(100 *)$ to get the percent value.
- Example the probability of getting heads would be plugged in as $c_i = 1$ for heads (1 event) divided by the total possible events (2 for heads or tails) $\sum c = 2$
- Example: a jar filled with colored marbles, what is the probability of picking out a marble of a particular color?
  - 40 blue, 30 yellow, 20 orange (total marbles is 90)
  - Blue: 40/90 = .444
  - Yellow: 30/90 = .333
  - Orange: 20/90 = .222
  - Note that these sum up to 99.9% (the reason is these are truncated results, they actually do add to 100% or 1)

### Code: Computing Probability

```python
import matplotlib.pyplot as plt
import numpy as np

# array representing a count of events - i.e. num times it rains in a week etc.
c = np.array([1,2,4,3]) # events sum up to 10
# probability in percent
prob = 100*c / np.sum(c) # [10.,20.,40.30.]

## Marbles in a jar exercise
blue = 40
yellow = 30
orange = 20
totalMarbs = blue + yellow + orange

# put all in a jar - large vector with a bunch of 1s,2s, and 3s (40 1s, 30 2s, 20 3s)
# hstack is horizontal stacking - all the numbers are sorted
jar = np.hstack((1*np.ones(blue),2*np.ones(yellow),3*np.ones(orange)))

# draw 500 marbles
numDraws = 500
drawColors = np.zeros(numDraws)

for drawi in range(numDraws):
  # draw random index (random num between 0 and 1 multiplied by num of marbles)
  randmarble = int(np.random.rand()*totalMarbs) # convert decimals to int
  # store color of marble
  drawColors[drawi] = jar[randmarble]

#Proportion
propBlue = sum(drawColors==1) / numDraws
propYell = sum(drawColors==2) / numDraws
propOran = sum(drawColors==3) / numDraws

# plot against the theoretical probability (probability are thin dark blue lines and proportion are thick bars) This illustrates the difference between proportion and probability
plt.bar([1,2,3],[propBlue,propYell,propOran],label='Proportion')
plt.plot([0.5,1.5],[blue/totalMarbs,blue/totalMarbs],'b',linewidth=3,label='Probability')
plt.plot([1.5,2.5],[yellow/totalMarbs,yellow/totalMarbs],'b',linewidth=3)
plt.plot([2.5,3.5],[orange/totalMarbs,orange/totalMarbs],'b',linewidth=3)

plt.xticks([1,2,3],labels=('Blue','Yellow','Orange'))
plt.xlabel('Marble Color')
plt.ylabel('Proportion/probability')
plt.legend()
plt.show()

# Note that this shows a probability mass function (dealing with discrete events and data)

```

### Odds vs. Probability

- Odds are closely related to probability, but not the same thing
- often odds are used in gambling or disease
  - Ex: "The odds of winning are 6:1 against" - the odds ratio can be a fraction 6/1
- The odds are a probability ratio of two probabilities:
  - The probability of an event not occuring (1 - p)
  - The probability of the event occurring (p)
  - $r = {1-p \over p}$
  - solved for p: $p = {1 \over {1+n/m}}$ (n/m is the odds ratio)
  - Ex: odds of drawing a king in deck of cards is 12:1 against: $r = {1-{4/52}\over{4/52}}$
    - The probability of drawing a random card that is not a king is 48/52 (or 1 - 4/52)

## Probability mass and Probability Density

Probability Mass: A function that describes probabilities for a set of exclusive **discrete events**.
Probability Density: A function that describes probabilities for a set of exclusive **continuous events**.

### Probability functions

Note: $X$ in the following functions denote all of the values of a Variable (it represents the set of all possible data values under consideration)

- little $x$ represents the unique values that X could be.

$$f(x) = p(X = x)$$

- A probability function is a function of variable $x$ that is defined by the probability of that event $x$ ocurring given the set $X$. The function is defined by the probability of each event ocurring.
  - The function outputs the probability of that particular event $x$ occuring
- $s.t.$ - Such That (conditional substatements on the function definition)
  $$
  f(x_i) = p(X = x_i) \\
    \quad s.t. \space p(X = x_i) \ge 0 \\
    \quad s.t. \space p(X \ne x_i) = 0 \\
    \quad s.t. \space \sum{p}(X = x_i) = 1 \\
  $$
  - This says that probabilities must be non-negative (such that p is greater than equal to 0), and such that events must be exclusive (the probability of all other x's $x_i$ in the data must be zero, only the current x in question and one x can be calculated, no simulatneous calculations), and such that the sum of probabilites of all data points must sum up to 1 (or 100 if working with percent)

### Probability Mass functions

- Used for discrete events (i.e. picking a card from a deck, picking out a marble from a jar, or guessing a side on a 6 sided die)
- Visualizations must be a bar plot or a histogram. (events on x axis, and probability on y axis)
- These are more common than density functions in practice for things like deep learning.

### Probability Density Functions

- used for continuous events (height, distance, age, not easily countable or precise, i.e. you're not 24 years old, you're in between 24 and 25 years old at any given point)
- These have curves and are smooth
  $$P(a \le x_i \le b) = \int_{a}^{b} f(x_i)dx$$
  - probability of an x being between a and b is equal to the integral of the boundaries a and b of the function f(x) and dx. (uses calculus)
- Density functions are more common in theoretical mathematical settings (not necessarily deep learning or in practice)
  - Note: in computers you cannot represent an actual analog signal, so we deal with mass functions since they are discrete in nature

### Code: Probability Mass/Density

```python
# we can't actually generate a continuous signal on computers, so we start with a lot of points to simulate that
N = 10004
datats1 = np.cumsum(np.sign(np.random.randn(N))) # generate brownian noise, cumulative sum of random 1s and -1s
datats2 = np.cumsum(np.sign(np.random.randn(N)))

# plot the random data to see it (Brownian noise)
plt.plot(np.arange(N),datats1,linewidth=2)
plt.plot(np.arange(N),datats2,linewidth=2)
plt.show()

# Plot the probability distributions (approximated density function)
# discretize using histogram
nbins = 50

y,x = np.histogram(datats1,nbins)
x1 = (x[1:]+x[:-1])/2
y1 = y/sum(y)

y,x = np.histogram(datats2,nbins)
x2 = (x[1:]+x[:-1])/2
y2 = y/sum(y)

plt.plot(x1,y1,x2,y2,linewidth=3)
plt.legend(('ts1','ts2'))
plt.xlabel('Data value')
plt.ylabel('Probability')
plt.show()

# note that you could argue that we're showing proportion vs. probability here, but if we pick a random point of the data at any point on the time axis, then it can be interpreted as probability. Sometimes proportion and probability can overlap. and depend on how you interpret the data or what question you're asking about the data.
```

### Cumulative Distribution Functions (a.k.a. Cumulative Density Functions)

- see [video](https://www.udemy.com/course/statsml_x/learn/lecture/20014640#content)
- also [video](https://www.udemy.com/course/statsml_x/learn/lecture/20249434#content) at timestamp 11:22 for concise visual and explanation.
- sometimes abbreviated 'cdfs'
- A CDF is the cumulative sum (or integral) of the probability distribution (or density)
  - The y-axis value at each x-value is the sum of all the probabilities to the left of that x-value.
  - Starts at zero and increases monotonically to 1. (note that the sum of a cdf is more than 1 naturally)
- Start from a probability distribution (could be discrete probs or continuous/density probability function) for example, a pdf (Probability Density Function) and sum up all the values to the left of a point on the x axis - that is the value of the CDF at that point.
  - The CDF is the continuous sum (a.k.a. Integral when working with continuous functions) to the left of a point on the x axis of a plot of a continuous probability series (a probability density function plot)
- CDFs always get to/approach 1 on the plot (y axis)
  - As you go right on the x axis of a CDF, the probabilities have been exhausted (The previous points of the pdf are cumulating)

$$C(x_a) = {\sum_{i=1}^a}p(x_i)$$

- the sum of the values beginning at $x_i$ up until $a$ (the current location on the x-axis)

#### Use of CDFs

- Used to evaluate the probability of obtaining a value up to X or at least X
- Example question: What is the probability of getting a score at least 1 std dev higher on the SATs than average?
  - We would take the CDF of SAT scores (which are gaussian distributed), plot the std dev units on the x-axis and probability cumulation on the y-axis. Start a 1 on the x-axis and get the y-value on the cdf curve and that represents all scores at or below 1 std above average, so we take the remaining difference from 1 (for ex., std 1 is .85 on y, so the answer is .15 or 15%)
  - Be careful about the problem question/goal - do you want more or less than the target std dev., if less then the answer is the exact value on the y axis.

```python
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

# it's important to use linear space (x points are equidistant from each other) with cdfs
x = np.linspace(0,5,1001)

# use .pdf and .cdf from stats package to make density and cumulative distribution functions
p1 = stats.lognorm.pdf(x,1) # using lognorm for ex, but this applies for any other kind of distribution
c1 = stats.lognorm.cdf(x,1) # second input is the distribution scaling

p2 = stats.lognorm.pdf(x,.1) # the 1 and .1 will determine how spread out the plot is (higher value results in a narrower distribution showing)
c2 = stats.lognorm.cdf(x,.1)

# draw the pdfs
fig,ax = plt.subplots(2,1,figsize=(4,7))

ax[0].plot(x,p1/sum(p1))
ax[0].plot(x,p1/sum(p1), x,p2/sum(p2))
ax[0].set_ylabel('probability')
ax[0].set_title('pdf(x)')

# draw cdfs superimposed on plot for their pdf
ax[1].plot(x,c1)
ax[1].plot(x,c1,x,c2)
ax[1].set_ylabel('probability')
ax[1].set_title('cdf(x)')
plt.show()
```

### Creating Sample Estimate Distributions

- It is impossible to answer the question, What is the population parameter of giraffe height? (i.e. how tall are giraffes).
  - We cannot measure ALL Giraffes, so it's impossible to get a true answer to this question.
  - We can on the other hand measure a sample of giraffes
- Sample Distribution: take N samples from a population and make a distribution of their means. (i.e. each mean for each sample is plotted on a histogram)
  - The mean as a parameter is just an example, you can do this for any parameter outside of the mean as well, variance/std, median, etc.

#### Random vs. Respresentative Sampling

- Sample estimates can be generalized only to the population that the sample represents.
- Example: Is it warmer in Netherlands than in Vietnam?
  - If we take random samples of temps in July in Netherlands and samples of temps in January for Vietnam, this is not representative of comparing apples to apples (Netherlands will appear to be warmer than Vietnam). Instead we need to ask "Is it warmer in Amsterdam than in Hanoi in July?"

### Monte Carlo Sampling ('Random Sampling')

- "Monte Carlo" is a general term that refers to many different types of sampling
- A method of randomly sampling the solution space instead of doing the full working out of a problem to solve a problem. It's an estimate of what the solution to a problem could be.
- Useful for exceptionally hard or complex problems. Comes up a lot in Bayesian statistics.
- "Monte Carlo Sampling" is a specific method of Monte Carlo whih is the same as randomly sampling from a population to estimate an unknown population parameter (since you can't measure the whole population, but samples are feasible)
  - There are variants of this sampling like Markov Chain Monte Carlo Sampling, etc. where the way you pick one sample depends on how you pick the previous sample

## Sampling Variability (noise and other annoyances)

- Different samples from the same population can have different values of the same measurement.
  - A single measurement is likely to be an unreliable estimate of a population parameter.
- Sources of variability:
  - Natural variation: biology, natural phenomenon. (but in some physics there is low variability for fixed measurements or laws)
  - Measurement noise: Sensors are imprefect or not precise enough to measure what you want with sufficient exactness.
  - Complex systems: Not taking into account all factors (example: measuring height while ignoring age)
  - Stochasticity(inherent randomness)

### Dealing with variability

- Take many samples: Average together many samples to get an approcimate true population mean. (samples will be below and above the average and their mean will converge towards the pop mean, i.e. law of large numbers)
- Measure confidence intervals of sample based parameter estimates (these will tell you about the relationship between the parameter estimates and the true parameter values in the population)

### Code: Variability

- Tip: if you are having trouble lining up different distributions for visual comparison on plot (i.e. the y axis scaling is a problem), you can normalize the y axis so they all have a maximum range of 1. This puts all distributions on the same scale in the y axis for comparing shape.

```python
import scipy.stats as stats

# create theoretical normal distribution (has a lot of points)
x = np.linspace(-5,5,10101)
theoNormDist = stats.norm.pdf(x)

# this discretizes the theoretical distribution above into bins (10101 bins!)
#theoNormDist = theoNormDist*np.mean(np.diff(x)) # The purpose of this normalization is to scale the distribution so that its total area under the curve (the sum of all probabilities) is approximately 1, making it a valid probability density function.
# note that this will not be good and show as a flat line - we use density=True in the plt.hist() function to fix this problem, but this should be commented out.

#NOTE: ways of normalizing (https://www.udemy.com/course/statsml_x/learn/lecture/20014670#questions/13039330)
# dividing by the sum vs. x diff mean: they both work for different reasons. Normalizing by x[1]-x[0] is the "calculus approach" because you are scaling by dx to compute the areas of the probability density (think of Riemann sums). Dividing by the sum is the "empirical approach" because you're just forcing the distribution to sum to 1.
# Those two methods will only be the same if you have a sufficient x-axis range. For example, instead of evaluating the distribution from -4 to +4, try it with -1 to +1. You'll see that the calculus approach no longer sums to 1 because you aren't sampling enough of the distribution.

# note this is an alternative to avoid the above.
# histogram(sampledata,'Normalization','pdf')
# In this way, there is no need to normalize theoNormDist.

numSamples = 40 # note how the blue hist in the plot does not align with the theoretical normal dist, but if we increase this to 400 samples, you'll see the histogram more closely resemble a normal distribution

sampledata = np.zeros(numSamples)

for expi in range(numSamples):
  # draw sample data randomly from a normal distribution
  sampledata[expi] = np.random.randn()

## Compare the theoretical distribution against a histogram of the analytic distribution of samples
plt.hist(sampledata,density=True) # Using density=True normalizes both distributions to a good range for comparison. shows analytic distribution of samples as blue histogram bins
plt.plot(x,theoNormDist,'r',linewidth=3)  # shows theoretical normal distribution curve as red line
plt.xlabel('Data values')
plt.ylabel('Probability')
plt.show()


##### Sample vs. Population parameters

# population data (mean is 0)
populationN = 1000000
population = np.random.randn(populationN)
population = population - np.mean(population) # this makes the mean zero (general formula is dataval_i - datamean)

# take random sample (30 data points) from population
samplesize = 30
sampleidx = np.random.randint(0,populationN,samplesize) # generate 30 random data points from 0 up to the pop (1 mill excl.)
samplemean = np.mean(population[sampleidx]) # take the 30 sample indexes and get their mean
# the sample mean will probably be far from the pop mean (0) because sample size is small.


# now try with a range of sample sizes (to experiment with increasing the sample size to get closer to real mean)
samplesizes = np.arange(30,1000) # [30,31,32,...,999]

samplemeans=np.zeros(len(samplesizes))

for sampi in range(len(samplesizes)):
  sampleidx = np.random.randint(0,populationN,samplesizes[sampi]) # (start,end,number to generate)
  samplemeans[sampi] = np.mean(population[sampleidx])

# plot the sample means agains the true population mean to see if as the sample size gets larger, the sample mean converges with the true mean
plt.plot(samplesizes,samplemeans,'s-') #s- is a square marker connected with a solid line
# plot the true mean as a red horizontal line on the plot, x coords are the first and last sample mean val
plt.plot(samplesizes[[0,-1]],[np.mean(population),np.mean(population)],'r',linewidth=3)
plt.xlabel('sample size')
plt.ylabel('mean value')
plt.legend()
plt.show()
# what this will show is that the means for the samples bounce around all over the place (though get generally narrower) even at high sample sizes (i.e. 1000). The means of the individual samples do not converge to the true mean if you increase sample sizes.

# if we take the mean of the sample means, though...
np.mean(samplemeans[:12]) # even with only twelve sample means, the mean of these sample means will be much closer to zero than the individual means (even with larger sample size)
```

## Expected Value

$$E[X] = {\sum_{i=1}^n}x_i p_i$$

- $E[X]$ is the expected value of the sample $X$
- $x_i$ is the i-th index in the data set, $n$ is the total number of elements in the sample, $p_i$ is the probability of observing value $x_i$

### Average vs. Expected Value

- Average: an empirical sample estimate based on a finite set of data.
- **Expectation Value:** the expected average in the population, or from a very very large number of samples (approaching infinity)
  - Does NOT involve sample data.
  - The average of the population, or the average of a very large number of samples from the population
  - Example: the expected value of a weighted die (1,2 have .25 p and 3,4,5,6 have .125 probability) is 3. This is different from the average (you need sample die rolls for the average)
  - If you were to repeat many series of dice throws (say 8 dice throws) and take the average outcome of all those roll samples, then you'd expect it to converge with the expected value. (but the average of a sample is not the same as the expected value)
    - **The expected value we can compute if we know the probabilities of values, we don't need any sample or data point, but with average we need data points/sample data**
- The average and expected value converge when drawing large and representative random samples from the population.
- Geometrically speaking, the Expected Value is a "balance point" of a probability distribution. (think of balancing a spoon)
- Expected value is a theoretical and the average is an empirical concept

## Conditional Probability

- Good example in [video](https://www.udemy.com/course/statsml_x/learn/lecture/20014676) at timestamp 9:40
- Probability that is calculated given additional information
  - The probability of event A **changes** based on what you know about event B
- What is the probability that you will pass your final exam? vs. What is the probability you will pass your final exam, given that you've been studying for weeks?

### Formula

$$P(A|B) = {P(A\cap{B})\over P(B)}$$

- The probability of A given B ($A|B$), is the probability of A and B (the "intersection" of: $A \cap B$, i.e. A and B co-occurring) divided by the probability of B overall.
- if the probability of P(A|B) is identical or close to P(A), then it tells you there is little no relationship between A or B (see independence).
- Example: if P(A|B) = 1, then if we know B, we know A with certainty. If P(B|A) = 0.25 (and P(B) = 0.09), then it means A is somewhat informative about B, but not very much (though we know more about the probability of B given A then just given B alone..).

#### INDEPENDENCE

- If two events are independent of each other (A and B), then the probability of those events A and B is equal to the probability of A times the probability of B: $P(A\cap{B}) = P(A) * P(B)$
- If A and B are NOT independent, then the formula is conditional probability as stated above: $P(A|B) = {(A\cap{B})\over{P(B)}}$
- NOTE: if A and B never co-occur, then the probability of A given B is zero! Their intersection is 0.
  - This is different than if A and B are INDEPENDENT! In this case the probability is just The probability of A ($P(B)$ cancels out in the numerator/denominator), so the probability of A given B should just be the P of A (B is not related)

### Code: Conditional Probability

```python
# long-spike time series (think of binaries as 0 is days that are sunny, 1 is days that are completely rainy)
N = 10000
spikeDur = 10 # duration of the plateau (x-axis) shown on the plot
spikeNumA = .01 # proportion of data that is in time series A
spikeNumB = .05 # proportion of data that is in time series B (so we'll have more spikes in time series B vs. time series A)

spike_tsA = np.zeros(N)
spike_tsB = np.zeros(N)

# populate time series A
spiketimesA = np.random.randint(0,N,int(N*spikeNumA)) # gen rand integers that are the spike centers

for spikei in range(len(spiketimesA)):
  # find boundaries - this accounts for where we have spikes near the boundaries and prevents us from getting an indexing error if we get a spike center that is really close to the spike beginning or end point in the time series
  bnd_pre = int(max(0,spiketimesA[spikei]-spikeDur/2))
  bnd_pst = int(min(N,spiketimeA[spikei]+spikeDur/2))

  # zeroes get turned into 1s:
  spike_tsA[bnd_pre:bnd_pst] = 1


# populate time series B
spiketimesA = np.random.randint(0,N,int(N*spikeNumA)) # gen rand integers that are the spike centers
# you can optionally simulate a strong conditional probability by forcing spike times to coincide
spiketimesB[:len(spiketimesA)] = spiketimesA # the first N spikes in B are exactly same spikes as in A

for spikei in range(len(spiketimesB)):
  # find boundaries - this accounts for where we have spikes near the boundaries and prevents us from getting an indexing error if we get a spike center that is really close to the spike beginning or end point in the time series
  bnd_pre = int(max(0,spiketimesB[spikei]-spikeDur/2))
  bnd_pst = int(min(N,spiketimeB[spikei]+spikeDur/2))

  # zeroes get turned into 1s:
  spike_tsB[bnd_pre:bnd_pst] = 1

#plot time series:
plt.plot(range(N),spike_tsA,range(N),spike_tsB)
plt.ylim([0,1.2]) # set y axis max range
plt.xlim([2000,2500])
plt.show()

## Compute probabilities and intersection
probA = sum(spike_tsA==1) / N # sum of total points that have a val of 1 divided by total points
probB = np.mean(spike_tsB) # because the data is now 0s and 1s probability can be simplified to just taking the mean (same effect as probA code)

# joint probability
probAB = np.mean(spike_tsA+spike_tsB==2) # test for when combined time series is 2 (spike in both channels)

### Note: everything up to this point has been mostly massaging the data to organize it for calculation - the methods here will not apply to other datasets and you need to organize your data as needed

# compute conditional probabilities
# p(A|B)
pAgivenB = probAB / probB

# p(B|A)
pBgivenA = probAB / probA # scale by the probability of A in this case (B|A)

# report
print('P(A) = %g'%probA)
print('P(A|B) = %g'%pAgivenB)
print('P(B) = %g'%probB)
print('P(B|A) = %g'%pBgivenA)

# If Probability of B or A and their given conditional probability is the same, it means they have zero influence on each other.
# If P(B|A) is 1,for example, then it means that if we know about A, we know about B (whenever A occurs, B will also occur)

```

### Visualizing Conditional Probabilities with Trees

- See [video](https://www.udemy.com/course/statsml_x/learn/lecture/20014684#content) for visual ex.
- Nodes are events/possible outcomes and the two braches from each node must sum to 1.
- probabilites are given along the edges from point X to point Y and represent the probability of Y given X.
- The probability of a specific node or point D for example (i.e. not given anything else), would be the probability of point A _ P(D) + P(B) _ P(D) (you need to multiply and sum the probabilities/branches)
  - see timestamp 5:30 in video

