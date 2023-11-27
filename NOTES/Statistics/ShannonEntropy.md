# Entropy in Statistics

## Shannon Entropy (Entropy in Information theory)

- Surprising things (less certain things) convey more information
  - A sample of all the same values has an entropy of 0 (it is not surprising and predictable to get a value of x in all x's). A sample of completely random and unique values, the entropy is 1 (each value is the maximal surprise)
  - In information theory, "surprising" means things that are unpredictable and not just unexpected. Predictability is the key measure of "surprise". i.e. a coin toss is very surprising - it is 50% chance and unpredictable.
  - Entropy is the amount of "surprise" in a system. When the probability of an event is 0 or 1, then there is no surprise -- no uncertainty. Then the entropy is 0.
- As you approach a probability value that represents more and more uncertainty, the entropy increases and therefore the more information the uncertain event conveys
- formula: $$H = {-\sum_{i=1}^np(x_i)log_2(p(x_i))}$$
  - Compute the probability of a certain event or data value `x`: $p(x_i)$, and multiply that by the log of that probability: $log_2(p(x_i))$
    - Note we are multiplying the probabilities of the data values, NOT the data values themselves (you need to convert the data values into probabilities to compute entropy)
  - We then sum up over all the probabilities for the different events/data: $\sum_{i=1}^n$
  - Attach a minus to that to get a negative number and that is the entropy $H$
    - the logarithm of a number between 0 and 1 is going to always give you a negative number, so that is why we attach the negative sign to make the result positive. (we think about increasing entropy as a positive number)
- If using $log_2$ (most common), then the units are "bits", if using log normal ($ln$), then the units are called "nats"
  - 1 value of entropy is 1 bit of entropy (binary since using log base 2)

## Use cases

- Can be applied to Nominal, Ordinal or Discrete types of data
- You can't really apply it to interval or ratio data (specific numerical data - i.e. height - does not make sense to get the probability of a very specific height - will essentially be close to zero)
  - You need to convert this data to discrete data by binning it into a histogram
  - NOTE: Entropy depends on bin size and number
- Used in deep learning, optimization to look at coherence or coupling between different systems (mutual information)

## Interpreting Entropy

- High entropy means dataset has a lot of variability
- Low entropy means that most of the values in the dataset repeat (and are redundant), i.e. there is less information in the total dataset when the entropy is low.
- In some sense entropy is similar to variance, but entropy is nonlinear and makes no assumptions about the distribution (can be used with any distribution).
  - Variance depends on validity of the mean and is appropriate for roughly normally distributed data.
  - used more often for nonlinear systems. But the interpretation is basically the same -- higher entropy (variance) means the signal bounces around more.

## Code

- Note that to avoid divide by zero errors we need to add epsilon (smallest number closest to zero in python) when calculating probabilities (dividing by N data points). See below comments in code sample.

```python
# simulate discrete data (numeric, ordinal, categorical, different flavors of ice cream, etc.)
N = 1000
# take random numbers, square them and multiply by 8 and round them up with ceil
numbers = np.ceil(8*np.random.rand(N)**2) # gives us nums between 1 and 8
# can show them as dots on a plot
plt.plot(numbers, 'o') # o shows them as dots

# Get the probability of the numbers
# get the unique numbers
u = np.unique(numbers)
probs = np.zeros(len(u))

# get the count of the unique number and divide it by the number of data points to get the probability
for ui in range(len(u)):
  probs[ui] = sum(numbers==u[ui]) / N

# Note how in code the entropy formula isn't exactly the same as the real formula
# -sum(probs*np.log2(probs)) is the same, but we add the epsilon (+np.finfo(float).eps) - this is machine precision error, the closest python can get to the number zero.
# epsilon is zero plus some rounding error, i.e. e-16 (10 to the negative 16 for example)
# the purpose of adding epsilon is to avoid divide by zero errors (log2(0) = -infinity in python)
entropee = -sum(probs*np.log2(probs+np.finfo(float).eps))

# show a histogram of the distribution of numbers (higher bar means they appear more frequently in the data)
plt.bar(u,probs)
plt.title('Entropy = %g'%entropee)
plt.xlabel('Data Value')
plt.ylabel('Probability')
plt.show()
```

### Computing Entropy from a Continuous Variable (code)

- Important thing to remember is that entropy is very sensitive to the number of bins (in the historgram for converting data to probabilities).
  - less bins = less entropy

```python
# create brownian noise to create a continuous variable (this is like a random walk process)
# come up with random nums that can be greater than or less than zero, and get the sign, then take the cumulative sum of all those signs (+1s and -1s)
N = 1123
brownnoise = np.cumsum(np.sign(np.random.randn(N)))

fig,ax = plt.subplots(2,1,figsize=(4,6))
# plot the random stream of plus 1s and minus 1s:
# note that brownian noise is one way to model the stock market (fluctuations that go up over long term trend)
ax[0].plot(brownnoise)
ax[0].set_xlabel('Data index')
ax[0].set_ylabel('Data value')
ax[0].set_title('Brownian noise')
# show a histogram to get the shape of the distribution (will be random each time)
ax[1].hist(brownnoise/sum(brownnoise),30)
ax[1].set_xlabel('Data value')
ax[1].set_ylabel('Probability')
plt.show()

# We cannot compute entropy directly on the data as is, we need to convert it to probabilities with a histogram
# NOTE: entropy is very sensitive to the number of bins used, less bins will make entropy drop
nbins = 50

nPerBin, bins = np.histogram(brownnoise,nbins)
# probabilities
probs = nPerBin / sum(nPerBin) # normalize so they all sum to 1

entropy = -sum(probs*np.log2(probs+np.finfo(float).eps))

print('Entropy = %g'%entropy) # 5.46833 for example
```
