# Entropy in Statistics

## Shannon Entropy (Entropy in Information theory)

- Surprising things (less certain things) convey more information
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
