# Signal Detection Theory

## Two perspectives of the world

- Reality: Objective
- Perception: Subjective

### The 4 Categories of Responses:

- "Hit" (confirmation presence): When reality matches perception.
- "Correct Rejection" (confirmation of absence)
- "Miss": missing confirmation of presence when presence was there (it was there but you said it wasn't, you missed it.)
- "False alarm": incorrectly declaring presence when presence is not there.
- These are similar to Type I and Type II errors table from the null hypothesis in hypothesis testing.

## d-prime (d')

- common metric used in signal detection theory; it quantifies accurate performance.
- A measure of discrimination (separating types of errors or correct conclusions) - it separates accurate responses from just saying 'yes' all of the time.
- Comparing the hits to misses can give you an idea of accuracy and reveal if there is a bias of confirming presence all the time indiscriminately.
- **We distinguish between all positive responses (hits + false alarms) and correct confirmations (hits only)**

### Interpreting d-prime

- The higher the d-prime, the more accurate the performance.
- a d-prime of 0 represents pure guessing on the observers part.
- a d-prime of 1 is considered good, d-prime of 2 is great. Between 1 and 2 is considered good discrimination.
- The interpretation can be relative as you compare d-prime across individuals, groups or conditions in an experiment.
- It is possible for d-prime to be negative when people have more false alarms than hits.
  - This means that there is probably a mistake in the experiment or analysis or whatever is creating the data is doing something wrong. You should never see a d-prime that's negative - means something is wrong.
- d-prime does not work when hits or false alarms are exactly 0% or exactly 100%
- d-prime is insensitive to response bias:

### Algorithm for d-prime

- Convert hits and false alarms to proportion relative to total number of veridical (truthful reality/actually occured) events.
- Start with computing the probability of hits: $$p(H) = {\text{Hit} / {\text{Hit+Miss}}}$$
- The probability of hits is the number of hits divided by the number of hits plus the number of misses
  i.e. the total real hits that were present in objective reality.
  - See [video](https://www.udemy.com/course/statsml_x/learn/lecture/20257754?start=300#questions) at timestamp 4:47
  - This is just a conditional probability: $P(\text{Saying Yes}|\text{Object was present})$ (saying yes given that the object was present)
- Compute the probability of False Alarms: $$p(FA) = FA / (FA+CR)$$
  - Number of false alarms divided by the false alarms plus the correct rejection total
- Convert proportion (probabilities) to a standard z (z-score)
  - This stretches out the extremes and gives us a larger dynamic range that facilitates the interpretation
  - see [video](https://www.udemy.com/course/statsml_x/learn/lecture/20257754?start=300#questions) at timestamp 6:05
  - Useful in statistics because we converted a metric that is uniform distributed to a metric that is normally distributed - facilitates subsequent analyses that rely on gaussian distributions (regression, ANOVA, t-tests etc.)
- d-prime is a subtraction of the z-score differences: $$d' = z(H) - z(FA)$$ z-score of hits minus z-score of False alarms (see timestamp 7:30 in video)
  - Note: the z-scores correspond with the probability of Hits and False Alarms
- The main idea is that we are not just looking at the proportion of hits but we're subtracting off and removing bias introduced by pressing yes all the time and having a lot of false alarms.

### Example

- 22 hits, 3 false alarms, 8 misses, 27 rejections.
- p(H) = .73
- p(FA) = .1
- z(H) = .622
- z(FA) = -1.28
- .622 - -1.28 = 1.9
  $$d' = 1.90$$

## Code: d-prime

- see [notebook](../../statsML/sigdet/stats_sigdet_dPrime.ipynb)
- compute it using the inverse cumulative density function (ppf) with the stats module by getting the z-values using that and then subtracting the hits from false alarms.

```python
import scipy.stats as stats

hitP = 22/30 # proportion of hits
faP  =  3/30 # proportion of false alarms

# step 2
# using stats module to get inverse of the normal cumulative density function - calculates the Percent Point Function (PPF), also known as the inverse cumulative distribution function (CDF), for a normal distribution.
# returns the x axis coordinate for the corresponding y-axis probability value
hitZ = stats.norm.ppf(hitP) # ppf gives us a z-value associated with the probability - it's the inverse of the cdf.
faZ  = stats.norm.ppf(faP)

# step 3 - subtract the hits from the false alarms zvalues
dPrime = hitZ-faZ
```

## Response Bias

- The tendency to respond "Present" more often vs. "Absent" more often.
- it is orthogonal (unrelated or independent) to d-prime because very different d-primes can be associated with the same bias. (or conversely, different biases can give rise to the same d-prime value)
- Response bias is a complement to the d-prime measure

### Example of Response Bias

- A person that is highly agreeable that likes to say yes and will have more hits than false alarms with very few misses and correct rejections.
  - This person can still have a high d-prime (indicating high accuracy though a bias exists)
- A disagreeable person will say no more and have a lot of misses and correct rejections and few hits and fewer false alarms.
  - This person can also still have a high d-prime

### Computing Response Bias (similar to computing d-prime)

- Convert hits and false alarms to proportion relative to total number of veridical events.
- Convert the proportion to standard z.
- Take the negative average: $$-[z(FA)+z(H)] / 2$$
  - Instead of taking difference, we take the average of the z-values for hits and false alarms.
  - A convention in signal detection literature is to take the negative of the average (this seems arbitrary and is just convention..).
- Note on not taking into consideration the number of times people said "no"
  - If you have a hit proportion lower than .5 it means people have more misses than hits
    - if more hits than misses, then hits proportion is larger than .5
  - If the false alarm proportion is larger than .5 it means they had more correct rejections than false alarms.
  - see [video](https://www.udemy.com/course/statsml_x/learn/lecture/20257758#content) at timestamp 4:14

### Why do we take the average instead of a difference (as in d-prime calculation)?

- see [video](https://www.udemy.com/course/statsml_x/learn/lecture/20257758#content) around timestamp 5:20
- Note that when the proportion is .5 that corresponds to a z-score of 0
  - When people say 'yes' more than 'no' you get a positive z-value average (gets flipped to negative by convention)
  - When people say 'no' more then the Hit or false alarms will be lower so the average is going to equate to a negative corresponding z-value (i.e. lower than .5 proportion) - sign is flipped by convention
- So instead of taking diff of hits and false alarm proportions we take the average and want to see if it will be greater than .5 or less than .5 (this converts to a z-score that is postive or negative)

### Interpreting the Response Bias

- The response bias can be positive or negative values (unlike d-prime)
- 0 means no response bias exists - people say "yes" as much as they say "no"
  - They have roughly same proportion of hits and false alarms, which means they have the same proportion of hits and correct rejections because false alarms mirror correct rejections.
  - IOW, when hits = correct rejections, the hits and false alarms proportions average to 0.5 (giving z-score of 0)
- A positive response bias value means there is a bias toward saying "no"
  - more misses and correct rejections and fewer hits and false alarms
- A negative response bias value means there is a bias to saying "yes"
  - more hits and false alarms, fewer misses and correct rejections

## Code: Response Bias

- See [notebook](../../statsML/sigdet/stats_sigdet_responseBias.ipynb)
- similar to d-prime calculation

```python
hitP = 22/30 # proportion of hits
faP  =  3/30 # proportion for false alarms

# step 2 - get z-values using inverse cumulative density functions (ppf)
hitZ = stats.norm.ppf(hitP)
faZ  = stats.norm.ppf(faP)

# step 3
respBias = -(hitZ+faZ)/2 # average the z-values
```

## F-score

- Another way of describing binary classification results.

### Precision and recall

$$\text{Precision} = {\text{Hits} \over {\text{Hits + FA}}}$$

- Number of hits divided by the number of hits plus the number of False Alarms
- It's a normalized measure of the number of hits. Normalized for the times you said yes, but it was a guess which you got wrong. Normalized for the margin across your perception.
  $$\text{Recall} = {\text{Hits} \over {\text{Hits + Miss}}}$$
- Normalizing hits based on misses instead of false alarms -- according to the margin of reality instead of the margin of perception as in Precision.

### Formula for the F (F1) Score:

$$F_1 \text{ Score} = {2 * \text{ Precision} * \text{ Recall} \over {\text{Precision} + \text{Recall}}}$$

- Product of the precision and recall (factored by 2) divided by the sum of precision and recall

#### Can be simplified to:

$$F_1 \text{ Score} = {\text{Hits} \over {\text{Hits} + (\text{FA + Miss})/2}}$$

- Numerator: Hits = correctly said yes (correct rejections are not included in the F score)
- Denominator: The sum of correct options with the incorrect/errors - FA (False Alarms) and Misses are the two types of errors you can make in the binary classification - we average these 2 together (with the /2)

### Interpreting F Score

- The highest F score is 1.0, the lowest is 0
- A high F score of 1 or closer to it means a higher number of correct hits.
- A F score of 0 means that you had no hits correctly
- A middle score (0.5) would indicate random guessing.
  - You'd have roughly equal number of hits, false alarms, misses etc.
- You want the numbers of hits to be high, so we take into the account the upper bound of hits
- If Hits in the numerator increase it doesn't bring up the F score that much because we have the Hits in the denominator as well (they equal 1 when divided).
  - Because of this, if False Alarms and Misses are 0, then the F score becomes 1 (the highest possible)
  - Alternatively, if hits are 0, then always 0 divided by something will be 0

### Weighting the F score

- You can add weights to Precision or Recall (multipliers) to get a $F_4$ score for example.
- $F_1$ score is the basic score where the weighting is 1 for precision and recall equally.

## Code: F-Score

- See [notebook](../../statsML/sigdet/stats_sigdet_Fscore.ipynb)

```python
N=50

H = np.random.randint(1,N)  # hits - random integer between 1 and 49
M = N-H                     # misses (total N 50 minus the rand int b/w 1 and 49)

# these two add up to 50:
CR = np.random.randint(1,N) # correct rejections - rand int between 1 and 49
FA = N-CR                   # false alarms - 50 minus the rand int

# Fscore
Fscores[expi] = H / (H+(FA+M)/2) # simple impl of the formula
```

## ROC (Receiver Operating Characteristic)

- Shows how d-prime and response bias values can have different combinations of hits/false alarm proportions
  - i.e. the same d-prime val of 1.5 can correspond to 60% hits/10% false alarms and 99% hits and 70% false alarms.
- R-O-C curves
- During WWII, people tried to look at radars to determine whether enemy objects or vehicles were spotted - these people were called "Operators" that were receiving data looked at characteristics to determine what the object on the radar was (i.e. was it a tank, etc.)
- Isosensitivity Curves (ROC curves) - suggested by R. Duncan Luce as a more precise term for ROC curves.
  - R. Duncan Luce is one of the founders of signal detection theory.
  - ROC curve term is used more than isosensitivity curves.

### ROC curves are drawn in a "yes" space

- Hits and False Alarms are the "yes" values and can be graphed in 2-d
  - the probability of hits is on the y axis and the probability of false alarms is on the x axis.
- Motivation, Part 1: Identical d-prime values can be obtained from a variety of specific and different probabilities of Hits - p(H), and probabilities of False Alarms - p(FA).
- Motivation, Part 2: Hits and False Alarms are separate independent events - we can manipulate them independent of each other.

### Mapping d-prime values to p(H) and p(FA) with a ROC curve in the "yes" space

- Because of the above, we can generate a two dimensional space for the "yes" observations (hits and false alarms)
  - i.e. on a graph, the probability of hits is on the y axis and the probability of false alarms is on the x axis.
  - [in this space you can draw a ROC curve](https://www.udemy.com/course/statsml_x/learn/lecture/20257762?start=255#content) (see timestamp 5:04)
    - Example: you can get the same d-prime for someone who had 25% Hits and only 5% False Alarms
      - versus someone with 90% Probability of hits and 90% probability of false alarms - they also have the same d-prime
    - Conclusion: lots of different combinations of p(Hits) and p(False Alarms) can produce the same d-prime.
  - The curve shows how the combinations of hits/false alarm probabilites map to a specific d-prime value
- A ROC curve for a higher d-prime value (i.e. d' = 2 vs. d' = 1) shows high discriminability - really high hits has very few false alarms, etc.. See [video](https://www.udemy.com/course/statsml_x/learn/lecture/20257762?start=255#content) at timestamp 5:58
- A d-prime of 0 would show a straight diagonal line from 0 origin to the top right of the graph. (the same number of hits as false alarms - just saying "yes" all the time regardless if it's correct)

### Mapping Response bias values to p(H) and p(FA) with a ROC curve

- In the same way you can draw a ROC curve that shows the response bias values to each set of hit and false alarm proportions on a 2-d graph as above.
- see [video](https://www.udemy.com/course/statsml_x/learn/lecture/20257762?start=255#content) at timestamp 6:34
- These have a more othoganal relationship to the d-prime ROC curves.
- Note that you can have positive and negative response bias values, so the curves can be drawn anywhere in the space.

### Summary and usage of ROC curves

The ROC curve illustrates that the same level of discriminability (d') can be achieved with different balances between sensitivity and specificity. This reflects the trade-off between detecting true positives and avoiding false positives, based on the threshold chosen for decision-making.

So, a high probability of hits and a high probability of false alarms indicates a low threshold for declaring a positive. This can be useful when missing a true positive is costly, even if it means accepting more false positives. Conversely, a more conservative approach, with a lower threshold for declaring positives, leads to fewer false alarms but also misses more true positives. This might be preferred when false positives are more costly.

In practice, people often compare multiple ROC curves from different companies, test environments, medication outcomes, instructions for detecting fraud or dangerous items in a scan, etc. You can compute the area under a single ROC curve to evaluate the decision-maker, but in many cases, ROC curves are compared against each other.

## Code: ROC curves

- see [notebook](../../statsML/sigdet/stats_sigdet_ROC.ipynb)
