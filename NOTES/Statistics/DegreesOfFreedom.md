## Degrees of Freedom

- The number of points in a dataset or an analysis that can freely vary or move around while the other points are constrained and cannot move.
  - The number of independent sample values
  - The number of sample data points that can vary
  - The number of data values that are unconstrained by the rest of the data values.
- Example: you have a set {a,b,c,d} that produce a given average 5, and you are given a,b,c: $5 = {{3 + 4 + 9 + d}\over 4}$ (a,b,c are constrained and not free)
  - In this example, figuring out d in a series of values whose average is 5, means that d is a dependent variable and does not have freedom - it depends upon the first three values.
  - If you only gave two values and asked for c and d, for instance, then those would be independent variables and not free (there are multiple possibilities and they're values do not depend on the other two terms directly.)
  - But, as it stands as you give 3 values from which you can determine the fourth (d), then it is said that there are 3 degrees of freedom (because a,b,c are allowed to be anything, but d is dependent on what a,b,c is).
  - if we have a set $x = {a,b,c,d}$ and a known average of 5, then this scenario has 3 degrees of freedom (N-1 degrees of freedom where N is the number of variables), because you have to know at least 3 values to compute the last one (if we know a,b,c then we can compute d)
- Another example: Given a population mean of 5 and variables in a sample data set of {a,b,c,d}, what is the degrees of freedom?
  - It has 4 degrees of freedom, because the contraint given (the population mean of 5) does not give us enough information (as, for ex, the sample mean) to fill in anything and calculate some remaining variable - all variables are independent and could be anything.
- Example using engineering: A seesaw has one degree of freedom - there is one point in the structure that can move or vary - up and down (the central fulcrum pivot point). A helicopter has 6 degrees of freedom: it can yaw, go up and down, go left and right, roll, go forwards and back, or pitch up or down

### General Formula

$$df = N-k$$

- where N is the number of data points and k is the number of parameters we have in the analysis
  - this generally applies, though not always. can depend on type of analysis or odoing corrections etc.

### Use of degrees of freedom

- The degrees of freedom can determine the shape of a Null Hypothesis distribution (mostly the width or other features)
  - A Null hypothesis distribution for a T value with 6 degrees of freedom has a different shape than a null hypothesis for a T value with 29 degrees of freedom.
- Higher degrees of freedom generally indicate more power to reject the null hypothesis (related to statistical power). You have more capability and power, and therefore can have more confidence, to reject the null hypothesis with higher degrees of freedom.
- Can be useful for checking and understanding someone else's results to make sure they are accurately reported (i.e. checking an ANOVA table for accuracy and understanding)
