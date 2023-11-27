# General Mathematics/Arithmetic Prerequisites

- [writing math symbols in markdown](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/writing-mathematical-expressions)
- Available markdown math syntax: https://katex.org/docs/supported.html

### Fractions/Exponents

- Shorthand for Fractions:
  - $n^{-a} = {1 \over {n^a}}$
    - With just -1: $n^{-1} = {1 \over n}$
  - negative exponents is a shorthand way of writing this fraction
  - You take the recipricol of n and make the exponent a positive.

### The Law of Exponents

- If you are multiplying more than one number that have exponents in them and the bases are the same, then you can add the exponents:
- $3^2 * 3^4 = 3^{2+4}$
  - This only works when the base (3 in this case) is the same

### Square roots

The square root is equivalent to taking a value to the power of 1/2: $\sqrt{x} = x^{1/2}$

### Isolating to solve for x

- If given an equation to solve for a particular variable, use division to isolate a particular var 'x'
- ${3x} = 42$ -> divide by 3 on both sides to isolate x

### Order Of Magnitude

- $10^3$ to $10^4$ is a difference of one order of magnitude
- ex: 10 to the first second and third powers: 10, 100, 1000

### Scientific Notation

- When dealing with exponents and numbers other than 1s and 0s (i.e. 10, 100, 1000 etc)
- `e` replaces a multiplication of an exponent:
  - $240 = 2.4 * 10^2$ ==> `240 = 2.4e2` (e is 10 and 2 is the exponent)
- if using negative exponent here, then you move the decimal places that many places to the left:
  - $2.4 * 10^{-2} = 0.024$ or `2.4e-2`

### Summation Notation

a = {1,2,3,4,5,6}

#### Short hand:

a<sub>n</sub> + a<sub>n</sub> + ... + a<sub>n</sub>

#### Equivalent Summation notation version:

$\sum_{i=1}^n a_i$

- This expression reads a little like a for loop:
  - from `i = 1` (underneath), up to `i = n`(top), sum all elements of `a`
- Note: if the sub and superscript below and above sigma is not listed then you can assume that i = 1 and the iteration goes up to the last element in the set.

### Absolute Values

- An absolute value is the distance of the number to the origin of the number line.
- For example, the origin on a real number line is 0. the distance of -2 from the origin 0 is 2.
  - This gets more complicated with imaginary numbers.
- Notation:
  - `abs(x)` - in computer code
  - `|x|` - written on paper

### Natural Exponents

- Syntax:

  - e<sup>x</sup>

- Note on the function for the natural exponent: `y = e^x`:
  - `e` stands for an irrational number (that never ends), 2.78.......
  - The function never gets to a negative result (a non negative producing function - `y` is never zero or below)
  - The function never gets to zero as x goes to negative infinity (gets extremely close asymptotic)
  - as x goes positive the result increases exponentially very quickly
  - $e$ raised to anything (including negative numbers) can never be zero (it can get asymptotically close, but never absolute zero)
  - see [lecture](https://www.udemy.com/course/statsml_x/learn/lecture/20009224)

### Natural Logarithm

- 'ln' stands for the natural logarithm, which is a specific type of logarithm that uses 'e' (the natural exponent) as its base. In mathematical notation, 'ln' is used to represent the natural logarithm function.
- has a close relationship with 'e' - the natural exponent

Logarithm: the relationship of the exponent `x` to `y` in the formula: $N^x = y$

- We can say that X is the logarithm of Y
- $x = log(y)$ or $10^{log(y)} = y$
  - log(y) is the exponent by which you would have to raise N to get Y
  - IOW, how many times you would have to multiply a number N by itself to get Y
  - The logarithm is the inverse of the exponential function, it "undoes" exponentiation:
    - $log(10^y) = y$ => this extracts the exponent of a number N that has been raised y times.
      - it simplifies to y because when you take the logarithm (base 10 in this case) of a number raised to an exponent (in this case, 10^y), it results in just the exponent itself.
    - log(5) => how many times you have to raise 10 to get the number 5
    - $log(A^n) = n * log(A)$ => ??

#### Syntax:

- natural log: ln(x) or `log(x)` in code
- natural log alt: log<sub>e</sub>(x)
- with base: log<sub>10</sub>(x)
- In statistics, this function is often used to optimize describing probabilities between very small numbers (that might be close to each other for example).

- Note on the function for natural logarithm: `y = log(x)`
  - This is the inverse of the natural exponent function above.
  - quickly goes to negative infinity as `x` approaches 0.
  - `x` cannot be negative - you cannot take the logarithm of a negative number
  - as `x` increases, `y` increases slower and slower to infinity

#### Natural Exponent and Logarithm functions are inverses of each other

- You can use them to cancel each other out.

y<sub>1</sub> = log(e<sup>x</sup>)
<br>
y<sub>2</sub> = e<sup>log(x)</sup>

- this cancels out to just `x`
- x from .0001 to 10 would plot a straight line (both coords would be `x` value)

``

### The logistic equation and Sigmoid

#### Ths Sigmoid function:

$f(x) = {1 \over 1 + e^{-\beta x}}$

- This equation is called a sigmoid because it produces a S shape on a graph when iterated.
- Sigmoid aka is called an **'S Curve'**

- The `BETA` parameter in this equation can be referred to as the HEAT/TEMPERATE parameter.
  - Defines the steepness of the S curve (how it climbs from the lower bound of 0 up to the upper bound - in this case 1)
- Note that if the numerator 1 is changed to 4 then that increases the Y axis upper bound to 4, and if we change the BETA to 2, then that means the height of the S Curve will be up to +2 etc.

#### The Logistic Function:

$ln{p \over {1-p}} = \beta$

- the natural log (a.k.a. `ln`) of p divided by 1 - p equals Beta
- `p` is a probability value - between 0 and 1 (inclusive).

- An important function in odds ratios, important for a softmax function in machine learning
- [See this video](https://www.udemy.com/course/statsml_x/learn/lecture/20009230#questions) for solving for `p`
  - use the natural exponent to cancel out the natural log and isolate:
    ${p \over {1-p}} = e^\beta$
  - Then multiply by 1-p on each side to get: $p = e^\beta(1-p)$
  - expand this out by multiplying $e^\beta$ by 1 and p: $p = e^\beta - e^\beta p$
  - move $e^\beta$ over to the left side in order to get the p's on one side: $e^\beta = p + e^\beta p$
  - pull the p out of the right side: $e^\beta = (1 + e^\beta)p$ (p is multiplied by 1 and $e^\beta$ to get equivalent of previous expression)
  - divide both sides by 1 + e^beta to isolate p: $p = {e^\beta \over 1 + e^\beta}$
  - this can be simplified further to: $p = {1 \over 1 + e^{-\beta}}$ - equivalent to the Sigmoid function

### Ranks and Tied-Ranks transforms

- Come up often and are important in statistics

#### RANK TRANSFORM

- looking at the numbers in a set as if they were sorted according the relative position on the number line (negative numbers on left, positive large on the right etc. - ASC order)
- The lowest number gets ranked lowest (1) and then numbers are ranked in order incrementing as they ascend.
  Example of a Rank Transform:
- Given a set: { 4.01, 1, -10, 4, 39234382838483 }
- We get a Rank transform of: { 4, 2, 1, 3, 5 }

#### Properties of rank transform

- rank transforms are **non linear** - numbers extremely close or far apart from each other are not relatively close or far in the rank transformed version
- Rank transforms are **lossy** - we lose information (ex. rank of number `5` represents a huge number, info that is lost in the transformation)
- **Non-invertible** transformation - there is no way to reverse engineer what the original data in the set was from a rank transformation.

#### TIED RANK TRANSFORMATIONS

- How to deal with sets with repeated numbers
- For the set: { 3, 1, -10, 4, 3 }
- We get: { **3.5**, 2, 1, 5, **3.5** }
  - We take the average of the ranks (in this case 3rd and 4th positions are tied - (3+4)/2 = 3.5)

# Calculus Basics

From the book Calculus Made Easy by Thompson, Gardner

- A fundamental notion of calculus is about growing or varying.

## Functions

- A fucntion is a relation between two terms called variables because their values vary.
- If every value of x is associated with exactly one value of
  y, then y is said to be a function of x. It is customary to use x for
  what is called the independent variable, and y for what is called the
  dependent variable because its value depends on the value of x.
- letters at the end of the alphabet are traditionally applied to variables, and letters else-where in the alphabet (usually first letters such as a,b,c. .. ) are applied to constants.
- $f(x)$ represents the dependent variable (representing y which depends on x). Ex: $f(x) = x^2$
- We call a function "continuous" if its graph can be drawn without lifting the pencil from the paper, and "discontinuous" otherwise.
- In x = y^2, x is a function of y, or alternatively, if y = 2x, then y is a function of x.
  - f(x) represents a value that changes when x is changing and is a dependent variable of x. x would be the independent variable in this scenario.
  - If multiple variables x,y,z have some sort of relation, then: $$x=f(y,z); y = f(x,z); z = f(x,y)$$

### Graphing functions of 2 variables on a Cartesian plane

- Values of the independent variable are represented by points along the horizontal x axis. Values of the dependent variable are represented by points along the vertical y axis.
- Points on the plane signify an ordered pair of x and y numbers. If a function is linear, that is, if it has one form y = ax + b-the curve representing the ordered pairs is a straight line. If the function does not have the form ax + b the curve is not a straight line (ex. $y=x^2$ is a parabola on the graph).
- Domain and Range: Values that can be taken by the independent variable are called the variable's domain. Values that can be taken by the dependent variable are called the range.
  - The domain is on the x axis and consists of numbers along that axis
  - The range consists of numbers along the y axis
- Note that if a vertical line from the x axis intersects more than one point on a curve, the curve cannot represent a function be-cause it maps an x number to more than one y number.

## Limits

### Terms

- Sequence: set of numbers like {1,2,3,4,...}
- Series: the terms of a finite sequence are added to obtain a finite sum.
- Partial Sum: If a series is infinite, the sum up to any specified term is called a "partial sum."
- Limit: If the partial sums of an infinite series get closer and closer to a number k, so that by continuing the series you can make the sum as close to k as you please, then k is called the limit of the partial sums, or the limit of the infinite series.

  - The terms (numbers in the set) either converge on k (meet at it) or diverge
  - Example: starting at 1/2 and halving the partial sums will converge on a limit of 1 (1/2 + 1/4 + 1/8 + 1/16 ...)

- Infinitesimals: As an infinite series approaches but never reaches its limit, the differences between a partial sum and the limit get closer and closer to zero. they get so close that you can assume they are zero and therefore, they can be "thrown away." In early books on calculus, terms said to become infinitely close to zero were called "infinitesimals." there is something spooky about numbers living in a neverland infinitely close to zero, yet somehow not zero. In the halving series, for example, the fractions approaching zero never become infinitesimals because they always remain a finite portion of 1. Infinitesimals are an infinitely small part of 1.

  - Controversy over the lack of useful meaning of infinitesimals in mathematics was previously common and they are preferred to be replaced by the idea of "approaching a limit", but "nonstandard analysis" (Abraham Robinson) introduced a way of making them definite and mathematically meaningful.
    - good intro to nonstandard analysis: ""Nonstandard Analysis," by Martin Davis and Reuben Hersh, in Scientific American, June 1972. "
  - Example of harmonic series: 1/2 + 1/3 + 1/4 + 1/5 + 1/6 - as the fractions converge on zero, the partial sums grow without limit.

## Derivative

- a derivative is the rate at which a function's dependent variable grows with respect to the growth rate of the independent variable. In geometrical terms, it determines the exact slope of the tangent to a function's curve at any specified point along the curve.
- the derivative of a function is another function that describes the rate at which a dependent variable changes with respect to the rate at which the independent variable changes.
- Imagine a runner that runs from an origin over a period of time. The derivative describes the rate of change of distance from the origin to the runner relative to a change in time (i.e. the rate is 10 meters per second)
- The derivative of any linear function y = ax is always just the constant a (y = 10x, the derivative is 10)
- The derivative of any constant is zero (example: a runner runs 10 meters from the starting line and stops. y = 10 means that the derivative is 0 - no rate of change)

### Terminology

- `df` - d means "a little bit of" df is a little bit of f. In mathematical terms "The element of"
- $\int dx$ - The intregal ("the sum of") of dx = the "sum of all the little bits of x", or IOW, the same thing as the whole x
- $dy\over dx$ represents the ratio of the growing of y to the growing of x. Example: $y = x^2$ ${df\over dy} = 2x$, or the growth of y with respect to x is 2 times x.

### dx/dy

- Does not mean d _ x over d _ y, it is just a symbol
- This means the derivative (differential coefficient) of y with respect to x. We are "differentiating" x with respect to y.
  - Example: $y = x^2$, the derivative is ${dy\over dx} = 2x$, or the ratio of the growing of y to the growing of x (it grows 2 times the amount x changes).
- In the process of "differentiating" We are hunting for a value of this ratio where dx and dy are infinitesimally small.
- If $y=f(x)$, then shorthand way of writing derivative is using an accent: $f'(x) = {d(f(x))\over dx}$

- In differential calculus, we are hunting for a mere ratio, namely, the proportion which dy bears to dx when both of them are infinitely small. It should be noted here that we can find this ratio dy only dx when y and x are related to each other in some way, so that when-ever x varies y does vary also.
  - Example: when you move the base of a ladder $x$ inches from a wall it is leaning on by $dx$ amount, it's height $y$ also decreases by $dy$ amount
- Note: $dx^2$ means a little bit of a little bit of x - it is not a square, but another symbol representing this in derivatives.
- Sometimes small values can be dropped (ex: if x = 100.1, then you can drop the ending 1 from the resulting y of 10,021.01 so that y is just 10,020 when calculating the derivative)

### Definition for a Derivative

$${\triangle{x} \over \triangle{y}} = \lim_{x\to 0}{f(x + \triangle{x})-f(x)\over \triangle{x}}$$

- Where $\triangle {x}$ is the delta (change or increase amount) of $x$
- $\lim_{x\to 0}$ expresses the limit when $\delta{x}$ diminishes to zero.
