# General Mathematics/Arithmetic Preqresuites

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
