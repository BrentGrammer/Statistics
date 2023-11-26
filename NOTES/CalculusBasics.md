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

### dx/dy

- Does not mean d _ x over d _ y, it is just a symbol
- This means the derivative (differential coefficient) of y with respect to x. We are "differentiating" x with respect to y.
  - Example: $y = x^2$, the derivative is ${dy\over dx} = 2x$, or the ratio of the growing of y to the growing of x (it grows 2 times the amount x changes).
- In the process of "differentiating" We are hunting for a value of this ratio where dx and dy are infinitesimally small.
- If $y=f(x)$, then shorthand way of writing derivative is using an accent: $f'(x) = {d(f(x))\over dx}$

- In differential calculus, we are hunting for a mere ratio, namely, the proportion which dy bears to dx when both of them are infinitely small. It should be noted here that we can find this ratio dy only dx when y and x are related to each other in some way, so that when-ever x varies y does vary also.
  - Example: when you move the base of a ladder $x$ inches from a wall it is leaning on by $dx$ amount, it's height $y$ also decreases by $dy$ amount
- Note: $dx^2$ means a little bit of a little bit of x - it is not a square, but another symbol representing this in derivatives.
