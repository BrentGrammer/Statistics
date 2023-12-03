# Deep Learning Math

# Derivatives

- See [python notebook](./derivative.ipynb) with code samples for plotting derivatives

- There is a great [visualization in this video](https://www.udemy.com/course/deeplearning_x/learn/lecture/27841966#content) at 5:36 timestamp
- Note this approach could be used to analyze the slope of rock formation contours?

- The derivative tells us how a function is changing over the x axis/x variable (time, space, or error/loss landscape).
  - It is the slope of the output line of a function as it's changing over the various inputs
  - It tells us if the function is increasing or decreasing and by how much
- If the derivative is negative, then the curve of the function is going down, if it is positive, the curve is going up/increasing

### Notation

- **Derivative**: The slope of a function curve at any given point. See [illustration](https://www.udemy.com/course/deeplearning_x/learn/lecture/27842092#questions/20865556)
- **Partial Derivative**: In multi dimensional spaces/graphs you can measure the derivative of each dimension (axis) separately. These are partial derivatives (i.e. only the derivative of the slope on the x axis or just the y axis etc.). Answers "how is the function changing over x axis/dimension?" etc.
  - You can take the two partial derivatives and put them in a list - this is the "Gradient"
- Notation: $${df\over dx} = f'(x) = df$$
- Partial Derivative: $${\partial f\over \partial x} = f_x(x) = \partial_x f$$
  - the $\partial$ symbol is called a "del"
- Gradient: $$\nabla = (\partial_x f,\partial_y f,...,\partial_z f)$$
  - The NABLA symbol ($\nabla$) is used to indicate a gradient result
  - Gradient is the collection of all the different partial derivates for the dimensions of the data.

### Examples of derivatives

- See [video](https://www.udemy.com/course/deeplearning_x/learn/lecture/27841964#content) at around 3:00

#### For Activation Functions:

- ReLU (Rectified Linear Unit): $f(x) = ReLU(x)$
  - $df(ReLU)$ is the derivative of ReLU function and tells us that if x is below 0, the slope is not changing and is 0 (ReLU turns all negative numbers to 0 so the x and y stays at zero).
  - After x rises above zero the output y of ReLU increases steadily at a consistent rate as x increases. The derivative will show that the slope is constantly 1 now.
- sigmoid: $f(x) = sigmoid(x)$

### Slope

- The derivative indicates the rate of increase or decrease, or "slope" of the function curve.
  - Increasing Function:
    - If f(x) is an increasing function, it means that as you move from left to right along the x-axis, the values of f(x) are getting larger. The slope of the function (derivative) is positive.
  - Decreasing Function:
    - If f(x) is a decreasing function, it means that as you move from left to right along the x-axis, the values of f(x) are getting smaller. The slope of the function (derivative) is negative.

### Computing a Derivative (of a polynomial)

- Polynomial: A function that has an unknown, a coefficient and some power/exponent.
- Formula for getting derivative of a polynomial: $${{d\over{dx}}(ax^n)} = nax^{n-1}$$
  - where $a$ is the coefficient, $n$ is the power, and $x$ is the unknown/variable
  - example: ${{d\over{dx}}x^3} = 3x^2$
  - example: ${{d\over{dx}}(3x^3)} = 9x^2$ (3 is the coefficient $a$, the power $n$ is 3)
  - Example: ${{d\over{dx}}1*x^2} = 2x^1$
    - The coefficient is 1, the variable/unknown is x and the power is 2.
  - Note that the $n$ and $a$ are multiplied by x raised to the power-1: in Python the derivative of polynomial manually calculated is:
  ```python
  def derivative(power, coefficent, x):
    exp = power - 1
    return power * coefficent * (x**exp)
  ```
- Alternative notation: $f(x)' = 2x$
  - (where $f(x) = x^2$)

### Local/Maximum Minima

- Derivatives are used in ML/DL to determine where the minima and maxima are on a function curve (low and high points in the curve)
  - Where the function output is minimum or maximum, the derivative has a value of 0 (the slope is flat). See [video](https://www.udemy.com/course/deeplearning_x/learn/lecture/27841966#content) at 1:55 for visual.
  - The points in the function curve where the derivative is zero is where the curve is turning around (it's going from negative to positive or from positive to negative). i.e. as inputs increase/decrease, these are points where the ouput "stops" mid-air to start going the other direction.
  - To find the minima, we get the derivative of the function (which is also another function), set the result to 0 and then solve for the unknown variable (i.e. $x$)
    - This gives us **Critical Points**, the minimum and maximum. It tells us at which points the derivative is 0 (not which are sepcifically the minimum or maximum points)
- The goal of minimizing error is to find the MINIMUM critical point of the function curve.

#### Determining Minima or Maxima critical points:

- To find which of the critical points are minimum/maximum we take the derivative values before and after the point where it is zero, and the rule is:
  - If the derivative to the left negative (decreasing) and the derivative value to the right of the point is positive (increasing), then the point is the minimum point.
  - If the derivative to the left is positive (increasing) and the derivative to the right is negative (decreasing), then the point is the maximum.
- Note: Vanishing Gradient: Where the derivative is zero, but we're not at a minimum or a maximum point in the curve. IOW, an area where the function has a flat curve for extended input range (this is a problem for deep learning, but there are solutions to get around it)

### Product and Chain rules

- Computing derivatives for more complicated interacting functions (like multiplication and embedding) in unintuitive and much more complicated than for simpler functions like polynomials.
- Pytorch and libraries have built in functions that should be leveraged to estimate complicated derivatives with high accuracy

#### Product rule of derivatives:

- The addition rule: Taking the derivative of two functions added together is equivalent to taking the derivative of each function separately and adding the derivatives together: $$(f_1 + f_2)' = f_1' + f_2'$$
  - The product rule is not as intuitive - multiplying the functions derivative is not the same as multiplying the derivatives separately as in the addition rule.
- The product rule: The derivative of the product of two functions is equal to the derivative of the first multiplied by the second function, plus the first function multiplied by the derivative of the second function: $$(f * g)' = f' * g + f * g'$$

#### The Chain Rule:

- applies when embedding one function insdide of another function
  - Example, one function $g(x)$ is embedded inside another function $f(x)$: $f(g(x))$
- Example: $${df\over{dx}}f(g(x)) = f'(g(x))g'(x)$$
  - the solution is the derivative of $f(g(x))$ multiplied by the derivative of $g(x)$
  - ${df\over{dx}}(x^2 + 4x^3)^5 = 5(x^2 + 4x^3)^4(2x + 12x^2)$
    - where $g(x)$ is $(x^2 + 4x^3)$, and $f(x)$ is $x^5$
    - the derivative of $f(g(x))$ is $5(x^2 + 4x^3)^4$ following $nax^{n-1}$
    - the derivative of $g(x)$ is $(2x + 12x^2)$: $x^2 => 2x$ (the exp becomes 1 and is dropped, since 2-1 = 1); $4x^3 => 12x$; $3 - 1 => 2$ for the exponent which gives us the $12x^2$
- The chain rule is used in Gradient Descent. Libraries handle these complicated calculations for you.

### Code for Product and Chain rule:

```python
# This makes Jupyter notebook display look nicer and print functions with nice latek printing etc.
from IPython.display import display

# initialize a symbolic variable to work with
x = sym.symbols('x')

# create two different functions that will interact
fx = 2*x**2
gx = 4*x**3 - 3*x**4

# compute their derivatives separately
df = sym.diff(fx)
dg = sym.diff(gx)

# Manually apply the product rule (f' * g + f * g'):
manual = df*gx + fx*dg
# Using sympy:
viasympy = sym.diff(fx*gx)

print('Functions:')
display(fx)
display(gx)
print('Derivatives:')
display(df)
display(dg)
print('Manually calculated product:')
display(manual)
print('Calculated using sympy:')
display(viasympy)

### Chain Rule (dealing with embedded functions and calculating the derivative)
gx = x**2 + 4*x**3
fx = (gx)**5 # f(g(x))

print('Chain Rule function: ')
display(fx)

print('Derivative:')
display(sym.diff(fx))
```
