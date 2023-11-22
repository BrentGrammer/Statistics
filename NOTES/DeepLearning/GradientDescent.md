# Gradient Descent
- "Gradient" is a term that is equivalent to "Derivative", but in the context of many dimensions (i.e., not just a one dimensional derivative).  It is a collection of multiple partial derivatives with respect to all the dimensions of that function (see below).
- "Descent" refers to following the derivative down to get to the loss function minimum (least error)
- If the derivative is negative, you move positive (to the right) on the error landscape and vice versa.
- The goal of the derivative in the gradient descent algorithm is to move towards a value of zero. The goal of the equation curve is to move towards the true local minimum of the function curve


### Derivatives

- **Derivative**: The slope of a function curve at any given point. See [illustration](https://www.udemy.com/course/deeplearning_x/learn/lecture/27842092#questions/20865556)
- **Partial Derivative**: In multi dimensional spaces/graphs you can measure the derivative of each dimension (axis) separately. These are partial derivatives (i.e. only the derivative of the slope on the x axis or just the y axis etc.).  Answers "how is the function changing over x axis/dimension?" etc.
  - You can take the two partial derivatives and put them in a list - this is the "Gradient"
- Notation: $${df\over dx} = f'(x) = df$$
- Partial Derivative: $${\partial f\over \partial x} = f_x(x) = \partial_x f$$
  - the $\partial$ symbol is called a "del"
- Gradient: $$\nabla = (\partial_x f,\partial_y f,...,\partial_z f)$$
  - The NABLA symbol ($\nabla$) is used to indicate a gradient result
  - Gradient is the collection of all the different partial derivates for the dimensions of the data.


### Local minima
- low points in the error landscape that are "dips" and not the real global minimum loss in gradient descent
- Gradient descent can get "stuck" in these dips that occur (it will move to the right as error loss decreases and move back the left when error loss increases)
  - see [video](https://www.udemy.com/course/deeplearning_x/learn/lecture/27842082) at 4:00
- Gradient descent is gauranteed to go downhill (down the error landscape)
  - This is not a gaurantee that a good solution will be found if parameters are not set right for the particular error landscape

Note: it is impossible to visualize this on a graph beyond 2 dimensions.

- Somehow deep learning overcomes this pitfall, is successful and it is a mystery and not understood how.
  - it is possible that there are many different good solutions (different weights and params result in same high accuracy)
  - it's also possible that there are very few local minima in high-dimensional space (impossible to visualize and confirm). i.e. the local minimum would need to be a local min in all dimensions (unlikely), otherwise gradient descent will go a different direction to continue reducing loss
- Ways of dealing with this 
    - (you can never be sure and have to rely on model accuracy) - use random weights and re-train the model many times.
    - increase the dimensionality and complexity of the model so that there will be fewer local minima