# Scaling

## Linear vs. Logarithmic axes

The scaling of the ticks on the y-axis of a graph

### Linear scaling

- The ticks on the y axis are scaled by addition - you add a consistent amount between each tick.
- i.e. 10 20 30 40 etc. (+10)
  <br>

- Easier to interpret for general audiences
- Scales well to big, small or negative numbers
- **Should prefer using linear unless you have a good reason to use log plots**

### Logarithmic scaling (aka Log scaling)

- The ticks are spaced by multiplying by a number
- Ex: 1 10 100 1000 10000 (x10)
- The ticks grow faster and the spacing between the ticks is not the same between each one.
- common multiplication factors are 10, 2, 8
  - most common is x10
- common to use scientific notation on the y axis: $10^0 > 10^1 > 10^2$
- You can also scale the x-axis logarithmically (you call this a **log log plot** since the y and x axis is log scaled)

  - if only one axis is scaled logarithmically then it is just a log scaled plot.
    <br>

- Difficult to use with negative numbers (need translation)
- More appropriate for physics/biology, finance, growth or really big differences between numbers/points in the data.
- **Prefer using linear and only use log plots when necessary**

### Visualizing log vs. linear scaled

- The same data can look very different on a linear scale vs. a logarithmic scale
- See [video](https://www.udemy.com/course/statsml_x/learn/lecture/20009354#content) at timestamp 4:50
