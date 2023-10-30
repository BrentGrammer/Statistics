# Visualizing Data

### Bar Plots

- Indicate bars of amounts on an x(category)-y(amount) graph
- Used for Categorical Data (nominal and ordinal)
- Numerical Data can be shown, but only DISCRETE numerical data
  - Ex: penguin height has to be in centimeters. no two penguins can have exactly the same height (because height is ratio data and not discrete, the more precise the ruler, the more precise the measurement).
    - a solution to show this data on a bar plot would be to group the data into bins (i.e. 60-70cm in height, 70-80 cm in height, 80-90 cm in height, etc.)
  - Numerical data MUST BE CONVERTED to discrete data to use in a bar plot
- Histograms (discrete data in ordered bins) and Bar plots are not exactly the same, but are used interchangeably.
- The way that data is grouped on a bar plot will have an impact on how the results are interpreted by the audience: [video at timestampe 7:28](https://www.udemy.com/course/statsml_x/learn/lecture/20009304#content)

### Error Bars

- little vertical lines superimposed on the bars in a bar plot.
- Error bars can show standard deviation, standard error, or confidence intervals
- can be ambiguous becasuse often the error bars are not labeled to indicate what they show

  - Careful about seeing bar plots with error bars that are not labeled - can be deceptive or misleading.

### Coding Bar plots in Python

- use the function `bar` which is in the matplotlib.pyplot library
- use `errorbar` to show the error bars

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# define dimensions of data (x,y) 30 rows by 6 cols
rows = 30 # rows, 30 individuals, how many sublists in the np list
cols = 6 # cols, 6  different features of individuals, how many items in each sublist

# initialize the data matrix with numpy
data = np.zeros((rows,cols))

# generate some random data for each column n:
for i in range(cols):
  # This means all rows ':' and the i-th column
  # row = each sublist in the list, col = which index in that sublist
  # data[rows, cols]
  data[:,i] = 30*np.random.randn(rows) * (2*i/(n-1)-1)**2 + (i+1)**2 # the end is a multiplicative/scaling term (gives large variance towards ends 1st and 6th col, and smaller variance in the middle) and a sum term (adds a growing squared number - makes sure each col to the right in the data is larger than the previous left col)
  # the purpose of the scaling and adding terms are just to make the data less uniform and are not necessary.

### Plot the bars ###
# get the figure and axes objects - setting up a layout of 3 plots side by side (1 row of plots in 3 "columns") and a figsize which is width x height in inches (the plot will have the size: 8 in by 2 in)
fig, ax = plt.subplots(1,3,figsize=(8,2))
# plot the x values (0 to 5), take the average of the data along the first axis/dimension 0 of the data.
# the ax[0] just means show the bar chart in the first plot column of the layout defined above
ax[0].bar(range(n),np.mean(data,axis=0))
ax[0].set_title('Bar plot')

# plotting just error bars in the second plot (at index 1 col from layout)
ax[1].errorbar(range(n),np.mean(data,axis=0),np.std(data,axis=0,ddof=1),marker='s',linestyle='')
ax[1].set_title('Errorbar plot')

# both - most common way of visualizing bar plots
ax[2].bar(range(n),np.mean(data,axis=0))
ax[2].errorbar(range(n),np.mean(data,axis=0),np.std(data,axis=0,ddof=1),marker='.',linestyle='',color='k')
ax[2].set_title('Error+bar plot')
```

### Bar plots with pandas

- Using pandas makes for nicer looking bar plots than matplotlib
- use `df.plot(kind='bar')`

```python
# matrix - each sublist is a row with 4 cols and there are two rows
m = [ [2,5,4,3], [1,1,8,6] ]

# define layout again, 2 rows of 2 plots (cols) sized 8x8 inches
fig,ax = plt.subplots(nrows=2,ncols=2,figsize=(8,8))

# imshow converts a matrix into an image, small or negative values are colored blue and larger colored yellow
ax[0,0].imshow(m) # 0,0 is the first plot in the first column
# this is not a bar plot it's a colorized visualization of a matrix..

# name the columns in a panda dataframe
df = pd.DataFrame(m, columns=['prop 0', 'prop 1', 'prop 2', 'prop 3'])
# ax[row,col]: for the first chart in the second row of layout (ax), plot as a bar chart
df.plot(ax=ax[1,0], kind='bar')
ax[1,0].set_title('Grouping by rows')

# Transpose the matrix - orient the bars to be grouped by column
df.T.plot(ax=ax[1,1]. kind='bar')
ax[1,1].set_title('Grouping by columns')
```
- You tanspose the matrix and data if you want to present the data in a different way to highlight a relationship etc. or want it to be interpreted in a certain way.
