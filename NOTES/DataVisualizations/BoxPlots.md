# Box (and Whisker) Plots
- https://www.udemy.com/course/statsml_x/learn/lecture/20009314#content
- They can be useful for showing the distribution of data and outliers quickly
  - quickly see the quartiles as they are divided on the box plot
- https://udemy.com/course/statsml_x/learn/lecture/20009310#content
- can use seaborn for visualizing - package with nice visualization built into it

```python
import pandas as pd
import seaborn as sns

# create matrix/data...

# using matplotlib
plt.boxplot(data)
plt.show()

# using seaborn (looks a little )
sns.boxplot(data=data,orient='v') # vertical orientation 'v'
plt.show()

# using pandas with seaborn:
# get data into a dataframe
df = pd.DataFrame(data, columns=['zero','one','two','three','four','five'])
sns.boxplot(data=df,orient='h') # here instead we orient horizontally, this just rotates the shown graph 90 degrees
plt.show()
```
