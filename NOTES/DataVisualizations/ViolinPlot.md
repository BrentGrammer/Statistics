# Violin Plots

- Mirror image of distributions
- Useful for comparing different distributions on one plot or presenting visualizations nicer
- Rotate the distribution plot and mirror it on the other side.

```python
# just use matplotlibs violinplot and pass the data in.
plt.violinplot(data)
plt.show()

# do a swarm plot with seaborn
import seaborn as sns
sns.swarmplot(data,orient='v')

```
