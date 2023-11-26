# Python Tips

### Data

- See data types in workspace (variable names and their types etc) in Jupyter notebook: `%whos`
- Clear all variables in the workspace: `%reset -sf`
  - Can also use `Kernel` dropdown in Jupyter and clear and reset space to clear vars from memory

### Pandas
-  it's a good idea to specify `low_memory=False` when reading data into dataframes unless Pandas actually runs out of memory and returns an error. The low_memory parameter, which is True by default, tells Pandas to only look at a few rows of data at a time to figure out what type of data is in each column. This means that Pandas can actually end up using different data type for different rows, which generally leads to data processing errors or model training problems later.
```python
df = pd.read_csv(path/'TrainAndValid.csv', low_memory=False)
```
