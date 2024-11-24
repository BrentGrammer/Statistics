# Data Journey

- Using Marriage data from the CDC
- see [notebook](../statsML/dataJourney/stats_dataJourney.ipynb)

### Import and load data into a panda dataframe

```python
data = pd.read_excel(marriage_url,header=5)
data
```

### Remove rows that have no data or are not needed in the dataset

```python
# axis=0 means remove rows instead of cols, inplace means update the dataframe in place
data.drop([0,52,53,54,55,56,57],axis=0,inplace=True)
data
```

### Find cols with missing data

```python
# cols missing data in this dataset will have '---', so we replace --- with nan
data = data.replace({'---': np.nan})
data
```

- Now we can replace those cells with a median value for example.

```python
# replace nan with column median
# this decision for what to replace empties with is made based on assumptions and judgements about the dataset
# in this case, all the states have roughly similar rates each year, so we use median instead of the mean just in case there are outliers - we don't want the replacement to be affected heavily by outliers
data.fillna(data.median(), inplace=True)
data
```
