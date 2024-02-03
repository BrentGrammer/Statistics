# Dealing with Missing Data

### Row wise removal

- Remove the entire row of the data with missing values
- If looking at paired data (i.e. individuals measured at time point 1 and time point 2), it is pointless to keep the row without having all the points.
- Sufficient approach if you have a lot of or enough data that removing some rows won't matter

### Selective Removal

- Can exclude the missing data value from analysis if appropriate and keep the remaining data points in a row, for example.

### Replace the data

- Usually you can compute the mean of the column and replace the missing value with the mean value.
  - Careful to get the mean of the feature (col) and not the row of the data.
- Useful in small datasets where you need to keep as much data as possible.
- Not ideal as the mean can add some noise potentially

### Predict the missing value

- This is better than replacing with the mean if you have sufficient amount of data and enough features
- You can use a regression model or some predictive model and use training/testing data to train the model (minus the missing point) and treat that as test data to generate a prediction for the missing value.
