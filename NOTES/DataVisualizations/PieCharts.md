# Pie Charts
- Types of data in pie charts: Nominal, Ordinal, Discrete
- In order for a pie chart to be valid, the total across the pie pieces must sum up to 1 or 100%

## Code

```python
import matplotlib.pyplot as plt
import numpy as np

# create data, 5 bins, 100 data points total
nbins = 5
totalN = 100

rawdata = # some fancy code to get log spaced numbers
[1,1,1,1,1,2,2,2,3,3,4] # something like this

# get the unique values from the repeated ones in the list data
uniquenums = np.unique(rawdata) # [1,2,3,4]
data4pie = np.zeros(len(uniquenums))

for i in range(len(uniquenums)):
    # This gets the total number of unique values that match the entry in rawdata (the comparison is done using numpy ndarrays not lists)
    data4pie[i] = sum(rawdata==uniquenums[i])
    # [5,3,2,1] five 1s, three 2s etc. counts of the unique values

#### Show a pie chart

# numpy can multiply each val in data4pie by 100
plt.pie(data4pie, labels=100*data4pie/sum(data4pie))
plt.show()

# create a pie chart with labels:
plt.pie(data4pie, labels=['label1','label2','label3','label4'], explode=[0,.1,0,.3,0]) # optional explode will pull out parts of the pie visually - the higher the number, the further out that part of the pie is pulled out.
plt.show()


########### DEALING WITH INTERVAL DATA ############

# You'll need to disperatize the data (make it discrete) by using the histogram() method from numpy

data = np.exp(np.random.randn(1000)/10) # log normal continuous numbers

# generate the data into bins 
histout = np.histogram(data,bins=6)

# show the pie chart
plt.pie(histout[0]) # just get the heights of the bars at index 0 (don't need the boundaries)
plt.show()
# NOTE: this is not a recommended way to show this data, but is an example if you have continuous data you need to make discrete in a pie chart.
```