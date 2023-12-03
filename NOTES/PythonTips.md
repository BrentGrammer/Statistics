# Python Tips

### Data

- See data types in workspace (variable names and their types etc) in Jupyter notebook: `%whos`
- Clear all variables in the workspace: `%reset -sf`
  - Can also use `Kernel` dropdown in Jupyter and clear and reset space to clear vars from memory

### Pandas

- it's a good idea to specify `low_memory=False` when reading data into dataframes unless Pandas actually runs out of memory and returns an error. The low_memory parameter, which is True by default, tells Pandas to only look at a few rows of data at a time to figure out what type of data is in each column. This means that Pandas can actually end up using different data type for different rows, which generally leads to data processing errors or model training problems later.

```python
df = pd.read_csv(path/'TrainAndValid.csv', low_memory=False)
```

- If you call a method on a dataframe it calls it on every column. You can also run a function which will run on each col and then do a reduction (sum) for each col

```python
# isna() treats boolean as a number (1 or 0), so we can get the number of missing values in a dataframe doing the following:
df.isna().sum()
# shows number of missing values for each column

# one way of dealing with empty values is to replace them with the mode (the most common value)
# this will return the mode of every column
modes = df.mode().iloc[0] # if tie there is multiple return vals so take the first one. iloc gives you the row located at the index, the zeroth row.

# replace the values
df.fillna(modes, inplace=True) # we pass inplace to modify the original df instead of returning a new one.
```

- Quick way to look at data to get a quick sense of it:

```python
# describe numeric vals
df.describe(include=np.number)

# describe non numeric vals
df.descibe(include=[object])
```

- Converting non gaussian data:

```python
# can generate a histogram on cols of interest
df['Fare'].hist()

# if we have a long tail distribution, for ex. which are not good for neural nets or linear models, we can convert them using log
# the log curve causes really big numbers to be less really big and doesn't change smaller numbers much at all
df['LogFare'] = np.log(df['Fare']+1) # log(0) = NaN, so add +1 to avoid that.

# Note: things that can grow exponentially like money or population you usually want to take the log of.
```

- Find unique values in a col: `uniq = df.Colname.unique()`

### Good code to study for overview of useful pandas features:

```python
# from kaggle Titanic dataset
df = pd.read_csv(path/'train.csv')

def add_features(df):
  df['LogFare'] = np.log1p(df['Fare'])
  df['Deck'] = df.Cabin.str[0].map(dict(A="ABC",B="ABC",C="ABC",D="DE",E="DE",F="FG",G="FG"))
  df['Family'] = df.SibSp+df.Parch
  df['Alone'] = df.Family==1
  df['TicketFreq'] = df.groupby('Ticket')['Ticket'].transform('count')
  df['Title'] = df.Name.str.split(', ',expand=True)[1].str.split('.',expand=True)[0]
  df['Title'] = df.Title.map(dict(Mr="Mr",Miss="Miss",Mrs="Mrs",Master="Master")).value_counts(dropna=False)
add_features(df)
```

## Pytorch

- Can do the same things that numpy can and also has differentiation and GPU usage.
- To use data in pytorch you need to turn it into a TENSOR. (A pytorch array/vector or matrix)
- If there is an underscore `_` at the end of a function in pytorch that means it is an inplace operation and will mutate the callee (ex: `myvar.requires_grad_()`)

```python
from torch import tensor
# turned the survived col (titanic data) into a tensor:
t_dep = tensor(df.Survived) # t_dep is the dependent variable (what we want to predict or the result of f(x))

# Changing multiple cols into a tensor:
indep_cols = ['Age','SibSp','Parch','LogFare']
t_indep = tensor(df[indep_cols].values(), dtype=torch.float) # make sure all are float types to enable multiplication in pytorch

# get the shape with:
t_indep.shape
# prints how many rows and how many cols: ex: [891,12]

# the rank of a tensor shape is the length of it:
rank = len(t_indep.shape) # returns num of dimensions or axes, ex: vector is rank-1, matrix/table is rank-2, scalar is rank-0 etc.
```

### Setting up a linear model

```python
# seed manually to make the random numbers produced the same each time
torch.manual_seed(442) # start the pseudo-random sequence with this number

n_coeff = t_indep.shape[1] # we need same number of coefficients as cols which is the second entry of the .shape of the tensor
coeffs = torch.rand(n_coeff)-0.5 # subtract 0.5 to center the data

# This multiplies a matrix by a vector:
t_indep*coeffs # takes each coefficient and multiplies them in turn for every row of data
# This is broadcasting (as if the coeffs are broadcasted to each row of t_indep data - note that the cols in both shapes are the same, *The last axes must match)

# Scale data to a fixed range to normalize effect of coefficient on data values in a row
# get the max
vals, indices = t_indep.max(dim=0) # dimension is zero to get the rows instead of the columns (we want the max values of the data in the rows to get something to normalize on, not in the columns)

# use broadcasting to take the datas and divide them by the max vals to normalize the range. This will bring all the values into a standardized range (instead of being in different magnitudes for ex.)
t_indep = t_indep / vals

# get the first set of predictions by summing the values multiplied by coefficients
def calc_preds(coeffs,indeps): return (indeps*coeffs).sum(axis=1)

# define a loss function to use with gradient descent
def calc_loss(coeffs,indeps,deps): return torch.abs(calc_preds(coeffs,indeps)-deps).mean() # mean absolute value loss function, using diff of predictions and actual values for survived.

# we want to calculate derivatives for the coefficient (prep for gradient descent). This allows us to call backward() on the loss function since coeffs are used in that function.
coeffs.requires_grad_()

# when we calculate loss now, a gradient function is stored which is the fn that python has remembered it would have to do to undo those steps to get back to the gradient by calling .backward()
loss = calc_loss(coeffs,t_indep,t_dep)

loss.backward()
# now we have the coefficients gradients
coeffs.grad # [-0.0106,0.0129,-0.0041...]
# If the gradient is negative, then you increase that coefficient to make the loss go down. If it is positive you need to decrease the coefficient (increasing the coefficient will change the loss by how much is shown)

# use gradient descent to check the loss effect applying the gradient directed change (mult by the learning rate)
with torch.no_grad(): # don't recalc grads since we use coeffs in a function here
  coeffs.sub_(coeffs.grad*0.1) # sub_ substitutes the coefficients in place and then we re-run the loss
  print(calc_loss(coeffs,t_indep,t_dep))
```

### Training the linear model

```python
# create training and validation splits of the data
from fastai.data.transforms import RandomSplitter
trn_split,val_split=RandomSplitter(seed=42)(df)
# set training and validation data to variables
trn_indep,val_indep = t_indep[trn_split],t_indep[val_split]
trn_dep,val_dep = t_dep[trn_split],t_dep[val_split]

def update_coeffs(coeffs,lr): coeffs.sub_(coeffs.grad * lr) # extract operation to function - lr = learning rate
def one_epoch(coeffs,lr):
  loss = calc_loss(coeffs,trn_indep,trn_dep)
  loss.backward()
  with torch.no_grad(): update_coeffs(coeffs,lr)
  print(f'{loss:.3f}',end=';')
# function to generate the coefficients with the requires grad setting
def init_coeffs(): return (torch.rand(n_coeff)-0.5).requires_grad_()

# function that encapsulates training cycle
def train_model(epochs=30,lr=0.01):
  torch.manual_seed(442)
  coeffs = init_coeffs()
  for i in range(epochs): one_epoch(coeffs, lr=lr)
  return coeffs

coeffs = train_model(18,lr=0.02) # this prints the loss at each epoch to see how it changes (should go down)

def show_coeffs(): return dict(zip(inep_cols,coeffs.requires_grad_(False)))
show_coeffs() # shows the ending coefficients after applying gradient descent for each column: { col: coeff }
```

### Showing Accuracy of the model

```python
preds = calc_preds(coeffs,val_indep)
# val_dep is the actual data
results = val_dep.bool()==(preds>0.5) # over a prediction of 0.5 that predicts survival success
results[:16] # shows if pred is right or wrong for first 16 rows [True,True,False,...]

results.float().mean() # true=1 false=0, take the mean of all trues/falses to get percent true/right

# bundle steps above into a function
def acc(coeffs): return (val.dep.bool()==(calc_preds(coeffs,val_indep)>0.5)).float().mean()
acc(coeffs) # 0.7921 - percent right i.e. 79%
```

### Use sigmoid function for normalization

- Whenever you have a binary dependent variable (0 or 1) put it through a sigmoid as last step to improve accuracy

```python
# https://www.youtube.com/watch?v=_rXzeWq4C6w&t=8s left off at 51:57
import sympy
sympy.plot("1/(1+exp(-x))",xlim=-5,5) # show a sigmoid for demonstration
# apply sigmoid to data
def calc_preds(coeffs,indeps): return torch.sigmoid((indeps*coeffs).sum(axis=1))

coeffs = train_model(lr=2) # uses the redefined calc_preds with the sigmoid since python is dynamic, note we're able to increase the learning rate from .1 to 2 (easier to optimize in this case)

acc(coeffs)

show_coeffs()
```

### Applying Matrix Multiplication and Creating a Neural Network

- The python symbol `@` is an official python operator that means matrix multiplier
  - We can't use the `*` since it is an element-wise operator
  - We can use this with pytorch tensors for example to do matrix multiplication

```python
# matrix multiplication (note below * element wise)
(val_indep*coeffs).sum(axis=1) # multiply coeffs to data in row (axis 1) and sum it up

# note that the * + / operations in pytorch are element-wise ops (corresponding elements). we need to use @
val_inep@coeffs # this does the same thing as the above line, but you don't need to use .sum etc.
# we can simplify the fn for predictions now with @:
def calc_preds(coeffs,indeps): return torch.sigmoid(indeps@coeffs)

# we want to move towards multiplying matrixes not a matrix by a vector as we have currently
def init_coeffs(): return (torch.rand(n_coeff, 1)*0.1).requires_grad_() # this creates a n by 1 matrix/rank 2 tensor. This makes the result of the multiplication also a matrix

# convert data to matrix
trn_dep = trn_dep[:,None] # if we index into a dimension that doesn't exist with None, it adds that dimension. [:] means everything there already, [:,None] means everything and add a new dimension
val_dep = val_dep[:,None]

trn_dep.shape # [713,1] has a trailing unit axis now. 713 rows with one column

coeffs = train_model(lr=2)

# the coeffs are now in a column vector/rank 2 matrix shape
coeffs.shape # [12,1]
acc(coeffs)


#### Create the neural network
# we need multiple sets of coeffs for a neural network
def init_coeffs(n_hidden=20):
  # this is a matrix of a set of coeffs by number of hidden activations (n_hiddden), centralize them with -0.5
  layer1 = (torch.rand(n_coeff,n_hidden)-0.5)/n_hidden # /n_hidden to make the next step in gradient descent smaller - otherwise it is too big and will over or undershoot

  # now we need to multiply the 20 sets of results by coefficients. this is going to be a rank2 matrix (col vector) to create one output (predictor of survival)
  layer2 = torch.rand(n_hidden,1)-0.3 # fiddling around subtracting .3 got this to train (?)
  # layer 2 needs a constant term - single scalar and number
  const = torch.rand(1)[0]
  # attach the grad prop to get derivatives
  return layer1.requires_grad_(),layer2.requires_grad_(),const.requires_grad_()

def calc_preds(coeffs,indeps):
  l1,l2,const = coeffs # unpack the 3 returned things from above fn (passed in as coeffs here)
  res = F.relu(indeps@l1) # replace negatives with zeros using relu and matrix multiply coeffs/data
  res = res@l2 + const # second layer matrix multiplier with the constant term
  return torch.sigmoid(res) # chuck through sigmoid since we're dealing with binary dependent var

# update the coefficients based on gradients since we use layers (this is used in train_model in previous block and overwrites the existing definition)
def update_coeffs(coeffs,lr):
  for layer in coeffs: layer.sub_(layer.grad *lr)

train_model(lr=1.4) # all the learning rates and constants were just fiddled with to get "right"
```
