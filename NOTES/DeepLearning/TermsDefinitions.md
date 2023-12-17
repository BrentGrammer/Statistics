# Terms

- **Supervised Learning:** A type of machine learning where the algorithm is trained on a labeled dataset, meaning the input data is paired with corresponding target labels. The goal is to learn a mapping from inputs to outputs. Given input X, predict output Y

- **Unsupervised Learning:** A type of machine learning where the algorithm is given unlabeled data, and its task is to find patterns or relationships within the data without predefined target labels. Just given input X - analyzing the structure of the data

  - Self-supervised learning: Training a model using labels that are embedded in the independent variable, rather than requiring external labels. For instance, training a model to predict the next word in a text.

- **Linear Regression**: One way to find the line of best fit given a plot of data. It is a type of supervised learning and the output is a real number.

- **Linear Model**: The sum of all Coefficients multiplied by data values. ax_1 + ax_2 + ax_3 etc

- **Feature:** An input variable or attribute used by a machine learning model to make predictions. Features are the characteristics of the data that the model learns from.

- **Label/Target:** The output or target variable in supervised learning. It represents the value that the model is trying to predict based on the input features.

- **Target**: "target value" or simply "target" refers to the true or actual value of the variable that you are trying to predict. It is the ground truth associated with a particular input or instance in your dataset.

- **Model:** A program or mathematical function that you run your data through to get results. a model consists of two parts: the architecture and the trained parameters. Models are things that fit functions to data. Models relate the value of features (data that affects the labels/output) to labels (output).

- **Training Data:** The dataset used to train a machine learning model. It consists of input features and corresponding labels, and the model learns to make predictions by adjusting its parameters based on this data.

- **Testing Data:** A separate dataset used to evaluate the performance of a machine learning model after it has been trained. It helps assess how well the model generalizes to new, unseen data.

- **Validation Data:** A portion of the dataset that is used during training to tune hyperparameters and prevent overfitting. It is separate from the training and testing datasets.

  - Difference between test and validation data: A validation set is data we hold back from training in order to ensure that the training process does not overfit on the training data. A test set is data that is held back even more deeply, from us ourselves, in order to ensure that we don't overfit on the validation data, as we explore various model architectures and hyperparameters.

- **Overfitting:** train a model to work well only on our training data.. A phenomenon where a machine learning model performs well on the training data but poorly on new, unseen data. It indicates that the model has learned the training data too closely and may not generalize well.

- **Underfitting:** A situation where a machine learning model is too simple to capture the underlying patterns in the data, resulting in poor performance on both training and testing datasets.

- **Hyperparameter:** Parameters of a machine learning model that are set before the training process and are not learned from the data. Examples include learning rate, regularization strength, and the number of hidden layers in a neural network.

- **Feature Engineering:** The process of transforming or creating new features from the existing data to improve the performance of a machine learning model.

- **Deep Learning:** A subfield of machine learning that involves neural networks with multiple layers (deep neural networks). It is particularly effective for tasks such as image and speech recognition.

- **Reinforcement Learning:** A type of machine learning where an agent learns to make decisions by interacting with an environment. The agent receives feedback in the form of rewards or punishments based on its actions.

- **Clustering:** An unsupervised learning technique where the goal is to group similar data points together based on their characteristics or features.

- **Regression:** A type of supervised learning where the model predicts a continuous output (numeric value) based on input features.

- **Classification:** A type of supervised learning where the model assigns input data to predefined categories or classes.

- **Bias-Variance Tradeoff:** The balance between underfitting (high bias) and overfitting (high variance) in a machine learning model. Achieving an optimal tradeoff is crucial for model performance.

- **Fitting:** The process of adjusting the parameters (weights and biases) of a machine learning model during training. The goal is to optimize the model's parameters to make accurate predictions on the training dataset.

- **Fine-Tuning:** A process that occurs after the initial training of a machine learning model. It involves further training the model on a different dataset or a subset of the original dataset to adapt it to a new task or domain. Fine-tuning often leverages knowledge learned by the model on a larger dataset.

- **Weights:** refer to the parameters that are associated with the connections between neurons in a neural network. These weights are multiplied by the input values to determine the contribution of each input to the output of the neuron. During training, the values of weights are adjusted to minimize the difference between the predicted output and the actual target, allowing the neural network to learn from the data. The weights play a crucial role in the model's ability to capture patterns and relationships in the input data.

- **Inference:** When we use a model for getting predictions, instead of training, we call it inference.

- **metric:** is a number that is calculated based on the predictions of our model, and the correct labels in our dataset, in order to tell us how good our model is. we want to calculate our metric over a validation set. This is so that we don't inadvertently overfit—that is, train a model to work well only on our training data.

  - we normally use `accuracy` as the metric for classification models.

- **Rank-3 Tensor**: Tensor with 3 dimensions. The length of a tensor's shape is its rank: `len(stacked_threes.shape)`. rank is the number of axes or dimensions in a tensor; shape is the size of each axis of a tensor.

  - Watch out because the term "dimension" is sometimes used in two ways. Consider that we live in "three-dimensonal space" where a physical position can be described by a 3-vector v. But according to PyTorch, the attribute v.ndim (which sure looks like the "number of dimensions" of v) equals one, not three! Why? Because v is a vector, which is a tensor of rank one, meaning that it has only one axis (even if that axis has a length of three). In other words, sometimes dimension is used for the size of an axis ("space is three-dimensional"); other times, it is used for the rank, or the number of axes ("a matrix has two dimensions"). When confused, I find it helpful to translate all statements into terms of rank, axis, and length, which are unambiguous terms.
  - Note: PyTorch tensors are similar to NumPy Arrays
  - NumPy has a wide variety of operators and methods that can run computations on these compact structures at the same speed as optimized C, because they are written in optimized C.
  - the restriction is that a tensor cannot use just any old type—it has to use a single basic numeric type for all components. For example, a PyTorch tensor cannot be jagged. It is always a regularly shaped multidimensional rectangular structure.
  - To take advantage of PyTorch/Numpy speed while programming in Python, try to avoid as much as possible writing loops, and replace them by commands that work directly on arrays or tensors.

- **CUDA**: the equivalent of C on the GPU

- **Activation**: Output of recitified linear units/matrix multplications

- **Independent Variable**: a.k.a. a "Feature". An input that determines the variability of another parameter - i.e. rainfall amount that affects the flood damage of a property.

- **Dependent Variable**: a.k.a. a "Label". A parameter that is affected by and depends upon another input/parameter.

- **Features**: categories of data points that affect the value of a label (output)

- **Continuous variables** are numerical data, such as "age," that can be directly fed to the model, since you can add and multiply them directly.

- **Categorical variables** contain a number of discrete levels, such as "movie ID," for which addition and multiplication don't have meaning (even if they're stored as numbers).

- "cardinality" refers to the number of discrete levels representing categories, so a high-cardinality categorical variable is something like a zip code, which can take on thousands of possible levels.

- **Bagging**: Randomly choose a subset of the rows of your data (i.e., "bootstrap replicates of your learning set").
  Train a model using this subset.
  Save that model, and then return to step 1 a few times.
  This will give you a number of trained models. To make a prediction, predict using all of the models, and then take the average of each of those model's predictions.

  - we can improve the accuracy of nearly any kind of machine learning algorithm by training it multiple times, each time on a different random subset of the data, and averaging its predictions.

- **Random Forest**: a model that averages the predictions of a large number of decision trees, which are generated by randomly varying various parameters that specify what data is used to train the tree and other tree parameters. bagging, when applied to decision tree building algorithms, was particularly powerful. randomly choosing rows for each model's training, but also randomly selected from a subset of columns when choosing each split in each decision tree.Today it is, perhaps, the most widely used and practically important machine learning method.
- Remember, a random forest just averages the predictions of a number of trees. And a tree simply predicts the average value of the rows in a leaf. Therefore, a tree and a random forest can never predict values outside of the range of the training data. This is particularly problematic for data where there is a trend over time, such as inflation, and you wish to make predictions for a future time. Your predictions will be systematically too low.
  But the problem extends beyond time variables. Random forests are not able to extrapolate outside of the types of data they have seen, in a more general sense. That's why we need to make sure our validation set does not contain out-of-domain data.
  - reasoning behind why random forests work so well: each tree has errors, but those errors are not correlated with each other, so the average of those errors should tend towards zero once there are enough trees.

```python
# creating a random forest with scikitlearn

# In the following function definition n_estimators defines the number of trees we want, max_samples defines how many rows to sample for training each tree, and max_features defines how many columns to sample at each split point (where 0.5 means "take half the total number of columns"). We can also specify when to stop splitting the tree nodes, effectively limiting the depth of the tree, by including the same min_samples_leaf parameter we used in the last section. Finally, we pass n_jobs=-1 to tell sklearn to use all our CPUs to build the trees in parallel.


def rf(xs, y, n_estimators=40, max_samples=200_000,
       max_features=0.5, min_samples_leaf=5, **kwargs):
    return RandomForestRegressor(n_jobs=-1, n_estimators=n_estimators,
        max_samples=max_samples, max_features=max_features,
        min_samples_leaf=min_samples_leaf, oob_score=True).fit(xs, y)

# One of the most important properties of random forests is that they aren't very sensitive to the hyperparameter choices, such as max_features. You can set n_estimators to as high a number as you have time to train—the more trees you have, the more accurate the model will be. max_samples can often be left at its default, unless you have over 200,000 data points, in which case setting it to 200,000 will make it train faster with little impact on accuracy. max_features=0.5 and min_samples_leaf=4 both tend to work well, although sklearn's defaults work well too
```

- **Out of Bag error**: since every tree was trained with a different randomly selected subset of rows, out-of-bag error is a little like imagining that every tree therefore also has its own validation set. That validation set is simply the rows that were not selected for that tree's training.

- **Partial Dependence**: from [here](https://github.com/fastai/fastbook/blob/master/09_tabular.ipynb). Partial dependence plots try to answer the question: if a row varied on nothing other than the feature in question, how would it impact the dependent variable?

For instance, how does YearMade impact sale price, all other things being equal?

To answer this question, we can't just take the average sale price for each YearMade. The problem with that approach is that many other things vary from year to year as well, such as which products are sold, how many products have air-conditioning, inflation, and so forth. So, merely averaging over all the auctions that have the same YearMade would also capture the effect of how every other field also changed along with YearMade and how that overall change affected price.

Instead, what we do is replace every single value in the YearMade column with 1950, and then calculate the predicted sale price for every auction, and take the average over all auctions. Then we do the same for 1951, 1952, and so forth until our final year of 2011. This isolates the effect of only YearMade (even if it does so by averaging over some imagined records where we assign a YearMade value that might never actually exist alongside some other values).

A: If you are philosophically minded it is somewhat dizzying to contemplate the different kinds of hypotheticality that we are juggling to make this calculation. First, there's the fact that every prediction is hypothetical, because we are not noting empirical data. Second, there's the point that we're not merely interested in asking how sale price would change if we changed YearMade and everything else along with it. Rather, we're very specifically asking, how sale price would change in a hypothetical world where only YearMade changed. Phew! It is impressive that we can ask such questions. I recommend Judea Pearl and Dana Mackenzie's recent book on causality, The Book of Why (Basic Books), if you're interested in more deeply exploring formalisms for analyzing these subtleties.

- **Data Leakage**: introduction of data that creates a tautology (i.e. predicting that it will rain on rainy days, or IBMs flub of predicting customers for a product using already customers of that product.)

  - Example predicting grant application approvals: For the identifier columns, one partial dependence plot per column showed that when the information was missing the application was almost always rejected. It turned out that in practice, the university only filled out much of this information after a grant application was accepted. Often, for applications that were not accepted, it was just left blank. Therefore, this information was not something that was actually available at the time that the application was received, and it would not be available for a predictive model—it was data leakage.
  - it is often a good idea to build a model first and then do your data cleaning, rather than vice versa. The model can help you identify potentially problematic data issues.

- **Boosting**: another important approach to ensembling, called boosting, where we add models instead of averaging them. Here is how boosting works:
  Steps:
  Train a small model that underfits your dataset.
  Calculate the predictions in the training set for this model.
  Subtract the predictions from the targets; these are called the "residuals" and represent the error for each point in the training set.
  Go back to step 1, but instead of using the original targets, use the residuals as the targets for the training.
  Continue doing this until you reach some stopping criterion, such as a maximum number of trees, or you observe your validation set error getting worse.
  Note that, unlike with random forests, with this approach there is nothing to stop us from overfitting. Using more trees in a random forest does not lead to overfitting, because each tree is independent of the others. But in a boosted ensemble, the more trees you have, the better the training error becomes, and eventually you will see overfitting on the validation set.

- **Embedding**: Looking something up in an array