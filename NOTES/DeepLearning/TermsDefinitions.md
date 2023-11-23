# Terms

- **Supervised Learning:** A type of machine learning where the algorithm is trained on a labeled dataset, meaning the input data is paired with corresponding target labels. The goal is to learn a mapping from inputs to outputs.

- **Unsupervised Learning:** A type of machine learning where the algorithm is given unlabeled data, and its task is to find patterns or relationships within the data without predefined target labels.

  - Self-supervised learning: Training a model using labels that are embedded in the independent variable, rather than requiring external labels. For instance, training a model to predict the next word in a text.

- **Feature:** An input variable or attribute used by a machine learning model to make predictions. Features are the characteristics of the data that the model learns from.

- **Label/Target:** The output or target variable in supervised learning. It represents the value that the model is trying to predict based on the input features.

- **Model:** A program or mathematical function that you run your data through to get results. a model consists of two parts: the architecture and the trained parameters. Models are things that fit functions to data. Models relate the value of features (data that affects the labels/output) to labels (output).

- **Training Data:** The dataset used to train a machine learning model. It consists of input features and corresponding labels, and the model learns to make predictions by adjusting its parameters based on this data.

- **Testing Data:** A separate dataset used to evaluate the performance of a machine learning model after it has been trained. It helps assess how well the model generalizes to new, unseen data.

- **Validation Data:** A portion of the dataset that is used during training to tune hyperparameters and prevent overfitting. It is separate from the training and testing datasets.

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
