# Deep Learning

## When is it useful?

### Useful:

- If dealing with something that a human can do reasonably quickly (even if the human needs to be an expert), then deep learning will work well for those tasks or goals.
  - Image generation, colorizing imgs, increasing resolution, removing noise from images, converting to different styles of other famous well known artists
  - Recommendation systems, web search, product recs, home page layout
  - Playing games, atari video games and rts games
  - Robotics, handling objects that are challenging to locate (transparent, shiny, lacking texture) or hard to pick up
  - Financial/logistical forecasting, text to speech
- can recognize what items are in an image at least as well as people can—even specially trained people, such as radiologists. This is known as object recognition. Deep learning is also good at recognizing where objects in an image are, and can highlight their locations and name each found object. This is known as object detection. (can pick out rock formations in a landscape photo for example.).
  - every pixel can also be categorized based on what kind of object it is part of—this is called segmentation
- summarize long documents into something that can be digested more quickly, find all mentions of a concept of interest
- Tabular data: specifically useful when dealing with high-cardinality categorical columns (i.e., something that contains a large number of discrete choices, such as zip code or product ID)
- sounds can be represented as spectrograms, which can be treated as images; standard deep learning approaches for images turn out to work really well on spectrograms.

### Not Useful:

- If the task involves complex logical thinking over a period of time, then deep learning will not be as applicable or useful.
  - Predicting election outcomes for example
- Deep learning algorithms are generally not good at recognizing images that are significantly different in structure or style to those used to train the model. For instance, if there were no black-and-white images in the training data, the model may do poorly on black-and-white images
- One major challenge for object detection systems is that image labelling can be slow and expensive. There is a lot of work at the moment going into tools to try to make this labelling faster and easier
- Generating Correct answers via NLP
- Tabular data/timeseries: If you already have a system that is using random forests or gradient boosting machines (popular tabular modeling tools that you will learn about soon), then switching to or adding deep learning may not result in any dramatic improvement.
  - Deep learning does greatly increase the variety of columns that you can include—for example, columns containing natural language (book titles, reviews, etc.), and high-cardinality categorical columns (i.e., something that contains a large number of discrete choices, such as zip code or product ID)
- Because of the issue of generating correct data, we generally recommend that deep learning be used not as an entirely automated process, but as part of a process in which the model and a human user interact closely. This can potentially make humans orders of magnitude more productive than they would be with entirely manual methods

### Math:

- Linear Algebra: specifically Matrix Multiplication. Site for learning about multiplying matrices: http://matrixmultiplication.xyz/
  - This is the fundamental mathematical operation in Deep Learning.
  - GPUs are particular good at matrix multiplication. They have tensor-cores to multiply matrices.
- Calculus: Basic understanding of derivatives.

### Foundation of Deep Learning

- We can add as many relus (Rectified Linear Units) together as we want and we can have an arbitrarily squiggly function and with enough relus, we can match it as close as we want to the data.
- With this foundation you can construct a precise model, the parameters are gotten by using gradient descent.
- When you have relus added together, and gradient descent to optimize the parameters, and samples of inputs and outputs that you want, the computer does the rest. the rest is just optimization.

### Neural Network

- see [video](https://youtu.be/8SF_h3xF3cE?t=4474)
- Basic idea: take inputs and weights into a model to produce results.
  - Multiply the inputs by the weights, replace negative numbers with zeros. - this is the main function that is iterated to improve and train the model.
  - for each iteration there is a loss value to measure how incorrect the results are. The goal is to find a mechanism to improve (reduce) the loss on each iteration and do that over and over again to make the model better.

## Image Recognition

### Data sets of images

- Squishing/stretching images can result in bad data, unrealistic shapes. what we normally do in practice is to randomly select part of the image, and crop to just that part. On each epoch (which is one complete pass through all of our images in the dataset) we randomly select a different part of each image. This means that our model can learn to focus on, and recognize, different features in our images. It also reflects how images work in the real world: different photos of the same thing may be framed in slightly different ways.

### Data Augmentation

Data augmentation refers to creating random variations of our input data, such that they appear different, but do not actually change the meaning of the data. Examples of common data augmentation techniques for images are rotation, flipping, perspective warping, brightness changes and contrast changes

## Training Data

- Large amounts of data are not necessarily needed. Sometimes the quality of the data is more important.
- Fine tuning is working with a pre-trained model and adjusting it with your new data
- Fitting is training a new model with a fresh dataset
- A Confusion Matrix based on a validation data set can be used to visually see how many results a model got wrong vs. how many they predicted or identified correctly. The rows are the data classes or categories and the columns represent which category/data class the model put the data in.
- Loss: The loss is a number that is higher if the model is incorrect (especially if it's also confident of its incorrect answer), or if it's correct, but not confident of its correct answer.
- Model Driven Data Cleaning: The intuitive approach to doing data cleaning is to do it before you train a model. But as you've seen in this case, a model can actually help you find data issues more quickly and easily. So, we normally prefer to train a quick and simple model first, and then use it to help us with data cleaning.
- Baseline: A simple model which you are confident should perform reasonably well. It should be very simple to implement, and very easy to test, so that you can then test each of your improved ideas, and make sure they are always better than your baseline.
- Iteration to improve training: "Suppose we arrange for some automatic means of testing the effectiveness of any current weight assignment in terms of actual performance and provide a mechanism for altering the weight assignment so as to maximize the performance. We need not go into the details of such a procedure to see that it could be made entirely automatic and to see that a machine so programmed would "learn" from its experience." - Arthur Samuel
- Pre-trained models can have some of the parameters known and set and then if we want to fine tune it, we take the parameters that are still unknown and adjusting the existing ones a bit as needed.
  - ULMFit was the first main process for fine tuning that was introduced

### Normalizing Data

- You want the data to be on a similar scale (Example from Titanic data: if you have male/female passenger as 0 and 1 and then prices for fares between small and large numbers, you want to perform some calculation to get the numbers closer to a similar range, i.e. between 0 and 1. With age for example, you could take the maximum age from the set and divide all ages by it to get a number in between 0 and 1)

### Common Beginner Mistakes:

- Don't worry about the best model at first - just use the quickest like resnet18 or resnet34
  - the number indicates number of layers - resnet50 for example has 50 layers
- Start training the model on day one, don't wait for more data. This helps answer the question of feasibility as well.

### Main steps:

1. Initialize the weights.
   - We initialize the parameters to random values that are small and on either side of zero (some negative and some positive)
1. For each data, use these weights to predict whether it is the target.
1. Based on these predictions, calculate how good the model is (its loss).
   - This is what Samuel referred to when he spoke of testing the effectiveness of any current weight assignment in terms of actual performance. We need some function that will return a number that is small if the performance of the model is good (the standard approach is to treat a small loss as good, and a large loss as bad, although this is just a convention).
1. Calculate the gradient, which measures for each weight, how changing that weight would change the loss
1. Step (that is, change) all the weights based on that calculation.
   - A simple way to figure out whether a weight should be increased a bit, or decreased a bit, would be just to try it: increase the weight by a small amount, and see if the loss goes up or down. Once you find the correct direction, you could then change that amount by a bit more, and a bit less, until you find an amount that works well. However, this is slow! As we will see, the magic of calculus allows us to directly figure out in which direction, and by roughly how much, to change each weight, without having to try all these small changes. The way to do this is by calculating gradients. This is just a performance optimization, we would get exactly the same results by using the slower manual process as well.
1. Go back to the step 2, and repeat the process.
1. Iterate until you decide to stop the training process (for instance, because the model is good enough or you don't want to wait any longer).

## Validation/Test and Training sets

- Datasets typically reserve a portion of data in a validation or test set which the model has not seen yet in order to measure how accurate predictions are.
- The training and validation/test sets are typically stored in separate folders in datasets

## Making a web app from Jupyter Notebook

- see [book chapter](https://github.com/fastai/fastbook/blob/master/02_production.ipynb)
  can create a complete working web application using nothing but Jupyter notebooks! The two things we need to make this happen are:

IPython widgets (ipywidgets): you can use input, output and label widgets for interfaces, buttons and display output
Voilà: helps us automatically convert the complex web application we've already implicitly made (the notebook) into a simpler, easier-to-deploy web application, which functions like a normal web application rather than like a notebook.

As you now know, you need a GPU to train nearly any useful deep learning model. So, do you need a GPU to use that model in production? No! You almost certainly do not need a GPU to serve your model in production. If you're doing (say) image classification, then you'll normally be classifying just one user's image at a time, and there isn't normally enough work to do in a single image to keep a GPU busy for long enough for it to be very efficient. So, a CPU will often be more cost-effective.

### GPU servers - managed service

Because of the complexity of GPU serving, many systems have sprung up to try to automate this. However, managing and running these systems is also complex, and generally requires compiling your model into a different form that's specialized for that system. It's typically preferable to avoid dealing with this complexity until/unless your app gets popular enough that it makes clear financial sense for you to do so.

### Deploying a Notebook

in early 2020 the simplest (and free!) approach is to use Binder. To publish your web app on Binder, you follow these steps:

Add your notebook to a GitHub repository.
Paste the URL of that repo into Binder's URL, as shown in <>.
Change the File dropdown to instead select URL.
In the "URL to open" field, enter /voila/render/name.ipynb (replacing name with the name of for your notebook).
Click the clickboard button at the bottom right to copy the URL and paste it somewhere safe.
Click Launch.

### Deploying ML applications

- Recommended book Building Machine Learning Powered Applications by Emmanuel Ameisen: https://www.oreilly.com/library/view/building-machine-learning/9781492045106/

## Neural Networks

### Image Processing

- see [FastAI: chapter 4](https://github.com/fastai/fastbook/blob/master/04_mnist_basics.ipynb)

- Example from image processing: array(im3)[4:10,4:10] -- The 4:10 indicates we requested the rows from index 4 (included) to 10 (not included) and the same for the columns. NumPy indexes from top to bottom and left to right, so this section is located in the top-left corner of the image.

### Measuring distance (i.e. between ideal and given image to make a prediction):

- Example baseline approach: Stack all the images on top of each other and take the average of the value of each pixel (black and white/gray image of a three would be numeric pixel values between 0 and 255 for example). use ` torch.stack` from pytorch

Cannot simply take the difference of an image against the overall averages (ideal image) since that would be misleading and result in negative numbers as well.
To combat this:

- Take the mean of the absolute value of differences (absolute value is the function that replaces negative values with positive values). This is called the mean absolute difference or L1 norm
- Take the mean of the square of differences (which makes everything positive) and then take the square root (which undoes the squaring). This is called the root mean squared error (RMSE) or L2 norm.
- Intuitively, the difference between L1 norm and mean squared error (MSE) is that the latter will penalize bigger mistakes more heavily than the former (and be more lenient with small mistakes).

Pytorch code:

```python
# Here mse stands for mean squared error, and l1 refers to the standard mathematical jargon for mean absolute value (in math it's called the L1 norm).
F.l1_loss(a_3.float(),mean7), F.mse_loss(a_3,mean7).sqrt()
```

- We need an iterable function that we can use to improve the model in an automated way: we could instead look at each individual pixel and come up with a set of weights for each one, such that the highest weights are associated with those pixels most likely to be black for a particular category. For instance, pixels toward the bottom right are not very likely to be activated for a 7, so they should have a low weight for a 7, but they are likely to be activated for an 8, so they should have a high weight for an 8

## PyTorch/Numpy

- Good [video](https://www.youtube.com/watch?v=hBBOjCiFcuo&t=4542s) from FastAI about comparing models and inspecting them.
- Create arrays or tensors to take advantage of underlying C optimizations.
- Select portions of the tensor or array by selecting as arr[row, col]
  - arr[:,1] # select all rows of the 2nd column (zero based index)
  - tns[1,1:3] # select the seconed and third col vals from the second row
  - tns+1 # add 1 to every value in the tensor array

### Broadcasting:

- PyTorch, when it tries to perform a simple subtraction operation between two tensors of different ranks, will use broadcasting. That is, it will automatically expand the tensor with the smaller rank to have the same size as the one with the larger rank

  - example:

  ```python
  tensor([1,2,3]) + tensor(1)

  # returns tensor([2, 3, 4])
  ```

## Linear Regression
- An approach to figure out a relationship between dependent and independent variables. We can try to find an algorithm that does that for us.
- Linear Regression is relatively fast (compared to K nearest neighbor algorithm etc.)
  - Involves the training of a model which can be used to make a prediction quickly 
- The goal of linear regression is to find an equation that relates an independent variable to a dependent variable that you are trying to predict. "Given x in a dataset, we can predict y".
- Using linear regression allows for as many independent variables as you need (you cannot do the same thing with spreadsheet trend lines etc. where you are limited to relating a single independent and dependent variable)