# Clustering

- Main idea is to group multidimensional data into k groups.
- Groups are defined based on distance (data points close to each other are part of a group and data points further away are part of a differnet group)
- The goal is to minimize the distance between points within a group, and maximize the distance between groups.
- Each group has an empirical centroid (the center point around which the other points revolve)

# K-means Clustering

### Basic K-means algo:

- Select the right k number (done by the human researcher)
  - Algo will create k centroids at random locations in the dataset
  - Compute sum of squared distances (errors) from all data points (every single one) to all/each of the centroids (as Euclidian distance)
  - label and categorize each data point according to its closest centroid.
  - Take an average of the centroids and move them accordingly (take all the data points labeled for centroid k, and move k centroid to the average of those distances).
  - Repeat steps 3 through 5 until reaching a measure of convergence (i.e. how much the mean is moving the centroid). At a certain point in these iterations the centroid will start to move around less and less - this is the signal to stop iterating.
- Heirarchical and Bayesian k-means are more advanced derivations

### Difficulties of k-means

- It is hard to know what the right k number is a priori.
- While there are methods for determining the proper k, they only work with simple and contrived data - in real world data, the methods fail to give reliable results
- Multi-dimensional clustering is very difficult to visualize
- Repeating/re-computing the algorithm can give different results (since we are randomly placing the centroids initially - not deterministic)
- Does not work well when cluster sizes are different in sizes (one group is larger/smaller than the other)
- There is no gaurantee that clusters in data are purely defined by Euclidian distance (k-means clustering only deals with Euclidian distance in the data space as a measure)
- In practice k-means clustering works well sometimes and does not other times.
- k-means clusters works really well when the cluster centroids are relatively far apart and when the data points are relatively close to their centroids. But when those features break down, then k-means under-performs.

### Code: K-means cluster

- see [notebook](../../statsML/clustdimred/stats_clusterdimred_kmeans.ipynb)
- Use the KMeans function from scikit learn

```python
from sklearn.cluster import KMeans

k = 3 # how many clusters
kmeans = KMeans(n_clusters=k) # create the kmeans object with num clusters
kmeans = kmeans.fit(data) # fit to the data passed in
# group labels - 0,1 or 2 - label tells us which group the data point is in (0 1 or 2)
groupidx = kmeans.predict(data)
# centroids
cents = kmeans.cluster_centers_ # x and y coords for the computed centers

```

## Determining cluster number

- Picking the "right" number of clusters is nearly impossible in many datasets. In practice, the number of clusters is selected based on qualitative and quantitative measures.

### Qualitative testing

## determining the appropriate number of clusters (qualitative)

- we run kmeans with a different number of clusters and look at the results to see if we can infer anything
- The idea in Qualitative testing is to re-run the kmeans over and over and look at the plot results to see if they change a lot
  - If the cluster assignments are changing alot each time you run the code with the same data, that means you are probably using too many clusters

### Quantitative Testing

#### Silhouette Testing

- Silhouette coefficient of a cluster refers to a metric about the quality of the clustering, i.e. how well that clustering worked out
  - we want to look for number of clusters that maximizes the silhouette coefficient

#### Elbow Test

- We also look for the Sum of Squared distances (how far apart data points are from the means)
  - the more clusters you have, the smaller the squared distances are going to be
  - we want to find the "elbow" - look at plot and determine visually which data point has the biggest bend (like an elbow point)
  - that point is the optimal number of clusters
  - see [video](https://www.udemy.com/course/statsml_x/learn/lecture/20246034#questions) at 20:40

# Dbscan (Density-based spatial clustering of applications with noise)

- A different means of clustering than K-means based on local distancing of data points and based on density (how they are grouped more close together).
- Able to label points as noise and identify them as not belonging to a cluster.
- Works for any number of dimensions (including higher dimensions)

### Basic Algorithm

- Pick a random data point N in the dataset
- Look at distances from N to all other data points - see if there are there other data points within a specific distance threshold (i.e. within a pre-defined radius etc.)
  - if not, then the point is considered noise and not belonging to any cluster, pick another random point B
  - Any points within distance threshold to B?
  - If yes, then B and the other data point become a cluster
  - Move to the point within the threshold and ask the same questions as above continuously expanding the cluster
- Based on the tentative clusters established in the previous loop, we ask if the points in the cluster(s) are more than a minimum number of points required for a cluster
  - If there are not enough points in the group, then it is not considered a cluster

#### Two parameters to Dbscan

- Epsilon $\epsilon$: Distance (the step size) which defines a neighborhood threshold or radius encompassing data points (how far data points can be from each other).
  - If this is too small, you get more clusters broken up into many small clusters
  - If it is too large, you get separate clusters combined into one.
  - It is important to try to get this parameter right
  - Epsilon is in the scale of the data (if you multiply your data by a lot, then you need to increase/change epsilon)
  - Typically Euclidian distance is used
- Minimum number of clusters - required num of data points to qualify as a cluster
  - Too small you get many small clusters
  - Too large and you get true small clusters that get ignored.
  - There is typically a large range of this parameter where the results will not change much.
  - A rule of thumb is that the minimum num of points should be 2x the dimensions of the data. (i.e. for data on x and y, i.e. 2 dimensions, the min num should be 4)

### K-means vs. Dbscan:

- **a major difference between dbscan and k-means is that dbscan will simply exclude points that don't fit into a cluster (assign them as noise), whereas k-means will assign every single data point to a cluster.**
  - see comparison in [notebook](../../statsML/clustdimred/stats_clusterdimred_dbscan.ipynb)
  - Implication is when you think you are dealing with a lot of outliers, Dbscan is a better method for clustering.
- Both look at distances to determine cluster assignment
  - Kmeans is based on distance to centroids
  - Dbscan based on distance to neighbors
- K-means: global distances vs. Dbscan: local distances
- K-means: You specify number of clusters a priori and distance threshold is determined by algorithm vs. Dbscan: You specify distance threshold and algo determines num of clusters.
- K-means: works well for spherical clusters vs. Dbscan: works for any/different shape clusters
- Both are sensitive to scaling effects (esp dbscan)
  - ex. if you have one cluster with data closer together and another with data further apart, dbscan will probably not give you a good solution.
- K-means: can assign every point to a cluster vs. Dbscan: some points are noise and unlabeled/unassigned to a cluster

## Code: Dbscan

- see [notebook](../../statsML/clustdimred/stats_clusterdimred_dbscan.ipynb)
- Use DBSCAN function from sci kit learn package:

```python
from sklearn.cluster import DBSCAN

# build a model using the DBSCAN fn
# Epsilon(eps) is the step size in the algo for looking for points within that distance from each other
# min_samples is the minimum num of data points needed to be considered a cluster
clustmodel = DBSCAN(eps=.6,min_samples=6).fit(data)
```

# K-Nearest Neighbor (KNN)

## The Basic Algo

- From any data point `i` find the `k` nearest neighboring data points (i.e. 3 nearest neighbors, etc.)
- Assign data point `i` to the group/cluster that the majority of those neighboring data points belongs to.
- see [video](https://www.udemy.com/course/statsml_x/learn/lecture/20246048#content) at timestamp 1:55 for illustration.
- Note that you can consider all the distances and then choosing the closest group of data to determine K
  - see [video](https://www.udemy.com/course/statsml_x/learn/lecture/20246048#content) at timestamp 6:20 for illustration and explanation.

### Difference from KMeans Clustering:

- The data has to be categorized and assigned labels ahead of time when using K Nearest Neighbor.
  - With K-means clustering, the dataset is not labeled or categorized before applying the procedure.
- Both do however rely on the assumption that Euclidian distance between points is the way to determine group membership

## Code: K-Nearest Neighbor

- See [notebook](../../statsML/clustdimred/stats_clusterdimred_KNN.ipynb)
- Use the KNeighborsClassifer from scikit learn

```python
from sklearn.neighbors import KNeighborsClassifier

## In practice you just use this function:
# use euclidian distance, k = 3 for example
knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
knn.fit(data,grouplabels) # grouplabels = np.concatenate((np.zeros(nPerClust),np.ones(nPerClust)))

# use the predict method to input the newpoint to predict which group they belong to
whichgroupP = knn.predict(newpoint.reshape(1,-1)) # need to reshape to fit the argument expectation for .predict()

print('New data belong to group ' + str(whichgroupP[0]))
```

# Principal Components Analysis (PCA)

- Identifying the most important axes that better capture the features and correlation of the data (i.e. axes other than the x or y axis on a graph for ex.)
  - See illustration in [video](https://www.udemy.com/course/statsml_x/learn/lecture/20246056#content) at timestamp 1:28
  - You "rotate" the data to conform to the new axes (Principle Components)
- Useful for dimension compression (reducing data down from high dimensions to make it simpler to analyze)
- Note: may be suboptimal or misleading when interpreting Principle Components as "factors" or unique sources of variance in the data.

### PC Space

- The data shown on the principle and PCA axes
  - Instead of X and Y coords, we have PC 1 ("Principle Component 1") and PC 2 coords for example.
- The data is rotated to conform to the PCA axes.
  - The most of the concentration of the data is along PC 1, the next most amount of varianced data is aligned with PC 2 etc.

### Data Compression

- Take each data point and project it only onto certain axes or dimensions that you care about.
- Once you have the PCA done, you can flatten some of the dimensions in the data space (because they are assumed to be noise and not meaningful variability)
  - i.e. if you think that data along a particular axes like PC 2 is noise then you can compress the data down to the first dimension to only account for PC 1 axis.
  - see [video](https://www.udemy.com/course/statsml_x/learn/lecture/20246056#content) at timestamp 3:46
- This can make analyses simpler by reducing dimensions (i.e. useful if you are working with a high number of dimensions that you can compress down)

## Math of PCA

- PCA starts from a covariance Matrix
- Perform an eigen value decomposition on the covariance matrix
  - decomposes the matrix into eigenvectors and eigenvalues
  - eigenvector has same num of elements as the matrix shape
  - multiply the eigenvectors by the original data that was used to create the covariance matrix
    - the eignenvector is a way of taking a weighted combination of all the different features in your dataset.
    - the eigenvectors point to some space in the dataspace (they don't tell us which directions are more important however)
    - The eigenvalue will help us determine which directions are more important (a higher eigenvalue indicates more importance for the corresponding eigenvector)
    - The eigenvector/largest eigenvalue is the first principle component (PC 1)
    - The eigenvectors are ways of rotating the data
- Look at a spectrum of eigenvalues to get the eigenspectra.
  - Look for the large eigenvalues that are high on the y axis of the spectrum plot - the higher the value, the more important the principle component is.
    - The rest of the eigenvalues that are low are called noise or more precisely are low variance components
    - see [video](https://www.udemy.com/course/statsml_x/learn/lecture/20246056#content) at timestamp 9:59
    - You can use the highest Eigenvalues to determine how much you can compress the dimensions down to include only the highest values.

## Limitations of PCA

1. All principle components are forced into being Orthogonal

- The correlation between any two principle components is necessarily 0.
- This can lead to unexpected results in some cases

2. Variance equates to relevance in PCA

- Whatever data has the most or least amount of variance - that is what PCA will focus and pick up on.
- If you have meaningful features in data that are isolated to a small number of characteristics, then PCA could disregard them because it sees they are not strongly correlated with a lot of other things in the data. (sometimes the unusual things are the most important things in data)
- It might be possible that using PCA causes you to push out signal in the case where noise is high amplitude and the majority in the data.

## Code: PCA

- see [notebook](../../statsML/clustdimred/stats_clusterdimred_PCA.ipynb)
- Use the PCA function from Scikitlearn

```python
from sklearn.decomposition import PCA

# PCA using scikitlearn's function
pca = PCA().fit(y)

# get the PC scores
# apply the transformation of the principle components analysis to the data to get pc scores
# The pc scores are the data transformed into the PC space
pcscores = pca.transform(y)
```

# Independent Component Analysis (ICA)

- based on the central limit theorem.
- Starts from the assumption that true signals are non-gaussian distributed, and random mixtures of signals ARE Gaussian distributed (and therefore noise or not useful).
- From the ICA perspective, it is a bad sign to see a Gaussian distribution - it means that you are either looking at noise or a random mixture of signals.
  - The goal is to pool the data across different features of the dataset in order to make the weighted combinations of the features look as least gaussian distributed as possible.
- Kurtosis can be used to measure the Gaussian-ness.
  - refers to the fatness of the tails
  - Gaussian distributions have relatively small tails, or low Kurtosis

## Steps in ICA

- Whiten the data - remove all the covariances
  - This step is done via PCA and involves converting the covariance matrix into a matrix that has all zeros on the off diagonal (all the correlations across all the variables are set to zero)
- ICA will then look for the relationships between the variables of the whitened data (note that the correlation between the data on the PC axes is zero by definition of how Principle Components work)
  - see [video](https://www.udemy.com/course/statsml_x/learn/lecture/20246064#questions) around timestamp 6:30
- ICA will look at the distributions from PCA and if a distribution looks Gaussian it will try to shift around the data and rotate the PC axes in order to make the distribution look less Gaussian.
  - Note that strict linear dependencies in the dataset are removed, there is still some shared information that is preserved (ICA will pick up on this shared info).
- ICA will rotate the PC axes and this is where it diverges from PCA
  - PCA is orthogonal and rotates the data in the space
  - ICA is more flexible, it takes the two vectors/PC axes and rotate them obliquely independntly of each other.
  - see [video](https://www.udemy.com/course/statsml_x/learn/lecture/20246064#questions) at timestamp 8:34
- This rotation and shifting creates an independent component space that changes the distribution of the signal (again, the goal is to move it away from gaussian shape)

## Code: ICA

- see [notebook](../../statsML/clustdimred/stats_clusterdimred_ICA.ipynb)
- Use the FastICA fn from scikitlearn toolbox:

```python
from sklearn.decomposition import FastICA

# two non-Gaussian distributions
# First channel: 40% of dataset 1 and 30% of dataset 2
# second channel: 80% of dataset 1 minus 70% of dataset 2
data = np.vstack((.4*dist1+.3*dist2, .8*dist1-.7*dist2))

# ICA and scores
fastica = FastICA(max_iter=10000,tol=.0000001)
b = fastica.fit_transform(data)
iscores = b@data
```

# Note on advanced analysis

- Uncertainty increases as analyses become more sophisticated. You can't assume with methods like PCA and ICA that throwing data at it will give you an appropriate result.
  - Anytime you apply clustering make sure you're aware of the assumptions those methods make on the data.
  - Ask if the data meets those assumptions and if you are getting from them exactly what you were looking to infer from the data.
  - \*It can help to simulate data so that you know what the result and ground truth is to test the methods on, and if you get the expected result you can proceed, otherwise should be cautious about applying the method to real data.
- Simple analyses like T-tests, Correlations, Regression analyses will give you better more reliable results as long as the model is set up correctly.
