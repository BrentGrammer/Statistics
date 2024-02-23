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

- Both look at distances to determine cluster assignment
  - Kmeans is based on distance to centroids
  - Dbscan based on distance to neighbors
- K-means: global distances vs. Dbscan: local distances
- K-means: You specify number of clusters a priori and distance threshold is determined by algorithm vs. Dbscan: You specify distance threshold and algo determines num of clusters.
- K-means: works well for spherical clusters vs. Dbscan: works for any/different shape clusters
- Both are sensitive to scaling effects (esp dbscan)
  - ex. if you have one cluster with data closer together and another with data further apart, dbscan will probably not give you a good solution.
- K-means: can assign every point to a cluster vs. Dbscan: some points are noise and unlabeled/unassigned to a cluster
- **a major difference between dbscan and k-means is that dbscan will simply exclude points that don't fit into a cluster (assign them as noise), whereas k-means will assign every single data point to a cluster.**
  - see comparison in [notebook](../../statsML/clustdimred/stats_clusterdimred_dbscan.ipynb)

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
