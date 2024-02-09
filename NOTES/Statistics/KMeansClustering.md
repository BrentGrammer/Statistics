# K-means Clustering

- Main idea is to group multidimensional data into k groups.
- Groups are defined based on distance (data points close to each other are part of a group and data points further away are part of a differnet group)
- The goal is to minimize the distance between points within a group, and maximize the distance between groups.
- Each group has an empirical centroid (the center point around which the other points revolve)

## K-means Algorithms

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

## Code: K-means cluster

- see [notebook](../../statsML/clustdimred/stats_clusterdimred_kmeans.ipynb)
- Use the KMeans function from scikit learn

```python
from sklearn.cluster import KMeans

```

## Determining cluster number

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
