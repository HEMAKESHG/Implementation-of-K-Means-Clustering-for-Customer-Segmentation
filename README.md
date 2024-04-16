# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 1.Import the necessary packages using import statement.
2.Read the given csv file using read_csv() method and print the number of contents to be displayed using df.head().
3.Import KMeans and use for loop to cluster the data.
4.Predict the cluster and plot data graphs.
5.Print the outputs and end the program. 

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: Hemakesh G
RegisterNumber:  212223040064
*/

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt

# Load data from CSV
data = pd.read_csv('/content/Mall_Customers_EX8.csv')
data

# Extract Features
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]
X

plt.figure(figsize=(4, 4))
plt.scatter(data['Annual Income (k$)'],data['Spending Score (1-100)'])
plt.xlabel('Annua Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()

# Number of clusters
k = 5

#Initialize KMeans
kmeans = KMeans(n_clusters=k)
# Fit the data
kmeans.fit(X)

centroids = kmeans.cluster_centers_
# Get the cluster labels for each data point
labels = kmeans.labels_
print("Centriods:")
print(centroids)
print("Labels:")
print(labels)

colors = ['r', 'g', 'b','c','m'] # Define colors for each cluster
for i in range(k):
  cluster_points = X[labels == i] # Get data points belonging to cluster i
  plt.scatter(cluster_points['Annual Income (k$)'], cluster_points['Spending Score (1-100)'],
              color=colors[i], label=f'Cluster {i+1}')
  # Find minimum enclosing circle
  distances = euclidean_distances(cluster_points, [centroids[i]])
  radius = np.max(distances)
  circle = plt.Circle(centroids[i], radius, color=colors[i], fill=False)
  plt.gca().add_patch(circle)
# Plotting the centriods 
plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=200, color='k', label='Centroids')
plt.title(' K-means Clustering')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True)
plt.axis('equal')  #Ensure aspect ratio is equal
plt.show()
```

## Output:
![K Means Clustering for Customer Segmentation](sam.png)


## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
