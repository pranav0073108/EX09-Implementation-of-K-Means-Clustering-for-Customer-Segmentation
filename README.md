EX 9 Implementation of K Means Clustering for Customer Segmentation
DATE:
AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

Equipments Required:
Hardware – PCs
Anaconda – Python 3.7 Installation / Jupyter notebook
Algorithm
1.Data processing 2. Data preprocessing 3. initialize centroids 4. Asign clustures 5. Update centroid

Program:
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: pranav k
RegisterNumber:  2305001026
*/
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt

data=pd.read_csv('/content/Mall_Customers_EX8.csv')
data


X=data[['Annual Income (k$)','Spending Score (1-100)']]

plt.figure(figsize=(4,4))
plt.scatter(X['Annual Income (k$)'],X['Spending Score (1-100)'])
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()

k=3
Kmeans=KMeans(n_clusters=k)
Kmeans.fit(X)
centroids=Kmeans.cluster_centers_


labels=Kmeans.labels_
print("Centroids:")
print(centroids)
print("Labels:")
print(labels)

colors=['r','g','b']
for i in range(k):
  cluster_points=X[labels==i]
  plt.scatter(cluster_points['Annual Income (k$)'],cluster_points['Spending Score (1-100)'],color=colors[i],label=f'Cluster{i+1}')
  distances=euclidean_distances(cluster_points,[centroids[i]])
  radius=np.max(distances)
  circle=plt.Circle(centroids[i],radius,color=colors[i],fill=False)
  plt.gca().add_patch(circle)
plt.scatter(centroids[:,0],centroids[:,1],marker='*',s=200,color='k',label='Centroids')
plt.title('K-means Clustering')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()
Output:
image image image image

Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
