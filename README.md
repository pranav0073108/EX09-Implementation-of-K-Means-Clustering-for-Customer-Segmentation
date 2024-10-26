# EX 9 Implementation of K Means Clustering for Customer Segmentation
## DATE:
## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Select the number of clusters K (the number of customer segments you want to create).
2. For each customer (or data point), calculate the Euclidean distance from the customer to each centroid.
3. Repeat the assignment step based on these updated centroids.
4.At convergence, each cluster represents a segment of customers with similar behavior or characteristics. 

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: pranav k
RegisterNumber:2305001026
*/
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt

data=pd.read_csv('/content/Mall_Customers_EX8.csv')
data

x = data[['Annual Income (k$)','Spending Score (1-100)']]


plt.figure(figsize=(4,4))
plt.scatter(data['Annual Income (k$)'],data['Spending Score (1-100)'])
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()

K = 3

Kmeans = KMeans(n_clusters=K)

Kmeans.fit(X)


centroids=Kmeans.cluster_centers_


labels=Kmeans.labels_
print("Centroids:")
print(centroids)
print("Labels:")
print(labels)


colors=['r','g','b']
for i in range(K):
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
```

## Output:

![image](https://github.com/user-attachments/assets/44993ba2-3138-458c-9d62-bb95ebc77d38)

![image](https://github.com/user-attachments/assets/087167dd-fe7d-42d9-ae56-73344fbcaf78)

![image](https://github.com/user-attachments/assets/6989ac6c-23f0-43a9-a416-56d76b28e755)

![image](https://github.com/user-attachments/assets/92043538-fb28-41e3-a894-8419f18723cc)

![image](https://github.com/user-attachments/assets/af3934b3-f402-42e9-b605-5b34a36c8657)






## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
