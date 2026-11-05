from PreProcessing import *
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt

df = pd.read_csv("../data/H2.csv")

column_drop = ['IsCanceled', 'Hotel', 'ReservationStatus', 'ReservationStatusDate', 'Agent', 'Country', 'Company', 'ADR',
               'AssignedRoomType', 'ArrivalDateYear']
features = df.drop(columns=column_drop, errors="ignore")

numerical_columns = features.select_dtypes(include="number").columns.tolist()
categorical_columns = features.select_dtypes(exclude="number").columns.tolist()

preprocessor = make_preprocessor(numerical_columns, categorical_columns)
new_features = preprocessor.fit_transform(features)

bestSilhouetteScore = 0
bestSilhouetteK = 0
k_values = []
silhouette_scores = []
sample_features = new_features[:10000]

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=0)
    clusters = kmeans.fit_predict(sample_features)
    silhouette = silhouette_score(sample_features, clusters)
    k_values.append(k)
    silhouette_scores.append(silhouette)
    if silhouette > bestSilhouetteScore:
        bestSilhouetteScore = silhouette
        bestSilhouetteK = k


final_kmeans = KMeans(n_clusters=bestSilhouetteK, random_state=42)
final_clusters = final_kmeans.fit_predict(new_features)
dbs = davies_bouldin_score(new_features, final_clusters)
print("Best Silhouette score: ", bestSilhouetteScore)
print("Best Silhouette K: ", bestSilhouetteK)
print("Best davies bouldin score: ", dbs)

plt.figure(figsize=(8, 5))
plt.plot(k_values, silhouette_scores, marker='o', linestyle='-', color='b')
plt.axvline(x=bestSilhouetteK, color='r', linestyle='--', label=f'Best K = {bestSilhouetteK}')
plt.title('Silhouette Score vs Number of Clusters (K)')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.xticks(k_values)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("../outputs/silhouette_scores_plot.png", dpi=150)
plt.close() # Close the plot to free memory
