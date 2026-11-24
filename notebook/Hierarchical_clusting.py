import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score

df = pd.read_csv(r"C:\Users\ASUS\PycharmProjects\SA4110\CAproject\data\scaler_version.csv")
sample_df = df.sample(n=2000,random_state=42)

print(sample_df.shape)
print(sample_df.isna().sum())

#instantialize the hierarchical clusting method
#the argument here is n_cluster

model = AgglomerativeClustering(linkage = "ward", n_clusters = 3) 

#cluster model don't need to change or calculate data so use fit_predict = fit + predict
cluster_labels = model.fit_predict(sample_df)
print(cluster_labels[0:5])

#here need to use sample_df or it would cause index unmatch because I just take 2000 sample to do
result_df = sample_df.copy()
result_df["Prediction"] = cluster_labels
print(result_df.head())

score = silhouette_score(sample_df, cluster_labels)
print(score)

sns.scatterplot(
    data = result_df,
    x = "Age",
    y = "Rating",
    hue = "Prediction" # it means that given a different colour to each value at Prediction column  
)

plt.show()

plt.figure(figsize=(12,6))

# actually here calculate again is repeated code but easy to understand for me since I need to draw the diagram
z = linkage(sample_df,method="ward")

dendrogram(
    z,
    # labels here only need to careful that the length need to be equals to z, the quantity of the sample data
    labels=sample_df.index.to_numpy(),   # find related columns to be the label cause in our dataset don't have ID so use its original index to be the label 
    leaf_rotation=90, # leaf labels are the labels shown at the bottom. without rotation, labels can overlap badly
    leaf_font_size=8 # set the size of the font
)

plt.title("Dendrogram of Clothes Samples")
plt.xlabel("Clothes Samples")
plt.ylabel("Distance")

plt.show()

