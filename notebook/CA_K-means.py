import pandas as pd
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score


df = pd.read_csv(r"C:\Users\ASUS\PycharmProjects\SA4110\CAproject\data\scaler_version.csv")
print(df.head(5))
print(df.shape)
print(df.isna().sum())

model = KMeans(
    n_clusters=3,
    random_state=42
)

cluster_labels = model.fit_predict(df)
print(pd.Series(cluster_labels).value_counts())

df['Cluster'] = cluster_labels
cluster_summary = df.groupby('Cluster')[
    ['Age', 'Rating']
].mean()

print(cluster_summary)
'''
sns.scatterplot(
    data=df,
    x='Age',
    y='Rating',
    hue='Cluster',
    palette='Set1',
    alpha=0.4
)

plt.title('K-Means Clustering (K=3)')
plt.show()
'''

k_values = range(2, 11)
wcss = []
silhouette_scores = []

for k in k_values:
    model = KMeans(
        n_clusters=k,
        random_state=42,
    )
    labels = model.fit_predict(df)

    wcss.append(model.inertia_)

    score = silhouette_score(
        df,
        labels,
        sample_size=2000,
        random_state=42
    )

    silhouette_scores.append(score)

    print(f"K={k}, WCSS={model.inertia_:.2f}, Silhouette={score:.3f}")