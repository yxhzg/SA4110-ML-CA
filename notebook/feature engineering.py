import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from pathlib import Path

output_dir = Path(
    r"C:\Users\ASUS\PycharmProjects\SA4110\CAproject\data"
)

output_dir.mkdir(parents=True, exist_ok=True)


df = pd.read_csv(r"C:\Users\ASUS\PycharmProjects\SA4110\CAproject\women-E\Womens Clothing E-Commerce Reviews.csv")
'''
print(df.head(5))
print(df.shape)
print(df.columns)
print(df.dtypes) #哪些是int float objects
df.info() #哪些是非空数据列以及有多少的非空数据
print(df.describe()) #数值字段的均值、最小值、最大值是多少？
print(df.isna().sum())  #统计每一行的非空数据总数
print(df.isna().mean() * 100)  #计算百分比

print(df[df["Division Name"].isna()])
print(df[df["Department Name"].isna()])
print(df[df["Class Name"].isna()])
'''
cols = ["Division Name", "Department Name", "Class Name"]
print(df[cols].isna().any(axis=1).sum())
print(df[cols].isna().all(axis=1).sum())

df_drop = df.dropna(subset = cols)
duplicate_count = df_drop.drop(columns = ['Unnamed: 0']) #这一行肯定不一样是id需要先剔除
print("duplicate count: ",duplicate_count.duplicated().sum()) #duplicated()可以检查是否重复，后面若出现重复的会直接返回true
                                         #同时用sum（）来统计
print("duplicate count with keep: ",duplicate_count.duplicated(keep=False).sum())
#print(duplicate_count[duplicate_count.duplicated(keep=False)])
duplicate_rows = duplicate_count[duplicate_count.duplicated(keep=False)]
print(duplicate_rows["Review Text"].isna().sum())

#df_clean = df_drop.drop_duplicates()

df_clean = df_drop.drop_duplicates(
    subset=df_drop.columns.drop('Unnamed: 0') #给这个删除重复添加条件，排除id的影响
).copy()
print(df_clean.shape)
'''
print(df_clean["Age"].describe())

print((df_clean["Age"]>79).sum())
print(df_clean[df_clean["Age"]>79]) #percentage is 0.46%
print((df_clean["Age"]==99).sum())

numerical_cols = ['Age', 'Rating', 'Positive Feedback Count']

for col in numerical_cols:
    print("\nColumn:", col)

    print(df_clean[col].describe())

    Q1 = df_clean[col].quantile(0.25) #calculate the correlate percentage point
    Q3 = df_clean[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = (
        (df_clean[col] < lower) |
        (df_clean[col] > upper)
    ).sum()

    print("Lower Bound:", lower)
    print("Upper Bound:", upper)
    print("Outliers:", outliers)
    '''
print(df_clean['Recommended IND'].value_counts())
print(pd.crosstab(
    df_clean['Rating'],
    df_clean['Recommended IND']
))
'''
features = [ 'Age',
    'Rating',
    'Recommended IND',
    'Positive Feedback Count']


sns.pairplot(df_clean[features])
plt.suptitle("Scatter Matrix of Customer Review Features")
plt.tight_layout()
plt.show()

correlation_matrix = df_clean[features].corr()
sns.heatmap(data = correlation_matrix,annot = True,cmap = "GnBu")
plt.show()
'''

print(df_clean['Department Name'].value_counts())
encoder = OneHotEncoder(sparse_output=False)

department_encoded = encoder.fit_transform(df_clean[['Department Name']])
#print(encoder.get_feature_names_out())

department_df = pd.DataFrame(department_encoded,
                             columns = encoder.get_feature_names_out(),
                             index = df_clean.index)
print(department_df.head(5))

selected_df = df_clean[["Age","Rating"]]
clusting_data = pd.concat([selected_df,department_df],axis=1)

df_clean['Age Group'] = pd.cut(
    df_clean['Age'],
    bins=[17, 29, 44, 59, 100],
    labels=['18-29', '30-44', '45-59', '60+']
)

print(df_clean["Age Group"].head(5))
#print(df_clean['Age Group'].value_counts().sort_index())
#(df_clean['Age Group'].isna().sum())

scaler = StandardScaler()
num_scaler = scaler.fit_transform(df_clean[["Age",'Rating']])
print(num_scaler[0:5])

num_scaler_df = pd.DataFrame(num_scaler,columns = ['Age','Rating'],
                             index = df_clean.index)

scaler_version = pd.concat(
    [num_scaler_df, department_df],
    axis=1
)

print(scaler_version.head())
print(scaler_version.shape)

scaler_version.to_csv( r"C:\Users\ASUS\PycharmProjects\SA4110\CAproject\data\scaler_version.csv",
    index=False)

Age_group_encoder = encoder.fit_transform(df_clean[['Age Group']])
#print(encoder.get_feature_names_out())

Age_group_df = pd.DataFrame(Age_group_encoder,
                             columns = encoder.get_feature_names_out(),
                             index = df_clean.index)

binning_version = pd.concat([Age_group_df,num_scaler_df["Rating"],department_df],axis=1)
print(binning_version.head(5))

binning_version.to_csv(r"C:\Users\ASUS\PycharmProjects\SA4110\CAproject\data\binning_version.csv",
    index=False)

clean_version = pd.concat([df_clean[["Age","Rating"]],department_df],axis=1)
clean_version.to_csv(r"C:\Users\ASUS\PycharmProjects\SA4110\CAproject\data\clean_version.csv",
    index=False)