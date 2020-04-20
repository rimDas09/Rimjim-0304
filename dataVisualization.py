import pandas as pd
import numpy as np
from Cython import inline
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

from kmeans import run_kmeans

data = pd.read_csv('data-sample-clinton-50k.csv')
print(data.head())
km = KMeans(n_clusters=5, init='k-means++', n_init=10)
km.fit(df1)

