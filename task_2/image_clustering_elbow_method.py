import os
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Loading images
data_dir = '/home/user/anjieyang/downloaded_images'
images = []
target_size = (224, 224)

for file in os.listdir(data_dir):
    if file.endswith('.jpg'):
        img = Image.open(os.path.join(data_dir, file))
        img = img.resize(target_size)
        images.append(np.asarray(img).reshape(-1))

# Pre-processing the data
scaler = StandardScaler()
X = scaler.fit_transform(iamges)

# Use PCA for dimensionality reduction
pca = PCA(n_components=50)
X_pca = pca.fit_transform(X)

# Elbow rule
inertias = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_pca)
    inertias.append(kmeans.inertia_)

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
plt.plot(range(2, 11), inertias)

plt.xlabel("Number of clusters")
plt.ylabel("Inertia")
plt.savefig("/home/user/anjieyang/elbow_plot.png", bbox_inches='tight', dpi=800)