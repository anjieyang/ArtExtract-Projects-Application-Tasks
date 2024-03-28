from transfromers import AutoImageProcessor, AutoModel
import os
from PIL import Image
import torch
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics import silhouette_score

# Define the enviroment
device = torch.device("cuda:0")

# Loading DINOv2 model
model_folder = '/home/user/anjieyang/dinov2'
processor = AutoImageProcessor.from_pretrained(model_folder)
model = AutoModel.from_pretrained(model_folder).to(device)

# Define the image folder path
image_folder = '/home/user/anjieyang/downloaded_images'

# Get the list of images
image_files = [os.path.join(image_folder, file) for file in os.listdir(image_files) if file.endswith(('png', 'jpg', 'jpeg'))]

# Define the batch size
batch_size = 256

# Extract image features
image_features = []
for i in range(0, len(image_files), batch_size):
    batch_files = image_files[i : 1 + batch_size]
    batch_features = []
    for image_file in batch_files:
        image = Image.open(image_file)
        with torch.no_grad():
            inputs = processor(
                images=image, 
                return_tensors="pt").to(device)
            outputs = model(**inputs)
            image_feature = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            batch_features.append(image_feature)
    # Convert to NumPy array
    batch_features = np.concatenate(batch_features)
    image_features.append(batch_features)

# Combine the features from all batch to an array
image_features = np.concatenate(image_features)

# Clustering
kmeans = KMeans(n_clusters=6)   # Based on elbow rule
kmeans.fit(image_features)
labels = kmeans.labels_

# Print image count of each cluster
for cluster_label in range(max(labels) + 1):
    num_images = np.sum(labels == cluster_label)
    print(f"Cluster {cluster_label + 1}: {num_images} images")

# User Silhouette Index to evaluate the quality of clusters
silhouette_score = silhouette_score(image_features, labels)
print("The Silhouette Index is ", silhouette_score)
