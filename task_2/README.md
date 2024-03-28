# Building a Model to Find Similarities in Paintings

### Objective
The objective is to develop a model capable of identifying similarities in paintings using the National Gallery of Art open data set.

### Approach Selection
We utilized a pre-trained DINOv2 model for our task. DINO (Self-Distillation with No Labels) leverages self-supervised learning to understand intricate visual representations without requiring labeled data. This characteristic makes it exceptionally suitable for analyzing artworks where labeling can be subjective and labor-intensive.

The process involves the following steps:
1. Feature Extraction: Use DINOv2 to convert paintings into high-dimensional feature vectors, capturing essential visual characteristics.
2. Clustering: Apply K-Means clustering to group paintings based on similarity in the high-dimensional feature space.
3. Evaluation: Use silhouette score to evaluate the quality of the clustering, ensuring that similar paintings are grouped while dissimilar ones are separated.

### Strategy
We first downloaded the dataset from the National Gallery of Art open data set and preprocess the images to a uniform size, ensuring they are compatible with the DINOv2 model input requirements.

Then we employed the DINOv2 model to extract features from each painting. Given the model's ability to capture detailed visual patterns, it can identify nuanced similarities across artworks.

We then applied PCA to reduce dimensionality while preserving significant variance and utilized K-Means to cluster the feature vectors. The number of clusters was determined by the elbow rule.

In the end we assessed the clustering quality using the silhouette score, which measures how similar an object is to its own cluster compared to other clusters. A high silhouette score indicates well-defined clusters. Adjust the number of clusters or the feature extraction process based on this feedback.

### Evaluation Metrics
Because the ground truth labels are not available, we use Silhouette Score for evaluation. It ranges from -1 to 1, where a high value indicates that the object is well matched to its own cluster and poorly matched to neighboring clusters.

We believe utilizing DINOv2 for feature extraction followed by K-Means clustering presents a promising approach to finding similarities in paintings. The self-supervised nature of DINOv2 makes it highly adept at capturing complex patterns without the need for labeled data. Evaluation through the silhouette score ensures the model's effectiveness in grouping paintings by similarity.