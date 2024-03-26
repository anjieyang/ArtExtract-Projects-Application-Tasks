# Task 1: Convolutional-Recurrent Architectures

### Objective:
The task aims to develop a model that leverages convolutional and recurrent neural network architectures for classifying various attributes of artworks, such as Style, Artist, Genre, among others. The ArtGAN dataset, specifically the [WikiArt](https://www.google.com/url?q=https://github.com/cs-chan/ArtGAN/blob/master/WikiArt%2520Dataset/README.md&sa=D&source=editors&ust=1711495904736468&usg=AOvVaw2UQIji7yaoriDH1CzIgyo5) Dataset, serves as the primary source of data for this project.

### Approach:
For this task, we chose a hybrid approach combining convolutional neural networks (CNNs) with recurrent neural networks (RNNs), specifically Long Short-Term Memory (LSTM) units. The CNN component, based on a pre-trained ResNet-101 model, extracts high-level features from the input images. We then pass these features to an LSTM network, which helps in understanding the sequential or temporal dependencies between features, a characteristic that might be beneficial given the nature of artistic styles and progressions.

The CNN model's last fully connected layer was replaced with an identity function to maintain the extracted features, which are then fed into the LSTM network. This network comprises two LSTM layers, followed by a fully connected layer that outputs the predictions for the number of classes defined in the dataset.

### Data Preprocessing:
The data augmentation strategy involved random cropping with padding if needed, ensuring a consistent input size for the network and introducing variability into the dataset to reduce overfitting. The dataset was split into training (60%), validation (20%), and test (20%) sets to evaluate the model's performance adequately.

In order to ensure the integrity of the dataset, we have incorporated an addtional preprocessing step aimed at identifying and exclusing potentially corrupt images.

### Evaluation Metrics:
The model's performance was evaluated using Accuracy and the F1 score. Accuracy provides a straightforward metric for overall performance, while the F1 score, being the harmonic mean of precision and recall, offers a more nuanced view, especially useful in imbalanced datasets.

### Results:
The training process involved tuning hyperparameters such as the learning rate, batch size, and regularization term. The learning rate was adjusted during training to improve convergence. The model was trained for a predefined number of epochs, with periodic logging of loss values and accuracy metrics to monitor the training process.

The validation phase after each epoch provided insights into the model's generalization capabilities, helping to avoid overfitting. Finally, the model was evaluated on a test set, which demonstrated its ability to classify unseen data accurately.

### Outliers:
To identify outliers, such as artworks incorrectly classified despite their assignment to a specific artist or genre, we can analyze cases where the model's confidence in its predictions is low. Additionally, comparing the model's predictions with the true labels and examining the misclassified examples in detail can reveal common characteristics of outliers.