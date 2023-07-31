from sklearn import datasets
from sklearn.model_selection import train_test_split as tts
from sklearn.naive_bayes import MultinomialNB as MNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler as MMS
import numpy as np

# Fetch dataset
dataset = datasets.load_iris()
features = dataset.data
labels = dataset.target

# MultinomialNB requires non-negative input.
scale_transformer = MMS(feature_range=(1, 5))
scaled_features = scale_transformer.fit_transform(features)

# Segment data
X_train, X_test, y_train, y_test = tts(scaled_features, labels, test_size=0.25, random_state=1)

# Laplace smoothing
model = MNB(alpha=1)

# Model training
model.fit(X_train, y_train)

# Test predictions
predicted_labels = model.predict(X_test)

# Evaluate accuracy
print("Accuracy Evaluation: ", accuracy_score(y_test, predicted_labels))

# Show predictions
class_probabilities = model.predict_proba(X_test)
print("Predicted Class Probabilities:\n", class_probabilities)

# User input
user_input = input("Please enter 4 values, separated by spaces: ")
user_features = np.array(user_input.split(), dtype=float).reshape(1, -1)
user_features = scale_transformer.transform(user_features)  # Scale user input

# Predict probability
user_prediction = model.predict(user_features)
user_probabilities = model.predict_proba(user_features)
print(f"Predicted class: {user_prediction}")
print(f"Predicted class probabilities: {user_probabilities}"
