import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB

# Load the dataset (replace 'your_dataset.csv' with the actual filename)
dataset = pd.read_csv('seattle-weather.csv')

# Data preprocessing
dataset = dataset[['precipitation', 'temp_max', 'temp_min', 'wind', 'weather']]

le = LabelEncoder()
dataset['weather'] = le.fit_transform(dataset['weather'])

X = dataset.drop(['weather'], axis=1)
y = dataset['weather']

# Assign feature names to X (this will suppress the warning)
X.columns = ['precipitation', 'temp_max', 'temp_min', 'wind']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Make predictions on the test set
y_pred = gnb.predict(X_test)

# Evaluate the model
accuracy = np.mean(y_pred == y_test)
print(f"Accuracy on the test set: {accuracy}")

# Now you can use the trained model (gnb) to make predictions for new data.
# For example:
new_data = np.array([[0.2, 25, 15, 10]])
new_pred = gnb.predict(new_data)
predicted_weather = le.inverse_transform(new_pred)
print(f"Predicted weather for new data: {predicted_weather[0]}")
