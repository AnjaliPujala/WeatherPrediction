# WeatherPrediction

This project utilizes the Gaussian Naive Bayes algorithm to predict weather conditions based on a dataset. The dataset includes features such as precipitation, maximum temperature, minimum temperature, and wind speed.

## Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn

## Dataset

The dataset is loaded from the 'seattle-weather.csv' file. Replace 'your_dataset.csv' with the actual filename if different.

## Feature Selection

The following features are selected for weather prediction:

- Precipitation
- Maximum Temperature
- Minimum Temperature
- Wind Speed

## Data Preprocessing

Label encoding is applied to the 'weather' column using `LabelEncoder`. The dataset is split into features (X) and the target variable (y). Feature names are assigned to X to suppress warnings.

## Model Training

The dataset is split into training and testing sets using the `train_test_split` method. A Gaussian Naive Bayes model is then trained on the training set.

```python
# Code snippet for model training
# ...

# Evaluate the model
accuracy = np.mean(y_pred == y_test)
print(f"Accuracy on the test set: {accuracy}")
