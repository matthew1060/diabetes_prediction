import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load the training data
train_data = pd.read_csv('train.csv')

# Select features and target variable
features = train_data.drop(['Outcome', 'ID'], axis = 1)
target = train_data['Outcome']

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(features, target, test_size = 0.2, random_state = 40)

# Train a machine learning model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
test_data = pd.read_csv('diabetes_test.csv')
test_features = test_data.drop('ID', axis = 1)


# Make predictions
predictions = model.predict(test_features)

# Create a DataFrame with ID and predicted columns
result_df = pd.DataFrame({'ID': test_data['ID'], 'predicted': predictions})

# Save the result to predictions.csv
result_df.to_csv('predictions.csv', index=False)
