import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

data=pd.read_csv("train.csv")

"""
# Check for missing values
missing_values = data.isnull().sum()

# Check if there are any missing values in the dataset
if missing_values.sum() > 0:
    print("The dataset contains missing values.")
else:
    print("The dataset does not contain any missing values.")
"""


def identify_dominant_features(data, threshold=0.99):
    """
    Identifies features where the majority of the values are the same.
    
    Parameters:
        data (pd.DataFrame): The input DataFrame.
        threshold (float): The threshold for dominance. Features where the most frequent value
                           exceeds this proportion will be flagged.
                           
    Returns:
        list: A list of column names to remove.
    """
    features_to_remove = []
    for col in data.columns:
        # Calculate the proportion of the most frequent value
        value_counts = data[col].value_counts(normalize=True)
        dominant_proportion = value_counts.max()
        
        # Check if it exceeds the threshold
        if dominant_proportion > threshold:
            features_to_remove.append(col)
    
    return features_to_remove


features_to_remove = identify_dominant_features(data, threshold=0.99)


"""
# Create a single figure with multiple subplots
num_features = len(features_to_remove)
fig, axes = plt.subplots(num_features, 1, figsize=(8, 6 * num_features))

# If there's only one feature, axes will not be an array, so we handle it accordingly
if num_features == 1:
    axes = [axes]

# Plot histograms for dominant numerical features
for i, feature in enumerate(features_to_remove):
    if data[feature].dtype in ['int64', 'float64']:  # Check if the feature is numerical
        axes[i].hist(data[feature], bins=30, color='skyblue', edgecolor='black')
        axes[i].set_title(f"Histogram of {feature}")
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel('Frequency')

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()

"""
# print("Features to remove due to dominance:\n", features_to_remove)
target='type_of_attack'

X = data.drop(columns=features_to_remove)  
X = X.drop(columns=target)

y = data[target]  
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
# print(numerical_features)
categorical_features = X.select_dtypes(include=['object', 'category']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=47)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# cm = confusion_matrix(y_test, y_pred)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
# disp.plot(cmap='viridis')
# plt.show()

# Load the test dataset
test_data = pd.read_csv("test.csv")

# Ensure that the test dataset includes only the relevant features
X_test_data = test_data.drop(columns=features_to_remove)

# Predict the target values using the pipeline (which includes preprocessing)
test_predictions = model.predict(X_test_data)

# Create the output DataFrame in the desired format
output = pd.DataFrame({
    "Id": test_data.index + 1,  # Assuming row numbers as IDs
    "type_of_attack": test_predictions
})

# Save the output DataFrame to a CSV file
output_file = "test_predictions.csv"
output.to_csv(output_file, index=False)

print(f"Predictions saved in the desired format to {output_file}")
