import pandas as pd
df=pd.read_csv('heart.csv')
df.head()
df['target'].value_counts()
df.corr()
#"""
#from sklearn.preprocessing import StandardScaler, LabelEncoder
#label_encoder = LabelEncoder()
#df['age'] = label_encoder.fit_transform(df['age'])
#df['sex'] = label_encoder.fit_transform(df['sex'])
#df['cp'] = label_encoder.fit_transform(df['cp'])
#df['trestbps'] = label_encoder.fit_transform(df['trestbps'])
#df['chol'] = label_encoder.fit_transform(df['chol'])
#df['fbs'] = label_encoder.fit_transform(df['fbs'])
#df['restecg'] = label_encoder.fit_transform(df['restecg'])
#df['thalach'] = label_encoder.fit_transform(df['thalach'])
#df['exang'] = label_encoder.fit_transform(df['exang'])
#df['oldpeak'] = label_encoder.fit_transform(df['oldpeak'])
#df['slope'] = label_encoder.fit_transform(df['slope'])
#df['ca'] = label_encoder.fit_transform(df['ca'])
#df['thal'] = label_encoder.fit_transform(df['thal'])
#df['target'] = label_encoder.fit_transform(df['target'])

#"""

X=df.drop(['target'],axis=1)
y=df['target']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle = True, test_size = 0.20, random_state = 44)
from sklearn.ensemble import RandomForestClassifier
rnf=RandomForestClassifier()
rnf.fit(X_train,y_train)
y_pred=rnf.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
# Saving the model
import pickle
with open('heart_disease_model_check.pkl', 'wb') as model_file:
    pickle.dump(rnf, model_file)
    # Get feature importances
import matplotlib.pyplot as plt
importances = rnf.feature_importances_

# Get feature names (replace with your feature names)
feature_names = X_train.columns  # If you have a DataFrame

# Sort feature importances in descending order
indices = importances.argsort()[::-1]
sorted_feature_names = [feature_names[i] for i in indices]
sorted_importances = importances[indices]

# Create a bar chart to visualize feature importances
plt.figure(figsize=(12, 6))
plt.bar(range(len(sorted_importances)), sorted_importances, align='center')
plt.xticks(range(len(sorted_importances)), sorted_feature_names, rotation=90)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Feature Importances')
plt.show()
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))
import pickle
import pandas as pd

# Load the trained Random Forest model
with open('heart_disease_model_check.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Define the input data as a dictionary (you can replace these values)
input_data = {
    'age': 53,
    'sex': 1,
    'cp': 0,
    'trestbps': 140,
    'chol': 203,
    'fbs': 1,
    'restecg': 0,
    'thalach': 155,
    'exang': 1,
    'oldpeak': 3.1,
    'slope': 0,
    'ca': 0,
    'thal': 3
}
# Create a DataFrame from the input data
df = pd.DataFrame([input_data])

#from sklearn.preprocessing import StandardScaler, LabelEncoder
#label_encoder = LabelEncoder()
#df['age'] = label_encoder.fit_transform(df['age'])
#df['sex'] = label_encoder.fit_transform(df['sex'])
#df['cp'] = label_encoder.fit_transform(df['cp'])
#df['trestbps'] = label_encoder.fit_transform(df['trestbps'])
#df['chol'] = label_encoder.fit_transform(df['chol'])
#df['fbs'] = label_encoder.fit_transform(df['fbs'])
#df['restecg'] = label_encoder.fit_transform(df['restecg'])
#df['thalach'] = label_encoder.fit_transform(df['thalach'])
#df['exang'] = label_encoder.fit_transform(df['exang'])
#df['oldpeak'] = label_encoder.fit_transform(df['oldpeak'])
#df['slope'] = label_encoder.fit_transform(df['slope'])
#df['ca'] = label_encoder.fit_transform(df['ca'])


# Preprocess the input data (e.g., one-hot encoding for categorical variables)
# You may need to apply the same preprocessing steps as done during model training

# Make predictions for HeartDisease column
heart_disease_predictions = model.predict(df)

# Print the predicted HeartDisease value (0 for No, 1 for Yes)
print("Predicted HeartDisease:", heart_disease_predictions)

print(y_pred[:50])


prediction_prob = model.predict_proba(df)
#(prediction_prob[0][1] * 100, 2)}%
print(prediction_prob)
print({round(prediction_prob[0][0] * 100, 2)})
