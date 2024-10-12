import pandas as pd 
import numpy as np 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE

import pickle


def get_data_clean():

    data  = pd.read_csv("datasets/Training Data.csv")
    # Clean the data
    conversion_rate = 0.011909906
    data['Income'] = data['Income'] * conversion_rate
    data = data.drop(['Id', 'CITY', 'STATE', 'CURRENT_HOUSE_YRS'], axis=1)
    
    label_encoder = LabelEncoder()
    object_columns = data.select_dtypes(include=['object']).columns
    for column in object_columns:
        data[column] = label_encoder.fit_transform (data[column])
    
    y = data["Risk_Flag"]
    X = data.drop(["Risk_Flag"], axis=1)
        
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
        
    data = pd.DataFrame(X_resampled, columns=X.columns)
    data['Risk_Flag'] = y_resampled
    
    return data

def create_model(data):
    
    y = data["Risk_Flag"]
    X = data.drop(["Risk_Flag"], axis=1)
    
    # Scale the data
    scaler = StandardScaler()
    X[:] = scaler.fit_transform(X[:])
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Train the model with Logistic Regression
    model = DecisionTreeClassifier()
    model.fit(X_train,y_train)
    
    # Test the model
    y_pred = model.predict(X_test)
    # Model Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy: .2f}\n")
    print(f"Classification report: \n {classification_report(y_pred, y_test)}")
    return model, scaler
    
def main():
    data = get_data_clean()
    model, scaler = create_model(data)
    
    with open('model/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
        
if __name__ == '__main__':
    main()