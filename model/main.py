import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
import pickle


def get_data_clean():
    """
    Load, clean, and preprocess the data.
    """
    data = pd.read_csv("datasets/Training Data.csv")

    # Clean the data
    conversion_rate = 0.011909906
    data['Income'] *= conversion_rate
    data = data.drop(['Id', 'CITY', 'STATE', 'CURRENT_HOUSE_YRS'], axis=1)

    # Encode object columns
    object_columns = data.select_dtypes(include=['object']).columns
    label_encoder = LabelEncoder()
    for column in object_columns:
        data[column] = label_encoder.fit_transform(data[column])

    # Split data into features and target
    X = data.drop("Risk_Flag", axis=1)
    y = data["Risk_Flag"]

    # Perform SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Reset the index
    data = pd.DataFrame(X_resampled, columns=X.columns)
    data['Risk_Flag'] = y_resampled

    return data


def create_model(data):
    """
    Train and evaluate a decision tree model.
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop("Risk_Flag", axis=1), data["Risk_Flag"], test_size=0.3, random_state=42
    )

    # Scale the data
    scaler = StandardScaler()
    X_train[:] = scaler.fit_transform(X_train[:])
    X_test[:] = scaler.transform(X_test[:])

    # Train the model
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    # Test the model
    y_pred = model.predict(X_test)

    # Model evaluation
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.2f}\n")
    print(f"Classification report: \n {classification_report(y_test, y_pred)}")

    return model, scaler


def main():
    """
    Train and save the model.
    """
    data = get_data_clean()
    model, scaler = create_model(data)

    with open('model/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)


if __name__ == '__main__':
    main()