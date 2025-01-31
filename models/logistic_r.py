import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

def prepare_data():
    # Read the CSV file
    df = pd.read_csv('data/train.csv')
    df2 = pd.read_csv('data/test.csv')
    # Separate features (X) and target variable (y)
    X_train = df.drop('target', axis=1)
    y_train= df['target']
    X_test = df2.drop('target',axis=1)
    y_test = df2['target']
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    features = X_train.columns
    # save the scaler
    pickle.dump(scaler, open('models/scaler.pkl', 'wb'))
    return X_train_scaled, y_train, X_test_scaled, y_test, features

def train_model(X_train_scaled, y_train):
    # Create and train the logistic regression model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    # Save the trained model
    model_filename = 'models/logistic_model.pkl'
    with open(model_filename, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model saved to {model_filename}")
    return model_filename

def load_model(model_filename):
    with open(model_filename, 'rb') as file:
        model = pickle.load(file)
        # load the scaler
    scaler = pickle.load(open('models/scaler.pkl', 'rb'))
    return model, scaler

def get_user_input():
    """Collect user input for cardiovascular disease prediction."""
    print("\nPlease enter the following information:")
    
    attributes = {
        'age': float,
        'gender': int,  # (0 = female, 1 = male)
        'chestpain': int,  # (1-4)
        'restingBP': float,
        'serumcholestrol': float,
        'fastingbloodsugar': int,  # (0 = false, 1 = true)
        'restingrelectro': int,  # (0-2)
        'maxheartrate': float,
        'exerciseangia': int,  # (0 = no, 1 = yes)
        'oldpeak': float,
        'slope': int,  # (1-3)
        'noofmajorvessels': int  # (0-3)
    }
    
    user_values = {}
    
    for attribute, attr_type in attributes.items():
        while True:
            try:
                value = input(f"Enter {attribute}: ")
                user_values[attribute] = attr_type(value)
                break
            except ValueError:
                print(f"Invalid input. Please enter a valid {attr_type.__name__} value.")
    
    # Convert to pandas DataFrame for prediction
    return pd.DataFrame([user_values])

def predict_logistic(feature_vector, model_filename):
    # feature vector for an individual consists of 12 features
    # feature vector is of Pandas series type
    # load the model and the scaler
    model, scaler = load_model(model_filename)
    # transform the test dataset
    X_test_scaled = scaler.transform(feature_vector)
    # make predictions on the test set
    yhat = model.predict(X_test_scaled)
    if yhat == 0.0:
        print("Patient has no cardiovascular disease")
        return 0.0
    elif yhat == 1.0:
        print("Patient has cardiovascular disease")
        return 1.0
    else:
        print ('Cannot infer diagnosis. Double check your input data.')
        return None

def evaluate_model(model, X_test_scaled, y_test, features):
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    # Print model performance
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    # Print feature importance
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': abs(model.coef_[0])
    })
    print("\nFeature Importance:")
    print(feature_importance.sort_values(by='Importance', ascending=False))

if __name__ == '__main__':
    X_train_scaled, y_train, X_test_scaled, y_test, features = prepare_data()
    model_filename = train_model(X_train_scaled, y_train)
    model, scaler = load_model(model_filename)
    evaluate_model(model, X_test_scaled, y_test, features)
    user_data = get_user_input()
    prediction = predict_logistic(user_data, model_filename)