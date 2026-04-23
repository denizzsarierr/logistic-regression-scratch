import pandas as pd
from sklearn.preprocessing import StandardScaler

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()

    def clean_data(self, df):
        df = df.copy()

        # Drop ID column
        df.drop('customerID', axis=1, inplace=True)

        # Binary encoding
        binary_cols = ['Partner','Dependents','PhoneService','PaperlessBilling','Churn']
        for col in binary_cols:
            df[col] = df[col].map({'Yes': 1, 'No': 0})

        df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})

        # Service columns
        service_cols = ['MultipleLines','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']

        for col in service_cols:
            df[col] = df[col].replace({"No internet service": "No", "No phone service": "No"})
            df[col] = df[col].map({'Yes': 1, 'No': 0})

        # One-hot encoding
        df = pd.get_dummies(df,columns=['PaymentMethod','Contract','InternetService'],drop_first=True)

        # Fix numeric columns
        df["MonthlyCharges"] = df["MonthlyCharges"].astype(str).str.replace(",", ".")
        df["TotalCharges"] = df["TotalCharges"].astype(str).str.replace(",", ".")

        df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

        # Missing values handling
        df.loc[df["TotalCharges"].isna(), "TotalCharges"] = df["MonthlyCharges"] * df["tenure"]

        return df

    def scale_features(self, X_train, X_test):
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled