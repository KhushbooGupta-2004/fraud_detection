import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_preprocess(data_path: str = None) -> tuple:
    if data_path is None:
        data_path = Path(__file__).parent.parent / "data" / "raw" / "indian_transactions.csv"
    
    df = pd.read_csv(data_path)

    # Feature engineering
    df['trans_datetime'] = pd.to_datetime(df['trans_datetime'])
    df['hour'] = df['trans_datetime'].dt.hour
    df['day_of_week'] = df['trans_datetime'].dt.dayofweek

    # Drop unnecessary columns
    drop_cols = ['trans_datetime', 'trans_date', 'trans_time', 'cc_num', 'cardholder_name',
                 'dob', 'upi_id', 'trans_num', 'fraud_reason']
    df = df.drop(columns=drop_cols)

    # One-hot encoding - categorical columns
    categorical_cols = ['gender', 'job', 'state', 'merchant', 'category', 'payment_method']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Split train and test data
    X = df.drop('is_fraud', axis=1)
    y = df['is_fraud'].astype(int)

    # Stratified train/test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1, stratify=y
    )

    # Scale features with StandardScaler (fit ONLY on train, transform test)
    sc = StandardScaler()
    X_train_scaled = pd.DataFrame(
        sc.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        sc.transform(X_test), columns=X_test.columns, index=X_test.index
    )

    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, sc


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, sc = load_and_preprocess()
    print(f"Training shape: {X_train_scaled.shape}")
    print(f"Testing shape: {X_test_scaled.shape}")
    print(f"Fraud rate in train: {y_train.mean():.4f}")
    print(f"Fraud rate in test: {y_test.mean():.4f}")
