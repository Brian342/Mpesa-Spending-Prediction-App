import pandas as pd
from sklearn.preprocessing import LabelEncoder


def encode_categorical_columns(df: pd.DataFrame, encoding_type: str = 'label') -> pd.DataFrame:
    df_encoding = df.copy()
    categorical_columns = df_encoding.select_dtypes(include=['object', 'category']).columns

    if encoding_type == 'label':
        for col in categorical_columns:
            le = LabelEncoder()
            df_encoding[col] = le.fit_transform(df_encoding[col])

    elif encoding_type == 'onehot':
        df_encoding = pd.get_dummies(df_encoding, columns=categorical_columns, drop_first=True)
    else:
        raise ValueError("Unsupported Label encoding_type. use label or onehot")

    return df_encoding


def encode_categorical_columns_training_encoder(df, label_encoder):
    encoded_columns = ['Transaction_type', 'Transaction_party', 'paid_in_or_Withdraw']
    for column in encoded_columns:
        if column in df.columns and column in label_encoder:
            le = label_encoder[column]
            def safe_transform(val):
                if val in le.classes_:
                    return le.transform([val])[0]
                else:
                    print(f"[WARN] Unknown value '{val}' in column '{column}', assigning -1")
                    return -1
            df[column] = df[column].apply(safe_transform)
        else:
            raise ValueError(f"Label encoder for column: {column} not found")
    return df

