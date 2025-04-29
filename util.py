import pandas as pd
from sklearn.preprocessing import LabelEncoder


def encode_categorical_columns(df: pd.DataFrame, encoding_type: str = 'label') -> pd.DataFrame:
    df_encoding = df.copy()
    categorical_columns = df_encoding.select_dtypes(include='category').columns

    if encoding_type == 'label':
        for col in categorical_columns:
            le = LabelEncoder()
            df_encoding[col] = le.fit_transform(df_encoding[col])

    elif encoding_type == 'onehot':
        df_encoding = pd.get_dummies(df_encoding, columns=categorical_columns, drop_first=True)
    else:
        raise ValueError("Unsupported Label encoding_type. use label or onehot")

    return df_encoding
