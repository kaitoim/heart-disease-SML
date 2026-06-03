import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def load_data(path):
    try:
        df = pd.read_csv(path)
        print(f"Data berhasil dimuat dari {path}")
        return df
    except FileNotFoundError:
        print(f"Error: File tidak ditemukan di {path}")
        return None


def preprocess_data(df):

    df_clean = df.copy()

    # Ubah target menjadi biner
    df_clean["target"] = df_clean["target"].apply(
        lambda x: 1 if x > 0 else 0
    )

    # Pisahkan fitur dan target
    X = df_clean.drop(columns=["target"])
    y = df_clean["target"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Kolom kategorikal pada dataset heart disease
    categorical_features = [
        "sex",
        "cp",
        "fbs",
        "restecg",
        "exang",
        "slope",
        "ca",
        "thal"
    ]

    # Ambil otomatis kolom numerik
    numeric_features = [
        col for col in X.columns
        if col not in categorical_features
    ]

    # Pipeline numerik
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    # Pipeline kategorikal
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    # Gabungkan preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    # Transform data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    print("\nKolom Numerik :", numeric_features)
    print("Kolom Kategorikal :", categorical_features)

    return (
        X_train_processed,
        X_test_processed,
        y_train,
        y_test,
        preprocessor
    )


if __name__ == "__main__":
    import os

    current_script_dir = os.path.dirname(
        os.path.abspath(__file__)
    )

    data_path = os.path.join(
        current_script_dir,
        "..",
        "heart_raw.csv"
    )

    print(
        f"Mencoba mencari file di: "
        f"{os.path.normpath(data_path)}"
    )

    df = load_data(data_path)

    if df is not None:
        X_train, X_test, y_train, y_test, preprocessor = (
            preprocess_data(df)
        )

        print("\nTest Run Berhasil!")
        print(f"Shape X_train : {X_train.shape}")
        print(f"Shape X_test  : {X_test.shape}")
        print(f"Jumlah Train  : {len(y_train)}")
        print(f"Jumlah Test   : {len(y_test)}")