import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import os
import yaml
import warnings

# Suppress specific sklearn warnings (e.g., about feature names)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Function to load the configuration from config.yaml


def load_config():
    """
    Loads the configuration from the config.yaml file.
    Assumes config.yaml is in the same directory as the script.
    """
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_script_dir, 'config.yaml')
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        print(f"Configuration loaded from: {config_path}")
        return config
    except FileNotFoundError:
        print(f"Error: config.yaml not found at {config_path}")
        print("Please ensure config.yaml is in the 'src/' directory or current working directory.")
        exit()
    except yaml.YAMLError as e:
        print(f"Error parsing config.yaml: {e}")
        exit()


def load_data(file_path):
    """
    Loads the dataset from a given CSV file path.
    """
    print(f"Loading data from: {file_path}")
    df = pd.read_csv(file_path)
    return df


def create_new_features(df):
    """
    Creates new engineered features from existing columns.
    """
    df_new_features = df.copy()
    print("\n--- Creating New Features ---")

    # Ensure necessary columns exist before creating features
    required_cols_for_new_features = [
        'n_tokens_content', 'num_hrefs', 'num_videos', 'num_imgs']
    for col in required_cols_for_new_features:
        if col not in df_new_features.columns:
            print(
                f"Warning: Required column '{col}' for new feature engineering not found. Skipping related feature creation.")
            return df_new_features  # Return original if a core column is missing

    # Add a small epsilon to denominators to prevent division by zero
    epsilon = 1e-6

    # Feature 1: Average words per link
    df_new_features['avg_words_per_link'] = df_new_features['n_tokens_content'] / \
        (df_new_features['num_hrefs'] + epsilon)
    print("Created 'avg_words_per_link'.")

    # Feature 2: Average words per video
    df_new_features['avg_words_per_video'] = df_new_features['n_tokens_content'] / \
        (df_new_features['num_videos'] + epsilon)
    print("Created 'avg_words_per_video'.")

    # Feature 3: Interaction between number of images and content tokens
    df_new_features['num_imgs_x_tokens'] = df_new_features['num_imgs'] * \
        df_new_features['n_tokens_content']
    print("Created 'num_imgs_x_tokens'.")

    print("--- New Features Created ---")
    return df_new_features


def transform_features(df, log_transform_cols):
    """
    Applies log transformations to specified numerical columns.
    """
    df_transformed = df.copy()
    for col in log_transform_cols:
        if col in df_transformed.columns:
            if pd.api.types.is_numeric_dtype(df_transformed[col]):
                # Ensure values are non-negative before log1p
                df_transformed[col] = np.log1p(
                    df_transformed[col].apply(lambda x: max(0, x)))
                print(f"Applied log transformation to '{col}'.")
            else:
                print(
                    f"Warning: Column '{col}' is not numerical, skipping log transformation.")
        else:
            print(f"Warning: Column '{col}' not found for log transformation.")
    return df_transformed


def handle_missing_values(df, numerical_imputation_cols):
    """
    Imputes missing values in numerical columns with the median and
    in 'data_channel' (categorical) with the mode.
    This function is primarily for *original* NaNs in the dataset.
    """
    df_imputed = df.copy()

    if 'data_channel' in df_imputed.columns:
        mode_data_channel = df_imputed['data_channel'].mode()[0]
        df_imputed['data_channel'] = df_imputed['data_channel'].fillna(
            mode_data_channel)
        print("Handled missing values in 'data_channel'.")
    else:
        print("Warning: 'data_channel' column not found for mode imputation.")

    for col in numerical_imputation_cols:
        if col in df_imputed.columns and df_imputed[col].isnull().any():
            median_val = df_imputed[col].median()
            df_imputed[col] = df_imputed[col].fillna(median_val)
            print(f"Handled missing values in numerical column '{col}'.")
        elif col not in df_imputed.columns:
            print(f"Warning: Numerical imputation column '{col}' not found.")

    print("Finished handling original missing values (if any).")
    return df_imputed


def handle_infinities_and_post_transform_nans(df, columns_to_check):
    """
    Handles infinity values and any NaNs that might have been introduced
    by transformations (like log1p on certain values) or new feature creation.
    It replaces inf/-inf with NaN, then imputes all NaNs using the median.
    """
    df_cleaned = df.copy()
    print("\n--- Handling Infinities and Post-Transform NaNs ---")

    # Replace inf/-inf directly with NaN first
    df_cleaned.replace([np.inf, -np.inf], np.nan, inplace=True)
    print("Converted all np.inf and -np.inf to np.nan.")

    imputer = SimpleImputer(strategy='median')

    # Filter columns to only include numerical ones that actually have NaNs in the current DataFrame
    numerical_cols_with_nans = [
        col for col in columns_to_check
        if col in df_cleaned.columns and pd.api.types.is_numeric_dtype(df_cleaned[col]) and df_cleaned[col].isnull().any()
    ]

    if not numerical_cols_with_nans:
        print("No (new) NaN values found in specified numerical columns after infinity conversion.")
        return df_cleaned

    print(
        f"Imputing NaN values (including former infinities) in: {numerical_cols_with_nans}")

    # Apply imputation only to the columns identified as having NaNs
    df_cleaned.loc[:, numerical_cols_with_nans] = imputer.fit_transform(
        df_cleaned[numerical_cols_with_nans]
    )

    print("Finished handling infinities and post-transform NaNs.")
    return df_cleaned


def encode_categorical_features(df, categorical_encoding_cols):
    """
    Performs one-hot encoding on specified categorical columns.
    """
    df_encoded = df.copy()

    actual_categorical_cols = [
        col for col in categorical_encoding_cols if col in df_encoded.columns and (pd.api.types.is_string_dtype(df_encoded[col]) or pd.api.types.is_categorical_dtype(df_encoded[col]))]

    if actual_categorical_cols:
        for col in actual_categorical_cols:
            df_encoded[col] = df_encoded[col].astype('category')

        df_encoded = pd.get_dummies(
            df_encoded, columns=actual_categorical_cols, drop_first=True)
        print(
            f"Applied one-hot encoding to: {', '.join(actual_categorical_cols)}")
    else:
        print("No specified categorical columns found for one-hot encoding.")
    return df_encoded


def split_data(df_processed, columns_to_drop, test_size, random_state):
    """
    Splits the processed DataFrame into training and testing sets for features (X) and target (y).
    """
    cols_to_drop_existing = [
        col for col in columns_to_drop if col in df_processed.columns]

    if 'shares' not in df_processed.columns:
        print("Error: 'shares' column (target) not found in DataFrame for splitting.")
        exit()
    y = df_processed['shares']
    X = df_processed.drop(columns=cols_to_drop_existing +
                          ['shares'], errors='ignore')

    print("Separated features (X) and target (y).")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)
    print(
        f"Split data into training and testing sets ({int(test_size*100)}/{int((1-test_size)*100)} split).")

    print(f"\nShape of X_train: {X_train.shape}")
    print(f"Shape of X_test: {X_test.shape}")
    print(f"Shape of y_train: {y_train.shape}")
    print(f"Shape of y_test: {y_test.shape}")

    # --- Diagnostic Check after Split ---
    print("\n--- Post-Split Data Sanity Checks (X_train, y_train) ---")
    if X_train.isnull().any().any():
        print(
            f"WARNING: X_train contains NaNs in columns: {X_train.columns[X_train.isnull().any()].tolist()}")
    if np.any(np.isinf(X_train)):
        print(
            f"WARNING: X_train contains Infs in columns: {X_train.columns[np.isinf(X_train).any()].tolist()}")
    if y_train.isnull().any():
        print("WARNING: y_train contains NaNs!")
    if np.any(np.isinf(y_train)):
        print("WARNING: y_train contains Infs!")
    print("-----------------------------------------------------")

    return X_train, X_test, y_train, y_test


def save_prepared_data(X_train, X_test, y_train, y_test, output_dir):
    """
    Saves the processed data (X_train, X_test, y_train, y_test) into .npy format
    in the specified output directory, ensuring they are saved as float arrays.
    """
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, 'X_train.npy'),
            X_train.astype(np.float32).values)
    np.save(os.path.join(output_dir, 'X_test.npy'),
            X_test.astype(np.float32).values)
    np.save(os.path.join(output_dir, 'y_train.npy'),
            y_train.astype(np.float32).values)
    np.save(os.path.join(output_dir, 'y_test.npy'),
            y_test.astype(np.float32).values)

    print(f"Prepared data saved to {output_dir} in .npy format.")


def main():
    """
    Orchestrates the data preparation process: loading, transforming,
    handling missing values, encoding, splitting, and saving, using config.
    """
    config = load_config()

    paths_config = config['paths']
    current_script_dir = os.path.dirname(os.path.abspath(__file__))

    data_dir = os.path.normpath(os.path.join(
        current_script_dir, paths_config['raw_data_directory']))
    output_dir = os.path.normpath(os.path.join(
        current_script_dir, paths_config['prepared_data_directory']))
    full_data_path = os.path.join(data_dir, paths_config['raw_data_filename'])

    data_prep_config = config['data_preparation']
    log_transform_cols = data_prep_config['log_transform_columns']
    numerical_imputation_cols = data_prep_config['numerical_imputation_columns']
    categorical_encoding_cols = data_prep_config['categorical_encoding_columns']
    columns_to_drop_from_features = data_prep_config['columns_to_drop_from_features']

    model_training_config = config['model_training']
    test_size = model_training_config['test_size']
    random_state = model_training_config['random_state']

    df = load_data(full_data_path)

    # --- New Step: Create new features ---
    df_with_new_features = create_new_features(df)

    df_initial_imputed = handle_missing_values(
        df_with_new_features, numerical_imputation_cols)

    df_transformed = transform_features(df_initial_imputed, log_transform_cols)

    all_numerical_cols_to_clean = df_transformed.select_dtypes(
        include=np.number).columns.tolist()
    if 'shares' in all_numerical_cols_to_clean:
        # Remove target from imputation if it's not a feature
        all_numerical_cols_to_clean.remove('shares')

    df_cleaned_infinities = handle_infinities_and_post_transform_nans(
        df_transformed, all_numerical_cols_to_clean)

    df_encoded = encode_categorical_features(
        df_cleaned_infinities, categorical_encoding_cols)

    X_train, X_test, y_train, y_test = split_data(
        df_encoded, columns_to_drop_from_features, test_size, random_state)

    save_prepared_data(X_train, X_test, y_train, y_test, output_dir)


if __name__ == '__main__':
    print("--- Starting the ML Pipeline (Data Preparation) ---")
    main()
    print("Data Preparation completed successfully.")
