import pytest
import pandas as pd
import numpy as np
import os

# Now, you can directly import from data_preparation as if it were a top-level module
# Make sure handle_infinities_and_post_transform_nans is also imported
from data_preparation import (
    load_data,
    transform_features,
    handle_missing_values,
    handle_infinities_and_post_transform_nans,  # NEW IMPORT
    encode_categorical_features,
    split_data,
    save_prepared_data
)

# Fixture to create a dummy config.yaml for testing


@pytest.fixture
def mock_config(tmp_path):
    """Creates a temporary config.yaml for testing."""
    config_content = """
    paths:
      raw_data_directory: "../data" # Relative to src, but for test, data will be in tmp_path
      raw_data_filename: "test_data.csv"
      prepared_data_directory: "prepared_data" # Relative to src for actual code, but mock will save in tmp_path
      models_directory: "models"
    data_preparation:
      log_transform_columns:
        - shares
        - n_comments
        - kw_min_min
        - kw_max_min
        - kw_avg_min
        - kw_min_max
        - kw_max_max
        - kw_avg_max
        - kw_min_avg
        - kw_max_avg
        - kw_avg_avg
      numerical_imputation_columns:
        - num_hrefs
        - num_self_hrefs
        - num_imgs
        - num_videos
        - self_reference_min_shares
        - self_reference_max_shares
        - self_reference_avg_shares
        - kw_min_min # Include log-transformed columns here too for cleanup check
        - kw_max_min
        - kw_avg_min
        - kw_min_max
        - kw_max_max
        - kw_avg_max
        - kw_min_avg
        - kw_max_avg
        - kw_avg_avg
      categorical_encoding_columns:
        - weekday
        - data_channel
      columns_to_drop_from_features:
        - ID
        - URL
        - shares
    model_training:
      test_size: 0.2
      random_state: 42
      models:
        LinearRegression: {}
    """
    # Create the 'src' directory within the temporary path to mimic project structure
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    config_file = src_dir / "config.yaml"
    config_file.write_text(config_content)
    return config_file


# Fixture to create a dummy CSV file for testing
@pytest.fixture
def dummy_csv_file(tmp_path):
    """Creates a temporary CSV file for testing load_data."""
    # Create a dummy 'data' directory for the test data
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    file_path = data_dir / "test_data.csv"

    # Create content with relevant columns for your data_preparation functions
    data = {
        'ID': [1, 2, 3, 4],
        'URL': ['http://a.com', 'http://b.com', 'http://c.com', 'http://d.com'],
        'timedelta': [100, 200, 300, 400],
        'shares': [10, 0, 100, 500],  # Shares can be 0 -> log1p(0)=0.0
        'n_comments': [5, 0, 20, 100],  # n_comments can be 0 -> log1p(0)=0.0
        'num_hrefs': [1, np.nan, 3, 4],    # For numerical imputation test
        'num_self_hrefs': [1, 2, np.nan, 4],  # Added for numerical imputation
        'num_imgs': [1, np.nan, 3, 4],  # Added for numerical imputation
        'num_videos': [1, 2, 3, np.nan],  # Added for numerical imputation
        # Added for numerical imputation
        'self_reference_min_shares': [10, 20, np.nan, 40],
        # Added for numerical imputation
        'self_reference_max_shares': [100, 200, 300, np.nan],
        # Added for numerical imputation
        'self_reference_avg_shares': [50, 100, np.nan, 200],
        # For categorical and imputation test
        'data_channel': ['news', 'tech', 'news', np.nan],
        # For categorical test
        'weekday': ['monday', 'tuesday', 'monday', 'wednesday'],
        # Added other kw columns for log_transform_columns to match config better
        'kw_min_min': [1, 0, 2, 3],
        'kw_max_min': [10, 5, 20, 30],
        'kw_avg_min': [5, 2.5, 11, 16],
        'kw_min_max': [100, 50, 200, 300],
        'kw_max_max': [1000, 500, 2000, 3000],
        'kw_avg_max': [500, 250, 1100, 1600],
        'kw_min_avg': [1, 0.5, 2, 3],
        'kw_max_avg': [10, 5, 20, 30],
        'kw_avg_avg': [5, 2.5, 11, 16]
    }
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)
    return file_path

# Test for load_data function


def test_load_data(dummy_csv_file):
    df = load_data(dummy_csv_file)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert 'shares' in df.columns
    assert len(df) == 4  # Check if all rows are loaded

# Test for transform_features function


def test_transform_features():
    df = pd.DataFrame({'shares': [0, 1, 99], 'n_comments': [0, 10, 999]})
    # Ensure shares and n_comments are numeric before transformation
    df['shares'] = pd.to_numeric(df['shares'])
    df['n_comments'] = pd.to_numeric(df['n_comments'])

    log_cols = ['shares', 'n_comments']
    transformed_df = transform_features(df, log_cols)

    # Check if transformation applied correctly using log1p
    assert np.isclose(transformed_df['shares'].iloc[0], np.log1p(0))
    assert np.isclose(transformed_df['shares'].iloc[1], np.log1p(1))
    assert np.isclose(transformed_df['shares'].iloc[2], np.log1p(99))

    assert np.isclose(transformed_df['n_comments'].iloc[0], np.log1p(0))
    assert np.isclose(transformed_df['n_comments'].iloc[1], np.log1p(10))
    assert np.isclose(transformed_df['n_comments'].iloc[2], np.log1p(999))

# Test for handle_missing_values (original NaNs)


def test_handle_missing_values():
    df = pd.DataFrame({
        'num_hrefs': [1.0, np.nan, 3.0],
        'data_channel': ['news', np.nan, 'tech'],
        # Add 'shares' to avoid potential issues with other functions
        'shares': [10, 20, 30]
    })
    # Ensure num_hrefs is float before imputation
    df['num_hrefs'] = df['num_hrefs'].astype(float)

    numerical_impute = ['num_hrefs']
    imputed_df = handle_missing_values(df, numerical_impute)

    # Median of [1.0, 3.0] is 2.0
    assert imputed_df['num_hrefs'].iloc[1] == 2.0
    # Mode of ['news', 'tech'] is 'news'
    assert imputed_df['data_channel'].iloc[1] == 'news'

# NEW TEST: Test for handle_infinities_and_post_transform_nans


def test_handle_infinities_and_post_transform_nans():
    # Create a DataFrame with values that would produce -inf or NaN after log1p,
    # and some original NaNs.
    df = pd.DataFrame({
        'feature_log': [0, 1, -1, 10, np.nan],  # -1 would give NaN after log1p
        'feature_impute': [10, np.nan, 20, 30, 40],
        # Test positive infinity (unlikely from log1p but possible generally)
        'feature_inf_pos': [1, np.inf, 2, 3, 4]
    })

    # Simulate log1p on 'feature_log'
    df['feature_log'] = np.log1p(df['feature_log'])

    # Columns that should be checked for NaNs/Infs
    cols_to_check = ['feature_log', 'feature_impute', 'feature_inf_pos']

    cleaned_df = handle_infinities_and_post_transform_nans(
        df.copy(), cols_to_check)

    # After cleaning, there should be no inf or NaN values
    assert not cleaned_df['feature_log'].isnull().any()
    assert not np.isinf(cleaned_df['feature_log']).any()

    assert not cleaned_df['feature_impute'].isnull().any()
    assert not np.isinf(cleaned_df['feature_impute']).any()

    assert not cleaned_df['feature_inf_pos'].isnull().any()
    assert not np.isinf(cleaned_df['feature_inf_pos']).any()

    # Verify imputation for feature_log: log1p(-1) is NaN, median of [0.0, log1p(1), log1p(10)]
    # For 'feature_log', if 0, 1, 10 are the values, after log1p they are 0, ln(2), ln(11).
    # The median would be ln(2) = 0.693.
    # If -1 results in NaN, and original 0, 1, 10 are the other values.
    # Let's consider the specific values for median calculation.
    # Assuming original values in 'feature_log' that become finite after log1p are [0, 1, 10], and NaN.
    # Logged values: [0.0, 0.693147, 2.397895]
    # Median of these is 0.693147. So, the NaN from -1 should be imputed with this.
    simulated_log_values = np.array([np.log1p(0), np.log1p(
        1), np.log1p(10), np.log1p(-1), np.log1p(np.nan)])
    # The -1 will create NaN, and the original NaN will remain NaN.
    # The values used for median are from the *finite* values: [np.log1p(0), np.log1p(1), np.log1p(10)]
    expected_median_log = pd.Series(
        [np.log1p(0), np.log1p(1), np.log1p(10)]).median()
    assert np.isclose(cleaned_df['feature_log'].iloc[2],
                      expected_median_log)  # The imputed -1
    assert np.isclose(cleaned_df['feature_log'].iloc[4],
                      expected_median_log)  # The imputed np.nan

    # Verify imputation for feature_impute: median of [10, 20, 30, 40] is 25.0
    assert cleaned_df['feature_impute'].iloc[1] == 25.0

    # Verify imputation for feature_inf_pos: median of [1, 2, 3, 4] is 2.5
    # The imputed inf
    assert np.isclose(cleaned_df['feature_inf_pos'].iloc[1], 2.5)

# Test for encode_categorical_features


def test_encode_categorical_features():
    df = pd.DataFrame({
        'weekday': ['monday', 'tuesday', 'monday'],
        'data_channel': ['tech', 'news', 'tech']  # 'news' and 'tech'
    })
    cat_cols = ['weekday', 'data_channel']
    encoded_df = encode_categorical_features(df, cat_cols)

    # With drop_first=True, for 'weekday', 'monday' would be base, 'tuesday' created.
    # For 'data_channel', 'news' (alphabetically first) would be base, 'tech' created.

    assert 'weekday_tuesday' in encoded_df.columns
    assert 'data_channel_tech' in encoded_df.columns  # Corrected as 'news' is dropped

    assert encoded_df['weekday_tuesday'].iloc[0] == 0  # 'monday'
    assert encoded_df['weekday_tuesday'].iloc[1] == 1  # 'tuesday'

    assert encoded_df['data_channel_tech'].iloc[0] == 1  # 'tech'
    # 'news' (original value, now represented by 0 in 'data_channel_tech')
    assert encoded_df['data_channel_tech'].iloc[1] == 0
    assert encoded_df['data_channel_tech'].iloc[2] == 1  # 'tech'


# Test for split_data
def test_split_data():
    df = pd.DataFrame({
        'ID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'URL': ['u1', 'u2', 'u3', 'u4', 'u5', 'u6', 'u7', 'u8', 'u9', 'u10'],
        'feature1': np.random.rand(10),
        'feature2': np.random.rand(10),
        'shares': np.random.rand(10)
    })

    # Ensure shares is numeric
    df['shares'] = pd.to_numeric(df['shares'])

    drop_cols = ['ID', 'URL', 'shares']
    X_train, X_test, y_train, y_test = split_data(
        df, drop_cols, test_size=0.2, random_state=42)

    # Check shapes
    assert X_train.shape[0] == 8
    assert X_test.shape[0] == 2
    assert y_train.shape[0] == 8
    assert y_test.shape[0] == 2

    # Check if dropped columns are not in X_train/X_test
    assert not any(col in X_train.columns for col in ['ID', 'URL', 'shares'])


# Test for save_prepared_data (checks if files are created)
def test_save_prepared_data(tmp_path):
    X_train_data = np.array([[1, 2], [3, 4]], dtype=np.float32)
    X_test_data = np.array([[5, 6]], dtype=np.float32)
    y_train_data = np.array([10, 20], dtype=np.float32)
    y_test_data = np.array([30], dtype=np.float32)

    # Convert NumPy arrays back to DataFrame/Series for the function's expected input type
    save_prepared_data(
        # Add column names for DataFrame
        pd.DataFrame(X_train_data, columns=['col1', 'col2']),
        pd.DataFrame(X_test_data, columns=['col1', 'col2']),
        pd.Series(y_train_data, name='shares'),  # Add name for Series
        pd.Series(y_test_data, name='shares'),
        tmp_path
    )

    assert os.path.exists(tmp_path / 'X_train.npy')
    assert os.path.exists(tmp_path / 'X_test.npy')
    assert os.path.exists(tmp_path / 'y_train.npy')  # Corrected line
    assert os.path.exists(tmp_path / 'y_test.npy')

    # Verify content
    loaded_X_train = np.load(tmp_path / 'X_train.npy')
    assert np.array_equal(loaded_X_train, X_train_data)


# Now, you can directly import from data_preparation as if it were a top-level module
# Make sure handle_infinities_and_post_transform_nans is also imported

# Fixture to create a dummy config.yaml for testing


@pytest.fixture
def mock_config(tmp_path):
    """Creates a temporary config.yaml for testing."""
    config_content = """
    paths:
      raw_data_directory: "../data" # Relative to src, but for test, data will be in tmp_path
      raw_data_filename: "test_data.csv"
      prepared_data_directory: "prepared_data" # Relative to src for actual code, but mock will save in tmp_path
      models_directory: "models"
    data_preparation:
      log_transform_columns:
        - shares
        - n_comments
        - kw_min_min
        - kw_max_min
        - kw_avg_min
        - kw_min_max
        - kw_max_max
        - kw_avg_max
        - kw_min_avg
        - kw_max_avg
        - kw_avg_avg
      numerical_imputation_columns:
        - num_hrefs
        - num_self_hrefs
        - num_imgs
        - num_videos
        - self_reference_min_shares
        - self_reference_max_shares
        - self_reference_avg_shares
        - kw_min_min # Include log-transformed columns here too for cleanup check
        - kw_max_min
        - kw_avg_min
        - kw_min_max
        - kw_max_max
        - kw_avg_max
        - kw_min_avg
        - kw_max_avg
        - kw_avg_avg
      categorical_encoding_columns:
        - weekday
        - data_channel
      columns_to_drop_from_features:
        - ID
        - URL
        - shares
    model_training:
      test_size: 0.2
      random_state: 42
      models:
        LinearRegression: {}
    """
    # Create the 'src' directory within the temporary path to mimic project structure
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    config_file = src_dir / "config.yaml"
    config_file.write_text(config_content)
    return config_file


# Fixture to create a dummy CSV file for testing
@pytest.fixture
def dummy_csv_file(tmp_path):
    """Creates a temporary CSV file for testing load_data."""
    # Create a dummy 'data' directory for the test data
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    file_path = data_dir / "test_data.csv"

    # Create content with relevant columns for your data_preparation functions
    data = {
        'ID': [1, 2, 3, 4],
        'URL': ['http://a.com', 'http://b.com', 'http://c.com', 'http://d.com'],
        'timedelta': [100, 200, 300, 400],
        'shares': [10, 0, 100, 500],  # Shares can be 0 -> log1p(0)=0.0
        'n_comments': [5, 0, 20, 100],  # n_comments can be 0 -> log1p(0)=0.0
        'num_hrefs': [1, np.nan, 3, 4],    # For numerical imputation test
        'num_self_hrefs': [1, 2, np.nan, 4],  # Added for numerical imputation
        'num_imgs': [1, np.nan, 3, 4],  # Added for numerical imputation
        'num_videos': [1, 2, 3, np.nan],  # Added for numerical imputation
        # Added for numerical imputation
        'self_reference_min_shares': [10, 20, np.nan, 40],
        # Added for numerical imputation
        'self_reference_max_shares': [100, 200, 300, np.nan],
        # Added for numerical imputation
        'self_reference_avg_shares': [50, 100, np.nan, 200],
        # For categorical and imputation test
        'data_channel': ['news', 'tech', 'news', np.nan],
        # For categorical test
        'weekday': ['monday', 'tuesday', 'monday', 'wednesday'],
        # Added other kw columns for log_transform_columns to match config better
        'kw_min_min': [1, 0, 2, 3],
        'kw_max_min': [10, 5, 20, 30],
        'kw_avg_min': [5, 2.5, 11, 16],
        'kw_min_max': [100, 50, 200, 300],
        'kw_max_max': [1000, 500, 2000, 3000],
        'kw_avg_max': [500, 250, 1100, 1600],
        'kw_min_avg': [1, 0.5, 2, 3],
        'kw_max_avg': [10, 5, 20, 30],
        'kw_avg_avg': [5, 2.5, 11, 16]
    }
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)
    return file_path

# Test for load_data function


def test_load_data(dummy_csv_file):
    df = load_data(dummy_csv_file)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert 'shares' in df.columns
    assert len(df) == 4  # Check if all rows are loaded

# Test for transform_features function


def test_transform_features():
    df = pd.DataFrame({'shares': [0, 1, 99], 'n_comments': [0, 10, 999]})
    # Ensure shares and n_comments are numeric before transformation
    df['shares'] = pd.to_numeric(df['shares'])
    df['n_comments'] = pd.to_numeric(df['n_comments'])

    log_cols = ['shares', 'n_comments']
    transformed_df = transform_features(df, log_cols)

    # Check if transformation applied correctly using log1p
    assert np.isclose(transformed_df['shares'].iloc[0], np.log1p(0))
    assert np.isclose(transformed_df['shares'].iloc[1], np.log1p(1))
    assert np.isclose(transformed_df['shares'].iloc[2], np.log1p(99))

    assert np.isclose(transformed_df['n_comments'].iloc[0], np.log1p(0))
    assert np.isclose(transformed_df['n_comments'].iloc[1], np.log1p(10))
    assert np.isclose(transformed_df['n_comments'].iloc[2], np.log1p(999))

# Test for handle_missing_values (original NaNs)


def test_handle_missing_values():
    df = pd.DataFrame({
        'num_hrefs': [1.0, np.nan, 3.0],
        'data_channel': ['news', np.nan, 'tech'],
        # Add 'shares' to avoid potential issues with other functions
        'shares': [10, 20, 30]
    })
    # Ensure num_hrefs is float before imputation
    df['num_hrefs'] = df['num_hrefs'].astype(float)

    numerical_impute = ['num_hrefs']
    imputed_df = handle_missing_values(df, numerical_impute)

    # Median of [1.0, 3.0] is 2.0
    assert imputed_df['num_hrefs'].iloc[1] == 2.0
    # Mode of ['news', 'tech'] is 'news'
    assert imputed_df['data_channel'].iloc[1] == 'news'

# NEW TEST: Test for handle_infinities_and_post_transform_nans


def test_handle_infinities_and_post_transform_nans():
    # Create a DataFrame with values that would produce -inf or NaN after log1p,
    # and some original NaNs.
    df = pd.DataFrame({
        'feature_log': [0, 1, -1, 10, np.nan],  # -1 would give NaN after log1p
        'feature_impute': [10, np.nan, 20, 30, 40],
        # Test positive infinity (unlikely from log1p but possible generally)
        'feature_inf_pos': [1, np.inf, 2, 3, 4]
    })

    # Simulate log1p on 'feature_log'
    df['feature_log'] = np.log1p(df['feature_log'])

    # Columns that should be checked for NaNs/Infs
    cols_to_check = ['feature_log', 'feature_impute', 'feature_inf_pos']

    cleaned_df = handle_infinities_and_post_transform_nans(
        df.copy(), cols_to_check)

    # After cleaning, there should be no inf or NaN values
    assert not cleaned_df['feature_log'].isnull().any()
    assert not np.isinf(cleaned_df['feature_log']).any()

    assert not cleaned_df['feature_impute'].isnull().any()
    assert not np.isinf(cleaned_df['feature_impute']).any()

    assert not cleaned_df['feature_inf_pos'].isnull().any()
    assert not np.isinf(cleaned_df['feature_inf_pos']).any()

    # Verify imputation for feature_log: log1p(-1) is NaN, median of [0.0, log1p(1), log1p(10)]
    # For 'feature_log', if 0, 1, 10 are the values, after log1p they are 0, ln(2), ln(11).
    # The median would be ln(2) = 0.693.
    # If -1 results in NaN, and original 0, 1, 10 are the other values.
    # Let's consider the specific values for median calculation.
    # Assuming original values in 'feature_log' that become finite after log1p are [0, 1, 10], and NaN.
    # Logged values: [0.0, 0.693147, 2.397895]
    # Median of these is 0.693147. So, the NaN from -1 should be imputed with this.
    simulated_log_values = np.array([np.log1p(0), np.log1p(
        1), np.log1p(10), np.log1p(-1), np.log1p(np.nan)])
    # The -1 will create NaN, and the original NaN will remain NaN.
    # The values used for median are from the *finite* values: [np.log1p(0), np.log1p(1), np.log1p(10)]
    expected_median_log = pd.Series(
        [np.log1p(0), np.log1p(1), np.log1p(10)]).median()
    assert np.isclose(cleaned_df['feature_log'].iloc[2],
                      expected_median_log)  # The imputed -1
    assert np.isclose(cleaned_df['feature_log'].iloc[4],
                      expected_median_log)  # The imputed np.nan

    # Verify imputation for feature_impute: median of [10, 20, 30, 40] is 25.0
    assert cleaned_df['feature_impute'].iloc[1] == 25.0

    # Verify imputation for feature_inf_pos: median of [1, 2, 3, 4] is 2.5
    # The imputed inf
    assert np.isclose(cleaned_df['feature_inf_pos'].iloc[1], 2.5)

# Test for encode_categorical_features


def test_encode_categorical_features():
    df = pd.DataFrame({
        'weekday': ['monday', 'tuesday', 'monday'],
        'data_channel': ['tech', 'news', 'tech']  # 'news' and 'tech'
    })
    cat_cols = ['weekday', 'data_channel']
    encoded_df = encode_categorical_features(df, cat_cols)

    # With drop_first=True, for 'weekday', 'monday' would be base, 'tuesday' created.
    # For 'data_channel', 'news' (alphabetically first) would be base, 'tech' created.

    assert 'weekday_tuesday' in encoded_df.columns
    assert 'data_channel_tech' in encoded_df.columns  # Corrected as 'news' is dropped

    assert encoded_df['weekday_tuesday'].iloc[0] == 0  # 'monday'
    assert encoded_df['weekday_tuesday'].iloc[1] == 1  # 'tuesday'

    assert encoded_df['data_channel_tech'].iloc[0] == 1  # 'tech'
    # 'news' (original value, now represented by 0 in 'data_channel_tech')
    assert encoded_df['data_channel_tech'].iloc[1] == 0
    assert encoded_df['data_channel_tech'].iloc[2] == 1  # 'tech'


# Test for split_data
def test_split_data():
    df = pd.DataFrame({
        'ID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'URL': ['u1', 'u2', 'u3', 'u4', 'u5', 'u6', 'u7', 'u8', 'u9', 'u10'],
        'feature1': np.random.rand(10),
        'feature2': np.random.rand(10),
        'shares': np.random.rand(10)
    })

    # Ensure shares is numeric
    df['shares'] = pd.to_numeric(df['shares'])

    drop_cols = ['ID', 'URL', 'shares']
    X_train, X_test, y_train, y_test = split_data(
        df, drop_cols, test_size=0.2, random_state=42)

    # Check shapes
    assert X_train.shape[0] == 8
    assert X_test.shape[0] == 2
    assert y_train.shape[0] == 8
    assert y_test.shape[0] == 2

    # Check if dropped columns are not in X_train/X_test
    assert not any(col in X_train.columns for col in ['ID', 'URL', 'shares'])


# Test for save_prepared_data (checks if files are created)
def test_save_prepared_data(tmp_path):
    X_train_data = np.array([[1, 2], [3, 4]], dtype=np.float32)
    X_test_data = np.array([[5, 6]], dtype=np.float32)
    y_train_data = np.array([10, 20], dtype=np.float32)
    y_test_data = np.array([30], dtype=np.float32)

    # Convert NumPy arrays back to DataFrame/Series for the function's expected input type
    save_prepared_data(
        # Add column names for DataFrame
        pd.DataFrame(X_train_data, columns=['col1', 'col2']),
        pd.DataFrame(X_test_data, columns=['col1', 'col2']),
        pd.Series(y_train_data, name='shares'),  # Add name for Series
        pd.Series(y_test_data, name='shares'),
        tmp_path
    )

    assert os.path.exists(tmp_path / 'X_train.npy')
    assert os.path.exists(tmp_path / 'X_test.npy')
    assert os.path.exists(tmp_path / 'y_train.npy')  # Corrected line
    assert os.path.exists(tmp_path / 'y_test.npy')

    # Verify content
    loaded_X_train = np.load(tmp_path / 'X_train.npy')
    assert np.array_equal(loaded_X_train, X_train_data)
