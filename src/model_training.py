import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV
import os
import joblib
import warnings
import yaml

# Add a very early debug print statement
print("DEBUG: model_training.py script initialized and imports loaded.")

# Suppress specific sklearn warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


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


def load_prepared_data(data_dir):
    """
    Loads the prepared data (X_train, X_test, y_train, y_test) from .npy files.
    """
    print(f"Attempting to load prepared data from: {data_dir}")
    try:
        x_train_path = os.path.join(data_dir, 'X_train.npy')
        x_test_path = os.path.join(data_dir, 'X_test.npy')
        y_train_path = os.path.join(data_dir, 'y_train.npy')
        y_test_path = os.path.join(data_dir, 'y_test.npy')

        if not os.path.exists(x_train_path):
            raise FileNotFoundError(f"{x_train_path} not found.")
        if not os.path.exists(x_test_path):
            raise FileNotFoundError(f"{x_test_path} not found.")
        if not os.path.exists(y_train_path):
            raise FileNotFoundError(f"{y_train_path} not found.")
        if not os.path.exists(y_test_path):
            raise FileNotFoundError(f"{y_test_path} not found.")

        X_train = pd.DataFrame(np.load(x_train_path))
        X_test = pd.DataFrame(np.load(x_test_path))
        y_train = pd.Series(np.load(y_train_path))
        y_test = pd.Series(np.load(y_test_path))
        print("Prepared data loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure data_preparation.py has been run and generated the files correctly in the specified directory.")
        exit()
    except ValueError as e:
        print(f"A ValueError occurred during loading: {e}")
        print("This usually means the .npy files contain non-numeric (object) data types or are corrupted.")
        exit()
    return X_train, X_test, y_train, y_test


def tune_hyperparameters(X_train, y_train, model_instance, model_name, param_grid, cv_strategy, search_method='grid', n_iter=None, random_state=None):
    """
    Performs hyperparameter tuning using GridSearchCV or RandomizedSearchCV.

    Args:
        X_train (pd.DataFrame): Training features DataFrame.
        y_train (pd.Series): Training target Series.
        model_instance: The machine learning model instance to tune.
        model_name (str): The name of the model (e.g., 'RandomForestRegressor').
        param_grid (dict): Dictionary with parameters names (str) as keys and lists of
                           parameter settings to try as values.
        cv_strategy: Cross-validation splitter (e.g., KFold instance).
        search_method (str): 'grid' for GridSearchCV or 'random' for RandomizedSearchCV.
        n_iter (int, optional): Number of parameter settings that are sampled when search_method is 'random'.
                                Required for RandomizedSearchCV.
        random_state (int, optional): Seed for random number generation, used in RandomizedSearchCV.

    Returns:
        tuple: A tuple containing the best estimator found and a dictionary of the best parameters.
    """
    print(
        f"\n--- Starting Hyperparameter Tuning for {model_name} using {search_method.capitalize()} Search ---")
    print(f"Parameter Grid: {param_grid}")

    if not param_grid:
        print(f"No parameter grid provided for {model_name}. Skipping tuning.")
        return model_instance, {}  # Return original model if no params for tuning

    if search_method == 'grid':
        search = GridSearchCV(estimator=model_instance, param_grid=param_grid, cv=cv_strategy,
                              scoring='r2', n_jobs=-1, verbose=2)  # Changed verbose to 2
    elif search_method == 'random':
        if n_iter is None:
            raise ValueError(
                "n_iter must be provided for RandomizedSearchCV when search_method is 'random'.")
        search = RandomizedSearchCV(estimator=model_instance, param_distributions=param_grid, n_iter=n_iter,
                                    cv=cv_strategy, scoring='r2', n_jobs=-1, verbose=2, random_state=random_state)  # Changed verbose to 2
    else:
        raise ValueError("search_method must be 'grid' or 'random'.")

    try:
        search.fit(X_train, y_train)
    except Exception as e:
        print(f"Error during hyperparameter tuning for {model_name}: {e}")
        print(f"Returning original (untuned) {model_name} model.")
        return model_instance, {}

    print(f"\nBest parameters found for {model_name}:")
    print(search.best_params_)
    print(
        f"Best cross-validation R^2 score for {model_name}: {search.best_score_:.4f}")
    print(f"--- Hyperparameter Tuning Completed for {model_name} ---")

    return search.best_estimator_, search.best_params_


def evaluate_model(model_name, model, X_test, y_test):
    """
    Evaluates the performance of a given model on the test set.
    """
    print(f"\n--- Evaluating {model_name} on Test Set ---")
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    metrics = {
        'Model': model_name,
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2
    }

    print(
        f"Metrics for {model_name}:\nMAE: {mae:.4f}\nMSE: {mse:.4f}\nRMSE: {rmse:.4f}\nR2: {r2:.4f}")
    return metrics


def save_model(model, model_path):
    """
    Saves the trained model to a file.
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Trained model saved to: {model_path}")


def main():
    """
    Orchestrates the model training, hyperparameter tuning, and evaluation process for multiple models.
    """
    config = load_config()

    paths_config = config['paths']
    current_script_dir = os.path.dirname(os.path.abspath(__file__))

    prepared_data_dir = os.path.normpath(os.path.join(
        current_script_dir, paths_config['prepared_data_directory']))
    model_output_dir = os.path.normpath(os.path.join(
        current_script_dir, paths_config['models_directory']))

    model_training_config = config['model_training']
    random_state = model_training_config['random_state']
    n_splits = model_training_config.get('n_splits', 5)  # For KFold CV

    tuning_config = config.get('hyperparameter_tuning', {})
    param_grid_all_models = tuning_config.get('param_grid', {})
    search_method = tuning_config.get('search_method', 'random')
    n_iter_random = tuning_config.get('n_iter_random', 10)

    X_train, X_test, y_train, y_test = load_prepared_data(prepared_data_dir)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    all_models_metrics = []
    trained_models = {}

    print("\n--- Starting Model Training and Hyperparameter Tuning for All Configured Models ---")

    for model_name, param_grid in param_grid_all_models.items():
        print(f"\n--- Initializing and processing {model_name} ---")
        base_model = None
        if model_name == 'LinearRegression':
            base_model = LinearRegression()
        elif model_name == 'RandomForestRegressor':
            base_model = RandomForestRegressor(random_state=random_state)
        elif model_name == 'XGBoostRegressor':
            # Corrected: changed 'reg:square_error' to 'reg:squarederror'
            base_model = xgb.XGBRegressor(
                objective='reg:squarederror', random_state=random_state)
        elif model_name == 'GradientBoostingRegressor':
            base_model = GradientBoostingRegressor(random_state=random_state)
        else:
            print(
                f"Warning: Model '{model_name}' specified in config.yaml param_grid is not recognized. Skipping.")
            continue

        tuned_model, best_tuned_params = tune_hyperparameters(
            X_train, y_train, base_model, model_name, param_grid, kf,
            search_method=search_method, n_iter=n_iter_random, random_state=random_state
        )

        metrics = evaluate_model(
            f"Tuned_{model_name}", tuned_model, X_test, y_test)
        all_models_metrics.append(metrics)
        trained_models[f"Tuned_{model_name}"] = tuned_model

        best_model_path = os.path.join(
            model_output_dir, f'tuned_{model_name.lower()}.joblib')
        save_model(tuned_model, best_model_path)

    if all_models_metrics:
        metrics_df = pd.DataFrame(all_models_metrics)
        metrics_df.set_index('Model', inplace=True)
        print("\n--- All Tuned Model Comparison ---")
        print(metrics_df.to_string())
        print("----------------------------------")

        best_overall_model_name = metrics_df['R2'].idxmax()
        print(
            f"\nBest performing model based on R2 score: {best_overall_model_name}")
    else:
        print("No models were successfully tuned or evaluated.")


if __name__ == '__main__':
    print("--- Starting the ML Pipeline (Model Training & Tuning) ---")
    main()
    print("Model Training & Tuning completed successfully.")
