from typing import Any, Callable
from category_encoders import WOEEncoder
from lightgbm import LGBMClassifier
from optuna import Trial, Study
from optuna.study import StudyDirection
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

def suggest_imputer_params(trial : Trial, categorical : bool=False) -> dict[str, Any]:
    if not categorical:
        strategy = trial.suggest_categorical('numerical_strategy', ['mean', 'median', 'most_frequent', 'constant'])
    else:
        strategy = trial.suggest_categorical('categorical_strategy', ['most_frequent', 'constant'])
    
    add_indicator = trial.suggest_categorical('add_indicator', [True, False])

    fill_value = 'missing' if categorical else -1

    params = {
        'strategy': strategy,
        'add_indicator': add_indicator,
        'fill_value': fill_value
    }

    return params

def suggest_ohe_params(trial : Trial) -> dict[str, Any]:
    params = {
        'drop': trial.suggest_categorical('drop', [None, 'first', 'if_binary']),
        'handle_unknown': trial.suggest_categorical('handle_unknown', ['ignore', 'infrequent_if_exist']),
        'min_frequency': trial.suggest_float('min_frequency', 0.01, 0.2),
        'max_categories': trial.suggest_int('max_categories', 5, 20),
    }
    return params

def suggest_woe_params(trial : Trial) -> dict[str, Any]:
    params = {
        'randomized': trial.suggest_categorical('randomized', [True, False]),
        'sigma': trial.suggest_float('sigma', 0.01, 10),
        'regularization': trial.suggest_float('regularization', 0.01, 10),
        'handle_unknown': 'value',
        'handle_missing': 'value',
        'random_state': 42,
    }
    return params

def get_processor(trial : Trial, numerical_columns : list[str], categorical_columns : list[str]) -> ColumnTransformer:
    impute = trial.suggest_categorical('impute', [True, False])
    if impute:
        numerical_imputer = SimpleImputer(**suggest_imputer_params(trial))
        categorical_imputer = SimpleImputer(**suggest_imputer_params(trial, categorical=True))

    encoder = trial.suggest_categorical('encoder', ['ohe', 'woe'])
    categorical_encoder = OneHotEncoder(**suggest_ohe_params(trial)) if encoder=='ohe' else WOEEncoder(**suggest_woe_params(trial))
    
    categorical_pipeline = categorical_encoder if not impute else Pipeline([('imputer', categorical_imputer), ('encoder', categorical_encoder)])

    processor = ColumnTransformer([
        ('numerical_pipeline', numerical_imputer, numerical_columns),
        ('categorical_pipeline', categorical_pipeline, categorical_columns),
    ]) if impute else ColumnTransformer([
        ('categorical_pipeline', categorical_pipeline, categorical_columns),
    ], remainder='passthrough')

    return processor


def suggest_lgbm_params(trial : Trial) -> dict[str, Any]:
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1500),
        'max_depth': trial.suggest_int('max_depth', 1, 30),
        'num_leaves': trial.suggest_int('num_leaves', 3, 63),
        'learning_rate': trial.suggest_float('learning_rate', 1e-6, 0.05),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1),
        'subsample': trial.suggest_float('subsample', 0.1, 1),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 20),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 20),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 20),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 200),
        'random_state': 42
    }
    return params


def get_lgbm_classifier(trial : Trial, numerical_columns : list[str], categorical_columns : list[str], suggest_function : Callable[[Trial], LGBMClassifier]=suggest_lgbm_params) -> Pipeline:
    processor = get_processor(trial, numerical_columns, categorical_columns)
    params = suggest_function(trial)
    model = Pipeline([
        ('processor', processor),
        ('model', LGBMClassifier(**params))
    ])
    return model


def suggest_lgbm_step_one(trial : Trial) -> dict[str, Any]:
    params = {
        'feature_fraction': trial.suggest_float('feature_fraction', 0.1, 1),
        'random_state': 42
    }
    return params

def suggest_lgbm_step_two(trial : Trial, previous_params : dict[str, Any]) -> dict[str, Any]:
    params = {
        'num_leaves': trial.suggest_float('num_leaves', 3, 63),
    } | previous_params
    return params

def suggest_lgbm_step_three(trial : Trial, previous_params : dict[str, Any]) -> dict[str, Any]:
    params = {
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.1, 1),
    } | previous_params
    return params

def suggest_lgbm_step_four(trial : Trial, previous_params : dict[str, Any]) -> dict[str, Any]:
    params = {
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 20),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 20),
    } | previous_params
    return params

def get_top_n_percent_trials(study : Study, percentage : int=10) -> list[Trial]:
    is_maximization = study.direction == StudyDirection.MAXIMIZE
    sorted_trials = sorted(study.trials, key=lambda trial: trial.value, reverse=is_maximization)
    n_select = max(1, len(sorted_trials) // percentage)

    top_trials = sorted_trials[:n_select]

    return top_trials