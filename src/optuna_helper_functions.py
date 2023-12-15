from copy import deepcopy
from functools import partial, reduce
from typing import Any, Callable, Optional, Sequence, TypeVarTuple
from category_encoders import WOEEncoder
from lightgbm import LGBMClassifier
from optuna import Trial, Study, create_study
from optuna.study import StudyDirection
from optuna.samplers import BaseSampler, TPESampler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.utils import get_categorical_columns, get_numerical_columns

Ts = TypeVarTuple("Ts")


def get_top_n_percent_trials(study: Study, percentage: int = 10) -> list[Trial]:
    is_maximization = study.direction == StudyDirection.MAXIMIZE
    sorted_trials = sorted(
        study.trials, key=lambda trial: trial.value, reverse=is_maximization
    )
    n_select = max(1, len(sorted_trials) // percentage)

    top_trials = sorted_trials[:n_select]

    return top_trials


def suggest_imputer_params(trial: Trial, categorical: bool = False) -> dict[str, Any]:
    if not categorical:
        strategy = trial.suggest_categorical(
            "numerical_strategy", ["mean", "median", "most_frequent", "constant"]
        )
    else:
        strategy = trial.suggest_categorical(
            "categorical_strategy", ["most_frequent", "constant"]
        )

    add_indicator = trial.suggest_categorical("add_indicator", [True, False])

    fill_value = "missing" if categorical else -1

    params = {
        "strategy": strategy,
        "add_indicator": add_indicator,
        "fill_value": fill_value,
    }

    return params


def suggest_ohe_params(trial: Trial) -> dict[str, Any]:
    params = {
        "drop": trial.suggest_categorical("drop", [None, "first", "if_binary"]),
        "handle_unknown": trial.suggest_categorical(
            "handle_unknown", ["ignore", "infrequent_if_exist"]
        ),
        "min_frequency": trial.suggest_float("min_frequency", 0.01, 0.2),
        "max_categories": trial.suggest_int("max_categories", 5, 20),
    }
    return params


def suggest_woe_params(trial: Trial) -> dict[str, Any]:
    params = {
        "randomized": trial.suggest_categorical("randomized", [True, False]),
        "sigma": trial.suggest_float("sigma", 0.01, 10),
        "regularization": trial.suggest_float("regularization", 0.01, 10),
        "handle_unknown": "value",
        "handle_missing": "value",
        "random_state": 42,
    }
    return params


def get_processor(
    trial: Trial, numerical_columns: list[str], categorical_columns: list[str]
) -> ColumnTransformer:
    impute = trial.suggest_categorical("impute", [True, False])
    if impute:
        numerical_imputer = SimpleImputer(**suggest_imputer_params(trial))
        categorical_imputer = SimpleImputer(
            **suggest_imputer_params(trial, categorical=True)
        )

    encoder = trial.suggest_categorical("encoder", ["ohe", "woe"])
    categorical_encoder = (
        OneHotEncoder(**suggest_ohe_params(trial))
        if encoder == "ohe"
        else WOEEncoder(**suggest_woe_params(trial))
    )

    categorical_pipeline = (
        categorical_encoder
        if not impute
        else Pipeline(
            [("imputer", categorical_imputer), ("encoder", categorical_encoder)]
        )
    )

    processor = (
        ColumnTransformer(
            [
                ("numerical_pipeline", numerical_imputer, numerical_columns),
                ("categorical_pipeline", categorical_pipeline, categorical_columns),
            ]
        )
        if impute
        else ColumnTransformer(
            [
                ("categorical_pipeline", categorical_pipeline, categorical_columns),
            ],
            remainder="passthrough",
        )
    )

    return processor


def suggest_lgbm_params(trial: Trial) -> dict[str, Any]:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1500),
        "max_depth": trial.suggest_int("max_depth", 1, 30),
        "num_leaves": trial.suggest_int("num_leaves", 3, 63),
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 0.05),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1),
        "subsample": trial.suggest_float("subsample", 0.1, 1),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 20),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 20),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 20),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 200),
        "random_state": 42,
    }
    return params


def get_lgbm_classifier(
    trial: Trial,
    numerical_columns: Sequence[str],
    categorical_columns: Sequence[str],
    suggest_function: Callable[[Trial], LGBMClassifier] = suggest_lgbm_params,
    previous_params: Optional[dict[str, Any]] = None,
) -> Pipeline:
    processor = get_processor(trial, numerical_columns, categorical_columns)
    
    if previous_params is not None:
        previous_lgbm_params = {key: value for key, value in previous_params.items() if 'lgbm' in key}
    
    params = (
        suggest_function(trial)
        if previous_params is None
        else suggest_function(trial, previous_lgbm_params)
    )

    model = Pipeline([("processor", processor), ("model", LGBMClassifier(**params))])
    return model


def suggest_lgbm_step_one(trial: Trial) -> dict[str, Any]:
    params = {
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 1),
        "verbose": -1,
        "random_state": 42,
    }
    return params


def suggest_lgbm_step_two(
    trial: Trial, previous_params: dict[str, Any]
) -> dict[str, Any]:
    params = previous_params | {
        "num_leaves": trial.suggest_int("num_leaves", 3, 63),
    }
    return params


def suggest_lgbm_step_three(
    trial: Trial, previous_params: dict[str, Any]
) -> dict[str, Any]:
    params = previous_params | {
        "subsample": trial.suggest_float("subsample", 0.1, 1),
    }
    return params


def suggest_lgbm_step_four(
    trial: Trial, previous_params: dict[str, Any]
) -> dict[str, Any]:
    params = previous_params | {
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 20),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 20),
    }
    return params

def get_previous_params(studies : Sequence[Study], objectives: Sequence[Callable[[Trial], float]]) -> Optional[dict[str, Any]]:
    missing_steps = len(studies)
    if missing_steps==0:
        previous_model_params = None
    else:
        best_trial = studies[-1].best_trial
        instantiation_params = {key: value for key, value in objectives[-1].keywords.items()}
        numerical_columns = get_numerical_columns(instantiation_params.get('X'))
        categorical_columns = get_categorical_columns(instantiation_params.get('X'))
        suggest = (
            partial(instantiation_params.get('suggest_function'), previous_params=get_previous_params(studies[:missing_steps-1], objectives[:missing_steps-1]))
            if len(studies)>1 else instantiation_params.get('suggest_function')
        )
        previous_model_params = get_lgbm_classifier(best_trial, numerical_columns, categorical_columns, suggest)[-1].get_params()
    return previous_model_params

class StepwiseStudy:
    def __init__(
        self,
        n_steps: int,
        storage: str,
        study_name: str = "study",
        samplers: BaseSampler | Sequence[BaseSampler] = TPESampler(),
        direction: str = "maximize",
    ) -> None:
        self.n_steps = n_steps
        self.study_name = study_name
        self.direction = direction
        self.samplers = (
            samplers
            if isinstance(samplers, Sequence)
            else [*map(lambda _: deepcopy(samplers), range(n_steps))]
        )

        study_names = map(
            lambda step: f"step_{step}_{study_name}", range(1, n_steps + 1)
        )
        self.studies = [
            *map(
                lambda sampler, name: create_study(
                    storage=storage,
                    sampler=sampler,
                    study_name=name,
                    direction=direction,
                ),
                self.samplers,
                study_names,
            )
        ]

    def optimize(
        self,
        objectives: Sequence[Callable[[Trial, *Ts], float]],
        n_trials: int | Sequence[int] = 100,
        all_trials: bool = False,
    ) -> None:
        n_trials_ = (
            n_trials
            if isinstance(n_trials, Sequence)
            else [*map(lambda _: n_trials, range(self.n_steps))]
        )

        best_params = []
        for step, (study, objective, trials) in enumerate(zip(self.studies, objectives, n_trials_)):
            if step==0:
                study.optimize(objective, trials)
            else:
                previous_model_params = get_previous_params(self.studies[:step], objectives[:step])
                study.optimize(partial(objective, previous_params=previous_model_params), trials)
            
            best_params.append(study.best_params)

        self.best_step_params = [*map(lambda study: study.best_params, self.studies)]
        self.best_params = reduce(
            lambda params_1, params_2: params_1 | params_2, best_params
        )

        if all_trials:
            self.trials = [*map(lambda study: deepcopy(study.trials), self.studies)]
        else:
            self.trials = deepcopy(self.studies[-1].trials)
