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
    """
    This function takes a study and returns the top `percentage` of trials
    depending on their value.

    Args:
        study (Study): The study to search.
        percentage (int, optional): The top percentage to select. Defaults to 10.

    Returns:
        list[Trial]: The list of the top-percentage trials.
    """
    is_maximization = study.direction == StudyDirection.MAXIMIZE
    sorted_trials = sorted(
        study.trials, key=lambda trial: trial.value, reverse=is_maximization
    )
    n_select = max(1, len(sorted_trials) // percentage)

    top_trials = sorted_trials[:n_select]

    return top_trials


def suggest_imputer_params(trial: Trial, categorical: bool = False) -> dict[str, Any]:
    """
    This function generates the parameters in an optuna trial for a simple imputer.

    Args:
        trial (Trial): The current trial.
        categorical (bool, optional): Whether the imputer is for categorical data.
        Defaults to False.

    Returns:
        dict[str, Any]: The parameters for the SimpleImputer.
    """
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
    """
    This function generates the parameters in an optuna trial for a one hot encoder.

    Args:
        trial (Trial): The current trial.

    Returns:
        dict[str, Any]: The parameters for the OneHotEncoder.
    """
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
    """
    This function generates the parameters in an optuna trial for a weight of
    evidence encoder.

    Args:
        trial (Trial): The current trial.

    Returns:
        dict[str, Any]: The parameters for the WOEEncoder.
    """
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
    trial: Trial, numerical_columns: Sequence[str], categorical_columns: Sequence[str]
) -> ColumnTransformer:
    """
    This function instantiates a processor in an optuna trial.

    Args:
        trial (Trial): The current trial.
        numerical_columns (Sequence[str]): An iterable containing the names of the numerical columns.
        categorical_columns (Sequence[str]): An iterable containing the names of the categorical columns.

    Returns:
        dict[str, Any]: An instantiated ColumnTransformer.
    """
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
    """
    This function generates the parameters in an optuna trial for a lightgbm model.

    Args:
        trial (Trial): The current trial.

    Returns:
        dict[str, Any]: The parameters for the lightgbm model.
    """
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
    """
    This function instantiates a processor and model in an optuna trial and
    returns them as a Pipeline object.

    Args:
        trial (Trial): The current trial.
        numerical_columns (Sequence[str]): An iterable containing the names of the numerical columns.
        categorical_columns (Sequence[str]): An iterable containing the names of the categorical columns.
        suggest_function (Callable[[Trial], LGBMClassifier], optional): A function that suggests the
        lightgbm model's parameters. Defaults to suggest_lgbm_params.
        previous_params (Optional[dict[str, Any]], optional): The parameters of the previous steps in a
        stepwise optimization. Defaults to None.

    Returns:
        dict[str, Any]: An instantiated ColumnTransformer.
    """
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
    """
    This function generates the parameters in an optuna trial for a lightgbm model
    for the first step in a stepwise optimization. The tuned parameter is `colsample_bytree`.

    Args:
        trial (Trial): The current trial.

    Returns:
        dict[str, Any]: The parameters for the lightgbm model.
    """
    params = {
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 1),
        "verbose": -1,
        "random_state": 42,
    }
    return params


def suggest_lgbm_step_two(
    trial: Trial, previous_params: dict[str, Any]
) -> dict[str, Any]:
    """
    This function generates the parameters in an optuna trial for a lightgbm model
    for the second step in a stepwise optimization. The tuned parameter is `num_leaves`.

    Args:
        trial (Trial): The current trial.
        previous_params (dict[str, Any]): The parameters found in the previous optimization step.

    Returns:
        dict[str, Any]: The parameters for the lightgbm model.
    """
    params = previous_params | {
        "num_leaves": trial.suggest_int("num_leaves", 3, 63),
    }
    return params


def suggest_lgbm_step_three(
    trial: Trial, previous_params: dict[str, Any]
) -> dict[str, Any]:
    """
    This function generates the parameters in an optuna trial for a lightgbm model
    for the third step in a stepwise optimization. The tuned parameter is `subsample`.

    Args:
        trial (Trial): The current trial.
        previous_params (dict[str, Any]): The parameters found in the previous optimization step.

    Returns:
        dict[str, Any]: The parameters for the lightgbm model.
    """
    params = previous_params | {
        "subsample": trial.suggest_float("subsample", 0.1, 1),
    }
    return params


def suggest_lgbm_step_four(
    trial: Trial, previous_params: dict[str, Any]
) -> dict[str, Any]:
    """
    This function generates the parameters in an optuna trial for a lightgbm model
    for the second step in a stepwise optimization. The tuned parameters are `reg_alpha`
    and `reg_lambda`.

    Args:
        trial (Trial): The current trial.
        previous_params (dict[str, Any]): The parameters found in the previous optimization step.

    Returns:
        dict[str, Any]: The parameters for the lightgbm model.
    """
    params = previous_params | {
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 20),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 20),
    }
    return params

def get_previous_params(studies : Sequence[Study], objectives: Sequence[Callable[[Trial], float]]) -> Optional[dict[str, Any]]:
    """
    This function takes an iterable of studies and objective funcitons and returns the parameters found in each of them.

    Args:
        studies (Sequence[Study]): The studies to extract the information from.
        objectives (Sequence[Callable[[Trial], float]]): The objective function for each study.

    Returns:
        Optional[dict[str, Any]]: The parameters for the previous models.
    """
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
        """
        The StepwiseStudy is a wrapper around optuna functionality that allows the running of sequential
        optimization steps.

        Args:
            n_steps (int): The amount of optimization steps to perform.
            storage (str): The storage to use (opuna storage database).
            study_name (str, optional): The name of the study. Defaults to "study".
            samplers (BaseSampler | Sequence[BaseSampler], optional): The samplers to use at each step. Defaults to TPESampler().
            direction (str, optional): The direction of the optimization. Defaults to "maximize".
        """
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
        """
        This function runs the optimize method on each of the provided studies.

        Args:
            objectives (Sequence[Callable[[Trial,): The objective functions to use for each study.
            n_trials (int | Sequence[int], optional): The amount of trials to run for each study. Defaults to 100.
            all_trials (bool, optional): Whether to store the outputs of all the steps or only the last
            one. Defaults to False.
        """
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
