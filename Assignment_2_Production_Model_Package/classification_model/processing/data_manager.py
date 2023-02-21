import typing as t
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pyparsing import Any
from sklearn.pipeline import Pipeline

from classification_model import __version__ as _version
from classification_model.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config



def pre_pipeline_preparation(*, input_data: pd.DataFrame) -> pd.DataFrame:
    #replace question marks with NaN
    data = pd.DataFrame.replace("?", np.nan)
    data["cabin"] = data["cabin"].apply(get_first_cabin)
    data["title"] = data["name"].apply(get_title)

    data["fare"] = data["fare"].astype("float")
    data["age"] = data["age"].astype("float")

    data.drop(columns=["name", "ticket", "boat", "body", "home.dest"], inplace=True)

    return data

def load_raw_dataset(*, file_name: str) -> pd.DataFrame:
    """Load dataset from raw data source."""
    _data = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
    return _data
    


def get_first_cabin(row: Any) -> str:
    """Get first character of cabin string."""
    try:
        return row.split()
    except AttributeError:
        return np.nan
    
def get_title(passenger_name: str) -> str:
    """Get title from passenger name."""
    return passenger_name.split(",")[1].split(".")[0].strip()
    


def load_dataset(*, file_name: str) -> pd.DataFrame:
    dataframe = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
    dataframe["MSSubClass"] = dataframe["MSSubClass"].astype("O")

    # rename variables beginning with numbers to avoid syntax errors later
    transformed = dataframe.rename(columns=config.model_config.variables_to_rename)
    return transformed


def save_pipeline(*, pipeline_to_persist: Pipeline) -> None:
    """Persist the pipeline.
    Saves the versioned model, and overwrites any previous
    saved models. This ensures that when the package is
    published, there is only one trained model that can be
    called, and we know exactly how it was built.
    """

    # Prepare versioned save file name
    save_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
    save_path = TRAINED_MODEL_DIR / save_file_name

    remove_old_pipelines(files_to_keep=[save_file_name])
    joblib.dump(pipeline_to_persist, save_path)


def load_pipeline(*, file_name: str) -> Pipeline:
    """Load a persisted pipeline."""

    file_path = TRAINED_MODEL_DIR / file_name
    trained_model = joblib.load(filename=file_path)
    return trained_model


def remove_old_pipelines(*, files_to_keep: t.List[str]) -> None:
    """
    Remove old model pipelines.
    This is to ensure there is a simple one-to-one
    mapping between the package version and the model
    version to be imported and used by other applications.
    """
    do_not_delete = files_to_keep + ["__init__.py"]
    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()
