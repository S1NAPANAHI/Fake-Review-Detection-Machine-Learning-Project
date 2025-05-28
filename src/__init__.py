
"""

Fake Review Detection - src package

  

This package contains all core modules for the fake review detection system, including:

- Data collection

- Preprocessing

- Feature engineering

- Modeling

- Evaluation

- Interpretation

- Deployment

- Monitoring

- Utilities

"""

  

from .data_collection import DataCollector

from .preprocessing import TextPreprocessor

from .feature_engineering import FeatureEngineer

from .modeling import ModelTrainer

from .evaluation import Evaluation

from .interpretation import ModelInterpreter

from .deployment import FakeReviewDetector

from .monitoring import Monitoring

from .utils import (

    set_seed,

    load_data,

    save_model,

    load_model,

    save_metrics,

    get_feature_names,

    setup_directories,

    log_hparams,

    timeit,

    validate_input_data

)

  

__all__ = [

    "DataCollector",

    "TextPreprocessor",

    "FeatureEngineer",

    "ModelTrainer",

    "Evaluation",

    "ModelInterpreter",

    "FakeReviewDetector",

    "Monitoring",

    # Utility functions

    "set_seed",

    "load_data",

    "save_model",

    "load_model",

    "save_metrics",

    "get_feature_names",

    "setup_directories",

    "log_hparams",

    "timeit",

    "validate_input_data"

]

