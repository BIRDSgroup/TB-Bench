from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import pickle
class AbstractModel(ABC):
    """
    An abstract base class that defines the blueprint for a model configuration.

    This class enforces that any inheriting concrete class must provide a name,
    a model instance, and parameter dictionaries. It is designed to create a
    standardized interface for handling different models in a machine learning
    pipeline.
    """

    def __init__(self):
        """
        Initializes the configuration, setting best_params to None.
        This attribute is intended to be populated after hyperparameter tuning.
        """
        self.best_params: Optional[Dict[str, Any]] = None

    @property
    @abstractmethod
    def name(self) -> str:
        """A string identifier for the model (e.g., 'LogisticRegression')."""
        raise NotImplementedError

    @property
    @abstractmethod
    def model(self) -> Any:
        """The scikit-learn (or similar) model instance."""
        raise NotImplementedError

    @property
    @abstractmethod
    def param_grid(self) -> Optional[Dict[str, Any]]:
        """
        A dictionary defining the hyperparameter grid for tuning.
        Should return None if no tuning is desired.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def static_params(self) -> Dict[str, Any]:
        """
        A dictionary of fixed parameters to be used with the model,
        often combined with the best_params after tuning.
        """
        raise NotImplementedError
        
    def save(self,final_model,data_key):
        with open('saved_models/'+ data_key +'_model.pkl', 'wb') as file:
            pickle.dump(final_model, file)

    def load(self,data_key):
        filename='saved_models/'+ data_key +'_model.pkl'
        with open(filename, "rb") as f:
            model = pickle.load(f)
        return model