# --- DecisiontreeManager Class ---
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import pickle
from .Model import AbstractModel

class DecisionTreeManager(AbstractModel):
    def __init__(self,n):
        super().__init__()
        print("DecisiontreeManager: Configuration initialized.")
        print("-" * 60)
        
    @property
    def name(self) -> str:
        return "Decisiontree"

    @property
    def model(self) -> DecisionTreeClassifier:
        return DecisionTreeClassifier(**self.static_params)

    @property
    def param_grid(self) -> dict[str, any]:
        return {

        }

    @property
    def static_params(self) -> dict[str, any]:
        return {
            'max_depth': 15,
            'criterion': 'gini',
            'splitter': 'best',
            'min_samples_leaf': 1,
            'random_state': 42
        }

    def load(self,data_key):
        filename='saved_models/'+data_key +'_model.pkl'
        with open(filename, "rb") as f:
            model = pickle.load(f)
        return model

    def tune_hyperparams(self, X, y, outer_cv):
        print(f"Tuning {self.name} using GridSearchCV...")
        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=self.param_grid,
            cv=outer_cv,
            scoring='average_precision',
            n_jobs=-1
        )
        grid_search.fit(X, y)
        self.best_params = {**grid_search.best_params_, **self.static_params}
        print(f"{self.name} best hyperparams: {self.best_params}")
        return self.best_params

'''
  #   'max_depth': [None, 7, 10,15],
         #   'min_samples_leaf': [1, 2, 4, 8],
         #   'max_features': [None, 'sqrt', 'log2'],
         #   'criterion': ['gini', 'entropy'],
         #   'splitter': ['best', 'random']
'''
