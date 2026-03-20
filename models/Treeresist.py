# --- TreeresistManager Class ---

import sys
sys.path.append("/data/users/brintha/Treesist-TB")
from sklearn.model_selection import GridSearchCV
import pickle
from .Model import AbstractModel
from new_splitter import NewBestSplitter
from typing import Dict, Any, List

import pandas as pd
import numpy as np
from collections import defaultdict
import sklearn
from sklearn import tree
from sklearn.tree._tree import TREE_LEAF
from sklearn.tree import _criterion as criterions
from sklearn.utils import check_random_state
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score

class NewDecisionTreeClassifier(tree.DecisionTreeClassifier):
    
    def prune_tree(self):
        for i in range(self.tree_.capacity):
            cnt_1, cnt_0 = self.tree_.value[i][0]
            if cnt_0 > cnt_1:
                # this is negative node => don't split it further
                self.tree_.children_left[i] = TREE_LEAF
                self.tree_.children_right[i] = TREE_LEAF
        return self.tree_
    
    
    def get_max_features(self, n_features_):
        if isinstance(self.max_features, str):
            if self.max_features == "auto":
                if is_classification:
                    max_features = max(1, int(np.sqrt(n_features_)))
                else:
                    max_features = n_features_
            elif self.max_features == "sqrt":
                max_features = max(1, int(np.sqrt(n_features_)))
            elif self.max_features == "log2":
                max_features = max(1, int(np.log2(n_features_)))
            else:
                raise ValueError(
                    'Invalid value for max_features. Allowed string '
                    'values are "auto", "sqrt" or "log2".')
        elif self.max_features is None:
            max_features = n_features_
        elif isinstance(self.max_features, (numbers.Integral, np.integer)):
            max_features = self.max_features
        else:  # float
            if self.max_features > 0.0:
                max_features = max(1,
                                   int(self.max_features * n_features_))
            else:
                max_features = 0
        return max_features
    
    def get_min_weight_leaf(self, sample_weight, n_samples):
        if sample_weight is None:
            min_weight_leaf = (self.min_weight_fraction_leaf *
                               n_samples)
        else:
            min_weight_leaf = (self.min_weight_fraction_leaf *
                               np.sum(sample_weight)) 
        return min_weight_leaf
    
    def fit(self, X, y, sample_weight=None, check_input=True,
            X_idx_sorted=None, prune_tree=True):
        if self.splitter == 'new_best':
            n_samples, n_features_ = X.shape
            crit = criterions.Gini(1, np.array([2]))
            self.splitter = NewBestSplitter(crit, #crit,
                                   self.get_max_features(n_features_),
                                   self.min_samples_leaf,
                                   self.get_min_weight_leaf(sample_weight, n_samples),
                                   check_random_state(self.random_state)
                                  )
        super(NewDecisionTreeClassifier, self).fit(
            X, y,
            sample_weight=sample_weight,
            check_input=check_input,
            X_idx_sorted=X_idx_sorted)
        
        if prune_tree:
            self.prune_tree()
        return self
    
    def get_stats(self, feature_names=None):
        def _name(idx):
            if feature_names is None:
                return idx
            else:
                return feature_names[idx
                                    ]
        if isinstance(self.splitter, NewBestSplitter):
            result = []
            ties = self.splitter.get_current_node_ties()
            for i in range(self.splitter.get_node_idx() + 1):
                result.append([_name(idx) for idx in np.flatnonzero(ties[i])])
            return result
        else:
            raise ValueError('this action is supported for NewBestSplitter only')

(LEAF_SENSITIVE, LEAF_RESISTANT, 
 NODE_REGULAR, 
 NODE_AND_UPPER, NODE_AND_LOWER, 
 NODE_REVERSE_UPPER, NODE_REVERSE_LOWER) = ['_leaf_sens_', '_leaf_res_', 
                                           '_reg_', 
                                           '_and_up_', '_and_low_',
                                           '_rev_up_', '_rev_low']
           







class ShortListDecisionTreeClassifier(NewDecisionTreeClassifier):
    
    def __init__(self, *args, **kwargs):
        self.feature_genes = kwargs.pop('feature_genes')
        self.short_list = kwargs.pop('short_list')
        self.max_difference = kwargs.pop('max_difference')
        super().__init__(*args, **kwargs)
    
    def prune_tree_short_list(self):
        tree_ = self.tree_
        short_list = set(self.short_list)
        updated_list = short_list.copy()
        self.node2type = get_node_types(self)
        
        visited = set()
        queue = []  # queue

        def can_add_gene(current_gene):
            difference = (updated_list.union([current_gene])) - short_list
            return len(difference) <= self.max_difference
        
        def make_leaf(i):
            tree_.children_left[i] = TREE_LEAF
            tree_.children_right[i] = TREE_LEAF
            
        def enqueue_children(i):
            for next_node in [tree_.children_left[i],  # go to left child first
                              tree_.children_right[i]  # go to right child then
                             ]:
                if not next_node in visited:
                    queue.append(next_node)
        
        def process_node(i):
            if i in visited:  
                return
            visited.add(i) 
        
            node_type = self.node2type[i]
            if node_type in [LEAF_RESISTANT, LEAF_SENSITIVE]:
                return
            cnt_1, cnt_0 = tree_.value[i][0]
            if cnt_0 > cnt_1:
                # this is negative node => don't split it further
                make_leaf(i)
                return
            else:
                current_feature = tree_.feature[i]
                current_gene = self.feature_genes[current_feature]
                can_add = can_add_gene(current_gene)
    #             print(f'{features_names[current_feature]} updated_list = {updated_list} can_add = {can_add}')
                if not can_add:
                    if node_type == NODE_AND_LOWER:  # dont prune and_low nodes
                        # we do no count and_low's gene as a new added gene
                        enqueue_children(i) 
                    else:
                        make_leaf(i)
                        return
                else:
                    if node_type == NODE_AND_LOWER:
                        # we do no count and_low's gene as a new added gene
                        enqueue_children(i)
                    else:
                        updated_list.update([current_gene]) # take this gene into account
                        enqueue_children(i)
            
            while queue:
                process_node(queue.pop(0))
        
        process_node(0)  # start with top node          
        return tree_
    
    def fit(self, X, y, sample_weight=None, check_input=True,
            X_idx_sorted=None):
        super().fit(
            X, y,
            sample_weight=sample_weight,
            check_input=check_input,
            X_idx_sorted=X_idx_sorted,
            prune_tree=False)
        
        self.prune_tree_short_list()
        return self
    
def fit_model(clf,X_input,y_input,show_accuracy=1):
    clf = clf.fit(X, y)
    if show_accuracy==1:
        accuracy_score=clf.score(X, y, sample_weight=None)
    print(accuracy_score,"accuracy score on training data")
    return(clf)

def get_node_types(clf):
    def is_leaf(i):
        return ((clf.tree_.children_left[i] == TREE_LEAF)
                and (clf.tree_.children_right[i] == TREE_LEAF))
    def is_sens(i):
        cnt_1, cnt_0 = clf.tree_.value[i][0]
        return cnt_0 <= cnt_1
    def is_res(i):
        cnt_1, cnt_0 = clf.tree_.value[i][0]
        return cnt_0 > cnt_1
    
    def is_and(i):
        r_child = clf.tree_.children_right[i]
        if is_leaf(r_child):
            return False
        r_grandchild = clf.tree_.children_right[r_child]
        if is_leaf(r_grandchild) and is_res(r_grandchild):
            return True
        return False
    
    def is_revers(i):
        r_child = clf.tree_.children_right[i]
        if is_leaf(r_child):
            return False
        r_grandchild = clf.tree_.children_right[r_child]
        if is_leaf(r_grandchild) and is_sens(r_grandchild):
            return True
        return False
    
    result = {}
    for i in range(clf.tree_.capacity):
        if i in result:
            continue
        if is_leaf(i):
            # it's leaf
            result[i] = LEAF_RESISTANT if is_res(i) else LEAF_SENSITIVE
        else:
            # it's node
            result[i] = NODE_REGULAR
            if is_and(i):
                result[i] = NODE_AND_UPPER
                result[clf.tree_.children_right[i]] = NODE_AND_LOWER
            if is_revers(i):
                result[i] = NODE_REVERSE_UPPER
                result[clf.tree_.children_right[i]] = NODE_REVERSE_LOWER
            
    return result


class TreeresistManager(AbstractModel):
    def __init__(self,n):
        self._n_features = n
        self.priority_genes=["Gene"]*self._n_features
        super().__init__()
        print("TreeresistManager: Configuration initialized.")
    
        print("-" * 60)
        
    @property
    def name(self) -> str:
        return "Treeresist"

    @property
    def model(self) -> NewDecisionTreeClassifier:
        """Return an instance of the custom Decision Tree classifier."""
        return ShortListDecisionTreeClassifier(**self.static_params)

    @property
    def param_grid(self) -> Dict[str, Any]:
        return {
           
        }

    @property
    def static_params(self) -> Dict[str, Any]:
        return {
            'criterion': 'gini', 
            'splitter': 'new_best',
            'class_weight' : {0:1.0, 1:1.0}, 
            #'class_weight' : 'balanced', 
            'max_depth' : 10,
            #'min_samples_leaf' : 1, 
            'feature_genes' : self.priority_genes,
            'short_list' : ["Gene"], 
            'max_difference' : 1,
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
