# scikit-learn 

## Cross-Validation
### Simple Cross-Validation
```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
import numpy as np

model = LogisticRegression(C=0.1, solver='sag')
np.mean(cross_val_score(model, x_train, y_train, cv=3, verbose=True, scoring='roc_auc'))
```

### Cross-Validation with Grid-Search
```python
from sklearn.model_selection import GridSearchCV
parameters = {'kernel': ('linear', 'rbf'), 'C':[1, 10]}
svc = svm.SVC()
clf = GridSearchCV(svc, parameters)
clf.fit(x, y)

predictions = clf.predict(x_test)
```

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer

pipe = Pipeline([('vectorizer', TfidfVectorizer()), ('classifier' , LogisticRegression())])

param_grid = [
    {'classifier__penalty' : ['l2'],
    'classifier__C' : np.logspace(-4, 4, 10),
    'classifier__solver' : ['lbfgs', 'liblinear', 'sag', 'saga']},

    'vectorizer__sublinear_tf' : [True, False],
    'vectorizer__ngram_range' : [(1, 1), (1, 2), (1, 3)],
    'vectorizer__min_df' : [1, 2, 3],
    'vectorizer__strip_accents' : ['unicode'],
    'vectorizer__max_features' : [5000, 10000],
    'vectorizer__stop_words' : [None, 'english']}
]

clf = GridSearchCV(pipe, param_grid = param_grid, cv = 3, verbose=True, scoring='roc_auc', n_jobs=-1)
grid = clf.fit(x, y)

print('best score: {}\nbest params: {}'.format(grid.best_score_, grid.best_params_))  

# Show top-k model configs
results = pd.DataFrame(grid.cv_results_)
results.sort_values(by='rank_test_score', inplace=True)
print(results.head())

# Select one of the top-k models
params_2nd_best = results.iloc[1, 'params']
clf_2nd_best = grid.best_estimator_.set_params(**params_2nd_best)
```

You can also pass a list of dictionaries as param_grid, in which case the grids spanned by each dictionary in the list are explored. This enables searching over any sequence of parameter settings.

This makes sense e.g. when some parameter configurations are not valid (e.g. 'sag' & 'l1' for LogisticRegression()):
```python
param_grid = [
    {'classifier__penalty' : ['l1'],
    'classifier__C' : np.logspace(-4, 4, 10),
    'classifier__solver' : ['liblinear', 'saga']},
    
    {'classifier__penalty' : ['l2'],
    'classifier__C' : np.logspace(-4, 4, 10),
    'classifier__solver' : ['lbfgs', 'liblinear', 'sag', 'saga']}
]
```

Or when you want to compare different models:
```python
pipe = Pipeline([('classifier' , LogisticRegression())])

param_grid = [    
    {'classifier' : [LogisticRegression()],
    'classifier__penalty' : ['l2'],
    'classifier__C' : np.logspace(-4, 4, 10),
    'classifier__solver' : ['lbfgs', 'liblinear', 'sag', 'saga']},
    
    {'classifier' : [RandomForestClassifier()],
    'classifier__n_estimators' : list(range(50,101,10))},
    'classifier__max_features' : ["auto", "log2"]},
    'classifier__criterion' : ['gini', 'entropy']},
    'classifier__oob_score' : [True, False]},

    {'classifier' : [KNeighborsClassifier()],
    'classifier_n_neighbors': np.arange(3, 15),
    'classifier_weights': ['uniform', 'distance'],
    'classifier_algorithm': ['ball_tree', 'kd_tree', 'brute']}
]
```


