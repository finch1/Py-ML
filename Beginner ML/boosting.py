from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV # facilitate hyper parameter tuning
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# we will focus on one hyper parameter: C
# how to view all attributes and methods of an object
print(GradientBoostingClassifier())
print(GradientBoostingRegressor())

# for every hyper parameter combinateion, print the accuracy score
# and the standar deviation for the accuracy scores
def print_results(results):
    print('Best Params: {}\n'.format(results.best_params_))

    means = results.cv_results_['mean_test_score']
    stds = results.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, results.cv_results_['params']):
        print('{} (+/-{}) for {}'.format(round(mean, 3), round(std *2, 3), params))


tr_features = pd.read_csv('X_train.csv')
tr_labels = pd.read_csv('y_train.csv')

gb = GradientBoostingClassifier()
parameters = {'n_estimators':[5, 50, 250, 500],
              'max_depth':[1, 3, 5, 7, 9],
              'learning_rate':[0.01, 0.1, 1, 10, 100]} # none will let the tree grow as much as it wants until it still makes sense

# model objectn, parameter dictionary, cross validation folds
cv = GridSearchCV(gb, parameters, cv=5)
cv.fit(tr_features, tr_labels.values.ravel()) # convert labels from column vector type to array

print_results(cv)

# best fit model
print(cv.best_estimator_)

# write out pickled model
joblib.dump(cv.best_estimator_, 'GB_model.pkl')