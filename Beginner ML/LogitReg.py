import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV # facilitate hyper parameter tuning
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# for every hyper parameter combinateion, print the accuracy score
# and the standar deviation for the accuracy scores
def print_results(results):
    print('Best Params: {}\n'.format(results.best_params_))

    means = results.cv_results_['mean_test_score']
    stds = results.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, results.cv_results_['params']):
        print('{} (+/-{}) for {}'.format(round(mean, 3), round(std *2, 3), params))

# we will focus on one hyper parameter: C
# how to view all attributes and methods of an object
#print(dir(LogisticRegression))

tr_features = pd.read_csv('X_train.csv')
tr_labels = pd.read_csv('y_train.csv')

lr = LogisticRegression()
parameters = {'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000]}

# model objectn, parameter dictionary, cross validation folds
cv = GridSearchCV(lr, parameters, cv=5)
cv.fit(tr_features, tr_labels.values.ravel()) # convert labels from column vector type to array

print_results(cv)

# best fit model
print(cv.best_estimator_)

# write out pickled model
joblib.dump(cv.best_estimator_, 'LR_model.pkl')