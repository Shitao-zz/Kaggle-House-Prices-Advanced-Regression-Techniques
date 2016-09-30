import datetime
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
 
def mean_squared_error_(ground_truth, predictions):
    return mean_squared_error(ground_truth, predictions) ** 0.5
RMSE = make_scorer(mean_squared_error_, greater_is_better=False)    
    
def create_submission(prediction,score):
    now = datetime.datetime.now()
    sub_file = 'submission_'+str(score)+'_'+str(now.strftime("%Y-%m-%d-%H-%M"))+'.csv'
    print ('Creating submission: ', sub_file)
    pd.DataFrame({'Id': test['Id'].values, 'SalePrice': prediction}).to_csv(sub_file, index=False)
    
def data_preprocess(train,test):
    tables = [train,test]
    print ("Delete features with high number of missing values...")
    total_missing = train.isnull().sum()
    to_delete = total_missing[total_missing>(train.shape[0]/3.)]
    for table in tables:
        table.drop(list(to_delete.index),axis=1,inplace=True)

    numerical_features = test.select_dtypes(include=["float","int","bool"]).columns.values
    categorical_features = train.select_dtypes(include=["object"]).columns.values
    print ("Filling all Nan...")
    for table in tables:
        for feature in numerical_features:
            table[feature].fillna(table[feature].median(),inplace=True)
        for feature in categorical_features:
            table[feature].fillna(table[feature].value_counts().idxmax(),inplace=True)

    print ("Handling categorical features...")
    for feature in categorical_features:
        le = preprocessing.LabelEncoder()
        le.fit(train[feature])
        for table in tables: 
            table[feature]=le.transform(table[feature])
  
    print ("Get features...")
    features = list(set(list(train.columns))&set(test.columns))
    features.remove('Id')
    
    print ("The size of the train set:",train.shape)
    print ("The size of the test set:",test.shape)
    
    return train,test,features


def model_random_forecast(train,test,features):
    
    X_train = train[features]
    y_train = np.log(train['SalePrice'])
    rfr = RandomForestRegressor(n_jobs=1, random_state=0)
    param_grid = {'n_estimators': [500], 'max_features': [10, 12, 14]}
    model = GridSearchCV(estimator=rfr, param_grid=param_grid, n_jobs=1, cv=10, scoring=RMSE)
    model.fit(X_train, y_train)
    print('Random forecast regression...')
    print('Best Params:')
    print(model.best_params_)
    print('Best CV Score:')
    print(-model.best_score_)

    y_pred = model.predict(test[features])
    return y_pred, -model.best_score_


# In[24]:

def model_gradient_boosting_tree(train,test,features):
    
    X_train = train[features]
    y_train = np.log(train['SalePrice'])
    gbr = GradientBoostingRegressor(random_state=0)
    param_grid = {
        'n_estimators': [500],
        'max_features': [10,15],
	'max_depth': [6,8,10],
        'learning_rate': [0.05,0.1,0.15],
        'subsample': [0.8]
    }
    model = GridSearchCV(estimator=gbr, param_grid=param_grid, n_jobs=1, cv=10, scoring=RMSE)
    model.fit(X_train, y_train)
    print('Gradient boosted tree regression...')
    print('Best Params:')
    print(model.best_params_)
    print('Best CV Score:')
    print(-model.best_score_)

    y_pred = model.predict(test[features])
    return y_pred, -model.best_score_

def model_xgb_regression(train,test,features):
    
    X_train = train[features]
    y_train = np.log(train['SalePrice'])
    
    xgbreg = xgb.XGBRegressor(seed=0)
    param_grid = {
        'n_estimators': [500],
        'learning_rate': [ 0.05],
        'max_depth': [ 7, 9, 11],
        'subsample': [ 0.8],
        'colsample_bytree': [0.75,0.8,0.85],
    }
    model = GridSearchCV(estimator=xgbreg, param_grid=param_grid, n_jobs=1, cv=10, scoring=RMSE)
    model.fit(X_train, y_train)
    print('eXtreme Gradient Boosting regression...')
    print('Best Params:')
    print(model.best_params_)
    print('Best CV Score:')
    print(-model.best_score_)

    y_pred = model.predict(test[features])
    return y_pred, -model.best_score_

def model_extra_trees_regression(train,test,features):
    
    X_train = train[features]
    y_train = np.log(train['SalePrice'])
    
    etr = ExtraTreesRegressor(n_jobs=1, random_state=0)
    param_grid = {'n_estimators': [500], 'max_features': [10,15,20]}
    model = GridSearchCV(estimator=etr, param_grid=param_grid, n_jobs=1, cv=10, scoring=RMSE)
    model.fit(X_train, y_train)
    print('Extra trees regression...')
    print('Best Params:')
    print(model.best_params_)
    print('Best CV Score:')
    print(-model.best_score_)

    y_pred = model.predict(test[features])
    return y_pred, -model.best_score_

def model_kernel_ridge_regression(train,test,features):
    
    X_train = train[features]
    y_train = np.log(train['SalePrice'])
    
    etr = KernelRidge(kernel = 'linear')
    param_grid = {'alpha': [1e-2,0.1,0.3,0.5,1,2,4], 'gamma': np.logspace(-3,2,6)}
    model = GridSearchCV(estimator=etr, param_grid=param_grid, n_jobs=1, cv=10, scoring=RMSE)
    model.fit(X_train, y_train)
    print('Kernel ridge regression...')
    print('Best Params:')
    print(model.best_params_)
    print('Best CV Score:')
    print(-model.best_score_)

    y_pred = model.predict(test[features])
    return y_pred, -model.best_score_

def model_KNN_regression(train,test,features):
    
    X_train = train[features]
    y_train = np.log(train['SalePrice'])
    
    etr = KNeighborsRegressor()
    param_grid = {'n_neighbors': [3,5,8,10,15]}
    model = GridSearchCV(estimator=etr, param_grid=param_grid, n_jobs=1, cv=10, scoring=RMSE)
    model.fit(X_train, y_train)
    print('KNN regression...')
    print('Best Params:')
    print(model.best_params_)
    print('Best CV Score:')
    print(-model.best_score_)

    y_pred = model.predict(test[features])
    return y_pred, -model.best_score_


# In[ ]:

# read data, build model and do prediction
train = pd.read_csv("../input/train.csv") # read train data
test = pd.read_csv("../input/test.csv") # read test data
train,test,features = data_preprocess(train,test)


#test_predict,score = model_random_forecast(train,test,features)
#test_predict,score = model_xgb_regression(train,test,features)
#test_predict,score = model_extra_trees_regression(train,test,features)
#test_predict,score = model_gradient_boosting_tree(train,test,features)
test_predict,score = model_KNN_regression(train,test,features)


#create_submission(np.exp(test_predict),score)



