from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import  OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
import numpy as np

def modelisation_avec_Random_Forest(X_train, y_train ,num_cols, cat_cols):
    
    preprocessor = ColumnTransformer(
    transformers=[('num', 'passthrough', num_cols),
                    ('cat' , OneHotEncoder(handle_unknown='ignore', sparse_output=False),cat_cols)])
    RFmodel = Pipeline([
        ('preprocessing', preprocessor),
        ('feature_selection',SelectFromModel(RandomForestRegressor(random_state = 42),threshold= 'mean')),
        ('RFR',RandomForestRegressor(random_state = 42))   
    ])        

    RFparams_grid = {
                'RFR__n_estimators': [50, 100, 200],
                'RFR__max_depth': [None, 10, 20, 30],
                'RFR__min_samples_split': [2, 5, 10],
                'RFR__min_samples_leaf': [1, 2, 4]
    }

    # gestion de Random Forest
    grid = RandomizedSearchCV(estimator=RFmodel,
        param_distributions=RFparams_grid,
        n_iter=10,cv=5,
    scoring='neg_root_mean_squared_error', random_state=42, n_jobs=-1,verbose=2
    )
    grid.fit(X_train,y_train)

    best_param = grid.best_params_
    model_best = grid.best_estimator_

    return model_best  # best_param 



def modelisation_avec_HBGR(X_train, y_train ,num_cols, cat_cols):
        
    preprocessor = ColumnTransformer(
    transformers=[('num', 'passthrough', num_cols),
                 ('cat' , OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),cat_cols)])

    HGBRmodel = Pipeline([
        ('preprocessing' , preprocessor),
        ('HGBR' , HistGradientBoostingRegressor(random_state = 42))
    ])

    HGBRparams_grid = {
                'HGBR__learning_rate': [0.01, 0.05, 0.1], # taux d'apprentissage (de 0.01 à 0.1)
                'HGBR__max_iter': [100, 200, 300], # nombre d'arbres (100 par défaut)
                'HGBR__max_depth': [6, 8], # complexité des arbres            
     }

    grid = RandomizedSearchCV(estimator=HGBRmodel,
        param_distributions=HGBRparams_grid,
        n_iter=10,cv=5, random_state=42, n_jobs=-1,verbose=2
    )
    grid.fit(X_train,y_train)
    best_param = grid.best_params_
    model_best = grid.best_estimator_

    return model_best # best_param 



def resultat_model(model_best, X_test, y_test):

    y_pred = model_best.predict(X_test) 
    if len(y_pred) == 0 :     
        return None, None, None  
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2_Random = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    return rmse, r2_Random, mae, y_pred



