# ~~~~~~~~~~~~~~ Creating a pipeline to get prediction -

# Creating pipeline for the categorical features - 
categorical_transformer = Pipeline(steps = [("ohe", OneHotEncoder(handle_unknown = "ignore"))])

# Creating the column transformer to apply it to the new data - 
preprocessing_pipeline = ColumnTransformer(transformers = [("categorical", categorical_transformer, categorical_features)])

# getting the tuned rf model from the grid search -
rf_regressor = rf_gscv.best_estimator_ 

# Instantiating the pipeline with the best parameters of the model -
pipeline_regressor = Pipeline(steps = [("preprocessing_pipeline", preprocessing_pipeline),
                        ("regressor", RandomForestRegressor(n_estimators = rf_best_param['n_estimators'],
                                                            max_depth = rf_best_param['max_depth'], random_state=42))])

# getting the data in the original state - 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Fitting the model -
pipeline_regressor.fit(X_train, y_train)

# Making prediction
pipeline_regressor.predict(X_trial)



