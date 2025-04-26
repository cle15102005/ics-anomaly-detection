import data_loader, main_eval, ocsvm, rf
def OCSVM_detect():
    """
    Load data and preprocessing
    x_train: contains only normal data, used for training model 
    x_test: contains both normal and abnormal data, used for prediction
    y_test= (0 if normal else 1)
    x_pred: predicted x
    y_pred= (0 if normal else 1)
    """
    x_train, _ = data_loader.load_train_data("BATADAL")
    x_test, y_test, _ = data_loader.load_test_data("BATADAL")    
    
    #Create model
    params = {
            'kernel' : 'rbf',   #mode: rbf / linear / poly / sigmoid
            'gamma' : 'auto',   #mode: scale / auto
            'nu' : 0.01,        #value = [0.01, 0.02, 0.03, ..., 0,1]     
    }
    ocsvm_model = ocsvm.OCSVM(**params)

    #Train model
    ocsvm_model.train(x_train) 

    #Predic model
    y_pred =  ocsvm_model.predict(x_test)
    return y_test, y_pred

def RF_detect():
    # Load training and test data (FULLY labeled)
    x_train, y_train, _ = data_loader.load_train_data_2("BATADAL", no_transform=False)
    x_test, y_test, _ = data_loader.load_test_data("BATADAL", no_transform=False)

    # Create model
    params = {
        'n_estimators': 100,
        'max_depth': None,
        'random_state': 42,
        'verbose': 0
    }
    rf_model = rf.RandomForestModel(**params)

    # Train model
    rf_model.train(x_train, y_train)

    # Predict
    y_pred = rf_model.predict(x_test)
    return y_test, y_pred
    
if __name__ == '__main__':
    model_name = input("Enter model name (OCSVM/RF/HYBRID): ")
    if model_name == 'OCSVM':
        y_test, y_pred = OCSVM_detect()
    elif model_name == 'RF':
        y_test, y_pred = RF_detect()
    elif model_name == 'HYBRID':
        y_test, ocsvm_pred = OCSVM_detect()
        _, rf_pred = RF_detect()
        y_pred = ((rf_pred + ocsvm_pred) >= 1).astype(int)
        
    #Evaluate model    
    #main_eval.show_confusion_matrix(matrix)
    main_eval.plot_evaluation(model_name, y_test, y_pred)