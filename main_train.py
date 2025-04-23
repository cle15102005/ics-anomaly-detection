import data_loader, ocsvm, main_eval

if __name__ == '__main__':
    """
    Load data and preprocessing
    x_train: contains only normal data, used for training model 
    x_test: contains both normal and abnormal data, used for prediction
    y_test= (0 if normal else 1)
    x_pred: predicted x
    y_pred= (1 if normal else -1)
    """
    x_train, _ = data_loader.load_train_data("BATADAL")
    x_test, y_test, _ = data_loader.load_test_data("BATADAL")      
    #Create model
    params = {
        'kernel' : 'rbf',   #determines the decision boundary shape.
        'gamma' : 'auto',   #determines the influence of a single training point reaches.
        'nu' : 0.01,        #determines the sensitivity to anomalies
        'verbose' : 0       #...
    }
    ocsvm_model = ocsvm.OCSVM(**params)

    #Train model
    ocsvm_model.train(x_train) 

    #Predic model
    y_pred =  ocsvm_model.predict(x_test)
    
    #Evaluate model
    matrix = main_eval.get_confusion_matrix(y_test, y_pred)
    #main_eval.show_confusion_matrix(matrix)
    main_eval.plot_evaluation(y_test, y_pred)