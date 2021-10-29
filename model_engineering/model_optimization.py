import pandas as pd
import os
import sys
import numpy as np
import category_encoders as ce
from scipy.sparse.construct import random
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold, GridSearchCV
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve, accuracy_score, f1_score, confusion_matrix, classification_report
from statistics import mean, stdev
import matplotlib.pyplot as plt
from rfpimp import *

def main():
    #get directory where data is located as path string
    working_directory = os.path.split(os.getcwd())[0]
    general_directory = os.path.split(working_directory)[0]
    data_location = os.path.join(general_directory, "data")

    prepped_data_folder = os.path.join(data_location, "prepped_data")

    #declaring data file names
    dataset =  "trainingsetvalues" +".csv"
    training_labels = "trainingsetlabels" + ".csv"
    test_data = 'testsetvalues.csv'
    prepped_file = '1_scenario_data' + ".csv"


    #get path to prepared training set data
    prepped_data_file_location = os.path.join(prepped_data_folder, prepped_file)

    #import data
    traindata = pd.read_csv(prepped_data_file_location)
    labeldata = pd.read_csv(os.path.join(data_location, training_labels))

    values_with_labels = pd.merge(left=traindata, right=labeldata, left_on = "id", right_on= "id")
    
    #drop id, since this was only neccesary to combine the two datasets

    values_with_labels.drop(columns=['id'], inplace= True)
    low_cardinal_cols = ['basin', 'region', 'scheme_management', 'extraction_type', 'payment', 'water_quality', 'source', 'waterpoint_type']
    high_cardinal_cols = ['funder', 'installer', 'subvillage', 'lga', 'ward']
    #encode categorical data
    print('creating encoders')
    nominal_encoder_low_cardinality = ce.HashingEncoder(cols= low_cardinal_cols, max_process=4, return_df=True)

    nominal_encoder_high_cardinality = ce.HashingEncoder(cols= high_cardinal_cols,  n_components=16 , max_process=4, return_df=True)

    ordinal_encoder = ce.OrdinalEncoder(cols=["quantity"], return_df=True,
                                            mapping=[{'col': 'quantity',
                                            'mapping': {'unknown': 0, 'dry': 1, 'insufficient': 2, 'seasonal': 3, 'enough': 4}}])

    #label_encoder = ce.OrdinalEncoder(cols=["status_group"], return_df=True,
    #                                        mapping=[{'col': 'status_group',
    #                                       'mapping': {'functional': 0, 'non functional': 1, 'functional needs repair': 2}}])
    print("encoding nominal categories")
    df_nominal_transformed_partially = nominal_encoder_low_cardinality.fit_transform(values_with_labels)


    df_nominal_transformed_fully = nominal_encoder_high_cardinality.fit_transform(df_nominal_transformed_partially)

    print("enconding ordinal categories")
    df__cat_clean = ordinal_encoder.fit_transform(df_nominal_transformed_fully)

    

    #fill missing
    df_clean = df__cat_clean.dropna(subset= ['gps_height', 'longitude', 'latitude', 'construction_year'])


    #split data into train and test set
    RSEED = 50
    print("splitting data")
    labels = np.array(df_clean.pop('status_group'))

    train, test, train_labels, test_labels = train_test_split(df_clean,
                                            labels, 
                                            stratify = labels,
                                            test_size = 0.3, 
                                            random_state = RSEED)

    def RandomForest_Create_Train():
        model = RandomForestClassifier(random_state=42)
        model.fit(train, train_labels)
        test_pred = model.predict(test)

        print(classification_report(y_true=test_labels, y_pred=test_pred, digits=3))


        disp = ConfusionMatrixDisplay.from_predictions(y_true=test_labels, y_pred=test_pred, display_labels=model.classes_, cmap="plasma", normalize="true")
        disp.plot()
        plt.show()
        return model

    x = RandomForest_Create_Train()

    def nested_cross_evaluation_gridsearch():
        cv_outer = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
        outer_results = list()
        cv_outer.get_n_splits(df_clean, labels)
        for train_index, test_index in cv_outer.split(df_clean, labels):
            # split data
            X_train, X_test = df_clean.iloc[train_index], df_clean.iloc[test_index]
            y_train, y_test = labels[train_index], labels[test_index]
            # configure the cross-validation procedure
            cv_inner = KFold(n_splits=3, shuffle=True, random_state=1)
            # define the model
            model = RandomForestClassifier(random_state=RSEED, n_jobs=6)
            # define search space
            space = dict()
            space['n_estimators'] = [10, 100, 500]
            space['max_features'] = ["auto", "sqrt", 'log2']
            space['criterion'] = ['gini', 'entropy']
            space['max_depth'] = [ 3,4,5,6,7,8]
            # define search
            search = GridSearchCV(model, space, scoring='accuracy', cv=cv_inner, refit=True)
            # execute search
            result = search.fit(X_train, y_train)
            # get the best performing model fit on the whole training set
            best_model = result.best_estimator_
            # evaluate model on the hold out dataset
            yhat = best_model.predict(X_test)
            # evaluate the model
            acc = accuracy_score(y_test, yhat)
            # store the result
            outer_results.append(acc)
            # report progress
            print('>acc=%.3f, est=%.3f, cfg=%s' % (acc, result.best_score_, result.best_params_))
        # summarize the estimated performance of the model
        print('Accuracy: %.3f (%.3f)' % (np.mean(outer_results), np.std(outer_results)))



if __name__== "__main__":
    main()
    
    
    

