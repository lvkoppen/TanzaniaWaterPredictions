from category_encoders.one_hot import OneHotEncoder
from numpy.lib.function_base import average
import pandas as pd
import os
import numpy as np
import category_encoders as ce
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, roc_auc_score, f1_score, accuracy_score, precision_score, recall_score, matthews_corrcoef
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import timeit
import pprint

pp = pprint.PrettyPrinter(indent=2)

def main():
    #get directory where data is located as path string
    working_directory = os.path.split(os.getcwd())[0]
    general_directory = os.path.split(working_directory)[0]
    data_location = os.path.join(general_directory, "data")

    prepped_data_folder = os.path.join(data_location, "prepped_data")

    #declaring data file names
    training_labels = "trainingsetlabels" + ".csv"
    prepped_file = '1_scenario_data' + ".csv"

    #get path to prepared training set data
    prepped_data_file_location = os.path.join(prepped_data_folder, prepped_file)

    #import data
    traindata = pd.read_csv(prepped_data_file_location)
    labeldata = pd.read_csv(os.path.join(data_location, training_labels))

    values_with_labels = pd.merge(left=traindata, right=labeldata, left_on = "id", right_on= "id")

    #drop id, since this was only neccesary to combine the two datasets
    values_with_labels.drop(columns=['id'], inplace= True)
    #drop missing numerical values
    values_with_labels.dropna(subset= ['gps_height', 'longitude', 'latitude', 'construction_year'], inplace= True)

    values_with_labels['district_code'] = pd.Categorical(values_with_labels.district_code)
    values_with_labels['status_group'] = pd.Categorical(values_with_labels.status_group)


    for col in values_with_labels.select_dtypes(include= object):
        values_with_labels[col] = pd.Categorical(values_with_labels[col])


    labels = values_with_labels.pop('status_group')


    labels = LabelEncoder().fit_transform(labels)

    #define columns used for specific pipelines
    low_cardinal_cols = ['basin', 'region', 'scheme_management', 'extraction_type', 'payment', 'water_quality', 'source', 'waterpoint_type']
    high_cardinal_cols = ['funder', 'installer', 'subvillage', 'lga', 'ward']
    quantity_cols = ['quantity']

    feature_skew = values_with_labels.select_dtypes(include=[np.number]).skew()

    log_features = feature_skew[abs(feature_skew)>0.9].index
    scale_features = [name for name in feature_skew.index if name not in log_features]



    encoders = {
    'BackwardDifferenceEncoder': ce.backward_difference.BackwardDifferenceEncoder,
    'BaseNEncoder': ce.basen.BaseNEncoder,
    'BinaryEncoder': ce.binary.BinaryEncoder,
    'CatBoostEncoder': ce.cat_boost.CatBoostEncoder,
    'HashingEncoder': ce.hashing.HashingEncoder,
    'HelmertEncoder': ce.helmert.HelmertEncoder,
    'JamesSteinEncoder': ce.james_stein.JamesSteinEncoder,
    
    'MEstimateEncoder': ce.m_estimate.MEstimateEncoder,
    
    'SumEncoder': ce.sum_coding.SumEncoder,
    'TargetEncoder': ce.target_encoder.TargetEncoder,
 
    }
    
    #'OrdinalEncoder': ce.ordinal.OrdinalEncoder,
    #'PolynomialEncoder': ce.polynomial.PolynomialEncoder,
    #'WOEEncoder': ce.woe.WOEEncoder

    X_train, X_test, y_train, y_test = train_test_split(values_with_labels, labels, test_size=0.3, random_state=1)


    #pipeline to figure out best encoder
    selected_model = RandomForestClassifier(random_state=42)

    df_results = pd.DataFrame()


    def model_prep_train_pipeline(encoder, model):    
        start_time = timeit.default_timer()
        
        nom_low_card_cat_pipeline = Pipeline(
            steps =[
                ('imputer', SimpleImputer(strategy='constant', fill_value= 'unknown')),
                ('encoder', encoders[encoder]())
        ])

        nom_high_card_cat_pipeline = Pipeline(
            steps =[
                ('imputer', SimpleImputer(strategy='constant', fill_value= 'unknown')),
                ('encoder', encoders[encoder]())
        ])

        quantity_pipeline = Pipeline(
             steps=[
                 ('onehot', OneHotEncoder())
        ])

        log_pipeline = Pipeline([
            ("log", FunctionTransformer(np.log1p, validate=False)),
            ('scale', StandardScaler())
        ])

        scale_pipeline = Pipeline([
            ("scale", StandardScaler())
        ])
        #full preprocessor pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ("nominal_low_card", nom_low_card_cat_pipeline, low_cardinal_cols),
                ("nominal_high_card", nom_high_card_cat_pipeline, high_cardinal_cols),
                ("quantity", quantity_pipeline, quantity_cols),
                ("log", log_pipeline, log_features),
                ("scale", scale_pipeline, scale_features)
            ], remainder='passthrough'
        )

        pipe = Pipeline(
            steps=[
                ("preprocessor",preprocessor),
                ('classifier', model)
            ]
        )
        print("running pipe with {}".format(key))
        model = pipe.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        end_time = timeit.default_timer()

        row = {
        'model' : type(model),
        'encoder': encoder,
        'accuracy': round(accuracy_score(y_test, y_pred), 3),
        'MCC' : round(matthews_corrcoef(y_test, y_pred), 3),
        'f1': round(f1_score(y_test, y_pred, average='macro'), 3),
        'precision' : round(precision_score(y_test, y_pred, average= 'macro'), 3),
        'recall': round(recall_score(y_test, y_pred, average= 'weighted'), 3),
        'time' : round(end_time - start_time, 3)
        }

        return (model, row)

    for key in encoders:
        results = model_prep_train_pipeline(key,selected_model)
        df_results.append(results[1], ignore_index=True)

    df_results = df_results[['model','encoder','accuracy', 'MCC','f1','precision','recall','time']].sort_values(by='accuracy', ascending=False).reset_index(drop=True)
    print(df_results.to_latex())
    pp.pprint(df_results)




    RSEED = 50
    print("splitting data")


    train, test, train_labels, test_labels = train_test_split(values_with_labels,
                                            labels,
                                            stratify = labels,
                                            test_size = 0.3,
                                            random_state = RSEED)
    #create model
    def RandomForest_Create_Train():
        model = RandomForestClassifier(random_state=42, oob_score=True)
        model.fit(train, train_labels)
        test_pred = model.predict(test)

        print(classification_report(y_true=test_labels, y_pred=test_pred, digits=3))


        disp = ConfusionMatrixDisplay.from_predictions(y_true=test_labels, y_pred=test_pred, display_labels=model.classes_, cmap="plasma", normalize="true")
        disp.plot()
        return model

    def imp_df(column_names, importances):
        df = pd.DataFrame(
            {
                'feature': column_names,
                'feature_importance': importances
            }).sort_values('feature_importance', ascending = False).reset_index(drop = True)
        return df

    def feature_importance(model):
        model_clone = clone(model)

        # training and scoring the benchmark model
        model_clone.fit(train, train_labels)
        benchmark_score = model_clone.score(train, train_labels)
        # list for storing feature importances
        importances = []

        # iterating over all columns and storing feature importance (difference between benchmark and new model)
        for col in train.columns:
            print("running model without {}".format(col))
            model_clone = clone(model)
            model_clone.random_state = 42
            model_clone.fit(train.drop(col, axis = 1), train_labels)
            drop_col_score = model_clone.score(train.drop(col, axis = 1), train_labels)
            importances.append(benchmark_score - drop_col_score)
            print(drop_col_score)

        importances_df = imp_df(train.columns, importances)
        importances_df.plot(kind='barh').set(xlabel="Permutation Importance Score")
        plt.show()
        return importances_df


if __name__== "__main__":
    main()