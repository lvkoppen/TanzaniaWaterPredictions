from category_encoders.one_hot import OneHotEncoder
from numpy.lib.function_base import average
import pandas as pd
import os
import numpy as np
import category_encoders as ce
from sklearn import preprocessing
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
import seaborn as sns

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

    low_cardinal_features = ['basin', 'region', 'scheme_management', 'extraction_type', 'payment', 'water_quality', 'source', 'waterpoint_type']
    high_cardinal_features = ['funder', 'installer', 'subvillage', 'lga', 'ward']
    quantity_features = ['quantity']

    feature_skew = values_with_labels.select_dtypes(include=[np.number]).skew()

    log_features = feature_skew[abs(feature_skew)>0.9].index
    log_features = list(log_features.values)
    scale_features = [name for name in feature_skew.index if name not in log_features]



    encoders = {
    'BaseNEncoder': ce.basen.BaseNEncoder,
    'BinaryEncoder': ce.binary.BinaryEncoder,
    'CatBoostEncoder': ce.cat_boost.CatBoostEncoder,   
    'JamesSteinEncoder': ce.james_stein.JamesSteinEncoder,
    'MEstimateEncoder': ce.m_estimate.MEstimateEncoder,
    'TargetEncoder': ce.target_encoder.TargetEncoder,
    }
    #'HelmertEncoder': ce.helmert.HelmertEncoder,
    #'HashingEncoder': ce.hashing.HashingEncoder,
    #'BackwardDifferenceEncoder': ce.backward_difference.BackwardDifferenceEncoder,
    #'SumEncoder': ce.sum_coding.SumEncoder,
    #'OrdinalEncoder': ce.ordinal.OrdinalEncoder,
    #'PolynomialEncoder': ce.polynomial.PolynomialEncoder,
    #'WOEEncoder': ce.woe.WOEEncoder

    X_train, X_test, y_train, y_test = train_test_split(values_with_labels, labels, test_size=0.3, random_state=1)


    #create model
    selected_model = RandomForestClassifier(random_state=42)

    


    def preprocessor_pipeline(encoder, columns=None):
        scale_cols = scale_features.copy()
        log_cols = log_features.copy()
        low_card_cols = low_cardinal_features.copy()
        high_card_cols = high_cardinal_features.copy()
        quantity_col = quantity_features.copy()
        if columns is not None:

            for x in low_card_cols:
                if x not in columns:
                    low_card_cols.remove(x)

            for x in high_card_cols:
                if x not in columns:
                    high_card_cols.remove(x)

            for x in quantity_col:
                if x not in columns:
                    quantity_col.remove(x)

            for x in log_features:
                if x not in columns:
                    log_cols.remove(x)

            for x in scale_features:
                if x not in columns:
                    scale_cols.remove(x)



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
                ("nominal_low_card", nom_low_card_cat_pipeline, low_card_cols),
                ("nominal_high_card", nom_high_card_cat_pipeline, high_card_cols),
                ("quantity", quantity_pipeline, quantity_col),
                ("log", log_pipeline, log_cols),
                ("scale", scale_pipeline, scale_cols)
            ])

        return preprocessor

    def train_score_model(preprocessor, model, confusion_matrix = False):
        pipe = Pipeline(
            steps=[
                ("preprocessor",preprocessor),
                ('classifier', model)
            ]
        )
        model = pipe.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        row = {
        'accuracy': accuracy_score(y_test, y_pred),
        'MCC' : matthews_corrcoef(y_test, y_pred),
        'f1': f1_score(y_test, y_pred, average='macro'),
        'precision' : precision_score(y_test, y_pred, average= 'macro'),
        'recall': recall_score(y_test, y_pred, average= 'weighted'),
        }


        if confusion_matrix is True:
            create_confusion_matrix(y_test=y_test, y_pred=y_pred, classes=['functional', 'needs repair', 'non functional'])
        return row
    
    def compare_encodings():
        df_results = pd.DataFrame()
    
        for key in encoders:
            print('running pipe with {}'.format(key))
            start_time = timeit.default_timer()
            preprocessing = preprocessor_pipeline(key)
            results = train_score_model(preprocessing,selected_model)
            end_time = timeit.default_timer()
            results['time'] = end_time - start_time
            pp.pprint(results)
            df_results = df_results.append(results, ignore_index=True)
        pp.pprint(df_results)
        df_results = df_results[['encoder','accuracy', 'MCC','f1','precision','recall','time']].sort_values(by='accuracy', ascending=False).reset_index(drop=True)
        print(df_results.to_latex())
        pp.pprint(df_results)


    def create_confusion_matrix(y_test, y_pred, classes):
        disp = ConfusionMatrixDisplay.from_predictions(y_true=y_test, y_pred=y_pred, display_labels=classes, cmap="plasma", normalize="true")
        disp.plot()
        plt.show()

    def imp_df(column_names, importances):
        df = pd.DataFrame(
            {
                'feature': column_names,
                'feature_importance': importances
            }).sort_values('feature_importance', ascending = True).reset_index(drop = True)
        df['feature_importance'] = df['feature_importance'].apply(lambda x: x*100)
        return df

    def feature_importance(selected_model):
        prep = preprocessor_pipeline('CatBoostEncoder')
        model_clone = clone(selected_model)

        # training and scoring the benchmark model
        benchmark_score = train_score_model(prep, model_clone)

        
        # list for storing feature importances
        importances = []

        # iterating over all columns and storing feature importance (difference between benchmark and new model)
        for col in X_train.columns:
            print("running model without {}".format(col))
            columns = list(X_train.columns.values)
            columns.remove(col)
            
            preperation = preprocessor_pipeline('CatBoostEncoder', columns= columns)
            model_clone = clone(selected_model)

            scoring = train_score_model(preperation, model_clone, False)

            importance_score = benchmark_score['accuracy'] - scoring['accuracy']
            importances.append(importance_score)


        importances_df = imp_df(list(X_train.columns.values), importances)
        sns.set_theme()
        sns.barplot(x="feature_importance",y='feature', data=importances_df, palette= 'viridis')
        plt.show()
        return importances_df


    pp.pprint(feature_importance(selected_model))



if __name__== "__main__":
    main()