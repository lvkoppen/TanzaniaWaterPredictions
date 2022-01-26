import pandas as pd
import os
import numpy as np
import category_encoders as ce
from scipy.sparse.construct import random
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import ConfusionMatrixDisplay, precision_score, recall_score,  accuracy_score, f1_score, matthews_corrcoef, make_scorer
from statistics import mean, stdev
import matplotlib.pyplot as plt
from optbinning import BinningProcess
import pprint
from tqdm import tqdm

pp = pprint.PrettyPrinter(indent=2)

def get_data():
    #get directory where data is located as path string
    working_directory = os.path.split(os.getcwd())[0]
    general_directory = os.path.split(working_directory)[0]
    data_location = os.path.join(general_directory, "data")

    prepped_data_folder = os.path.join(data_location, "prepped_data")

    #declaring data file names
    training_labels = "trainingsetlabels" + ".csv"
    prepped_file = '4_scenario_data' + ".csv"

    #get path to prepared training set data
    prepped_data_file_location = os.path.join(prepped_data_folder, prepped_file)

    #import data
    traindata = pd.read_csv(prepped_data_file_location)
    labeldata = pd.read_csv(os.path.join(data_location, training_labels))

    values_with_labels = pd.merge(left=traindata, right=labeldata, left_on = "id", right_on= "id")

    values_with_labels.dropna(subset= ['gps_height', 'longitude', 'latitude', 'construction_year'], inplace= True)

    #values_with_labels['district_code'] = pd.Categorical(values_with_labels.district_code)
    values_with_labels['status_group'] = pd.Categorical(values_with_labels.status_group)



    #transform object dtype to category dtype
    for col in values_with_labels.select_dtypes(include= object):
        values_with_labels[col] = pd.Categorical(values_with_labels[col])
    
    return values_with_labels.copy()

def drop_columns(df_undropped, standard, abstraction, extra):
    df = df_undropped.copy()
    #standard columns to drop
    if standard:
        df.drop(columns=standard, inplace= True)

    #abstraction columns
    if abstraction:
        df.drop(columns=abstraction, inplace= True)

    #drop for feature selection
    if extra:
        df.drop(columns=extra, inplace=True)

    #drop id, since this was only neccesary to combine the two datasets
    df.drop(columns=['id'], inplace= True)


    print(df.columns.values)
    print(df.shape)
    return df

def split_data(df_clean, labels):
    X_train, X_test, y_train, y_test = train_test_split(df_clean, labels, test_size=0.3, stratify = labels, random_state=42)
    return X_train, X_test, y_train, y_test

def main(df):
    df_unclean = df
    #BASIC TO DROP LISTS DO NOT EDIT
    #standard = ['wpt_name', 'public_meeting',"num_private", 'recorded_by', 'permit','scheme_name',
    # 'payment_type', 'quantity_group','scheme_management', 'date_recorded']

    
    # extra = [ ]
    # abs_list = ['source_extraction_type', 'region', 'waterpoint_type_group', 'management_group', 'water_quality', 'waterpoint_age']

    #v.123 source class, public meeting and date recorded
    standard = ['wpt_name', 'public_meeting',"num_private", 'recorded_by', 'permit','scheme_name',
     'payment_type', 'quantity_group','scheme_management', 'date_recorded']

        #list of columns to be dropped
    extra = ['waterpoint_type', 'extraction_type']
    abs_list = []


    abstraction_dict = {"region": ['region', 'region_code'],
                        "extraction": ['extraction_type', 'extraction_type_group', 'extraction_type_class', 'source_extraction_type'],
                        "management": ['management','management_group'],
                        "source": ['source', 'source_type', 'source_class', 'source_extraction_type'],
                        "waterpoint_type": ['waterpoint_type','waterpoint_type_group'],
                        "quality" : ['water_quality', 'quality_group'],
                        'age' : ['waterpoint_age', 'construction_year']}
                        

    abstraction = []



    def set_abstraction_cols(used_abs_list):
        abstracts = []
        if used_abs_list:
            for key, value in abstraction_dict.items():
                abstracts = (abstracts + [x for x in value if x not in used_abs_list])
        return abstracts.copy()

    abstraction = set_abstraction_cols(abs_list)

    #drop columns
    df_clean = drop_columns(df_unclean,standard, abstraction, extra)

    labels = df_clean.pop('status_group')
    labels = LabelEncoder().fit_transform(labels)

    #define columns used for specific pipelines
    quantity_features = []

    cat_cols = list(df_clean.select_dtypes(include= ['category']))
    quantity_features.append(cat_cols.pop(cat_cols.index('quantity')))
    category_features = cat_cols
    
    feature_skew = df_clean.select_dtypes(include=[np.number]).skew()

    log_list = feature_skew[abs(feature_skew)>0.9].index
    log_features = list(log_list.values)
    scale_features = [name for name in feature_skew.index if name not in log_features]


    #list of to be used encoders for comparison
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

    print("shape of data used for training is: {}".format(df_clean.shape))
    
    def preprocessor_pipeline(encoder, columns=None, log_binning = True):
        scale_cols = scale_features.copy()
        log_cols = log_features.copy()
        category_cols = category_features.copy()
        quantity_col = quantity_features.copy()
        if columns is not None:

            for x in category_cols:
                if x not in columns:
                    category_cols.remove(x)

            for x in quantity_col:
                if x not in columns:
                    quantity_col.remove(x)

            for x in log_features:
                if x not in columns:
                    log_cols.remove(x)

            for x in scale_features:
                if x not in columns:
                    scale_cols.remove(x)



        category_pipeline = Pipeline(
            steps =[
                ('imputer', SimpleImputer(strategy='constant', fill_value= 'unknown')),
                ('encoder', encoders[encoder]())
        ])

        quantity_pipeline = Pipeline(
             steps=[
                 ('onehot', OneHotEncoder())
        ])

        log_pipeline = None

        binning_process = BinningProcess(log_cols)
        if(not log_binning):
            log_pipeline = Pipeline([
                ("log", FunctionTransformer(np.log1p, validate=False)),
                ('scale', StandardScaler())
            ])
        else:
            log_pipeline = Pipeline([
                ("log", FunctionTransformer(np.log1p, validate=False)),
                ('binning', binning_process)
            ])


        scale_pipeline = Pipeline([
            ("scale", StandardScaler())
        ])
        #full preprocessor pipeline
        preprocessor = None 
        

        preprocessor = ColumnTransformer(
            transformers=[
                ("nominal_low_card", category_pipeline, category_cols),
                ("quantity", quantity_pipeline, quantity_col),
                ("log", log_pipeline, log_cols),
                ("scale", scale_pipeline, scale_cols)
            ])

        return preprocessor

    def train_score_model(preprocessor, model, X_train, y_train, X_test, y_test, confusion_matrix = False):
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

        pp.pprint(row)
        if confusion_matrix is True:
            create_confusion_matrix(y_test=y_test, y_pred=y_pred, classes=['functional', 'needs repair', 'non functional'])
        return row

    def create_prep_train_pipe(preprocessor, model):
        pipe = Pipeline(
            steps=[
                ("preprocessor",preprocessor),
                ('classifier', model)
            ]
        )
        return pipe
    def create_confusion_matrix(y_test, y_pred, classes):
        disp = ConfusionMatrixDisplay.from_predictions(y_true=y_test, y_pred=y_pred, display_labels=classes, cmap="plasma", normalize="true")
        #disp.plot()
        plt.show()

    def nested_cross_evaluation_gridsearch(pipe):
        cv_outer = StratifiedKFold(n_splits=10, shuffle=True, random_state=4)
        outer_results = list()
        cv_outer.get_n_splits(df_clean, labels)
        for train_index, test_index in tqdm(cv_outer.split(df_clean, labels)):
            # split data
            X_train, X_test = df_clean.iloc[train_index], df_clean.iloc[test_index]
            y_train, y_test = labels[train_index], labels[test_index]
            # configure the cross-validation procedure
            cv_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            # define search space             
            space = [{'classifier__n_estimators': [1000],
                    'classifier__max_features' : ["auto"],
                    'classifier__max_depth' : [8,12],
                    'classifier__class_weight' : ['balanced'],
                    'classifier__criterion' : ['entropy', 'gini'],
                    'classifier' : [RandomForestClassifier()]
            }]
            #define scoring function
            scorer_function = make_scorer(matthews_corrcoef)
            # define search
            search = GridSearchCV(pipe, space, scoring=scorer_function, cv=cv_inner, n_jobs=8, verbose=1)
            # execute search
            result = search.fit(X_train, y_train)
            # get the best performing model fit on the whole training set
            best_model = result.best_estimator_
            # evaluate model on the hold out dataset
            yhat = best_model.predict(X_test)
            # evaluate the model
            acc = accuracy_score(y_test, yhat)
            mcc = matthews_corrcoef(y_test, yhat)
            f1 = f1_score(y_test, yhat, average='weighted')
            # store the result
            outer_results.append(mcc)
            # report progress
            print('>acc=%.3f, mcc=%.3f, f1=%.3f, est=%.3f, cfg=%s' % (acc, mcc,f1, result.best_score_, result.best_params_))
        # summarize the estimated performance of the model
        print('MCC: %.3f (%.3f)' % (np.mean(outer_results), np.std(outer_results)))

    model = RandomForestClassifier(random_state=42)
    encoder = 'BaseNEncoder'
    pip = preprocessor_pipeline(encoder, log_binning=True)
    train_pipe = create_prep_train_pipe(pip,model)

    nested_cross_evaluation_gridsearch(train_pipe)

if __name__== "__main__":
    df = get_data()
    main(df)
    
    
    

