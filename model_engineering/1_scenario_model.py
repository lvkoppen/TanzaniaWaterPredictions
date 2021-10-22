import pandas as pd
import os
import sys
import numpy as np
import category_encoders as ce
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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

    #encode categorical data
    print('creating encoders')
    nominal_encoder = ce.HashingEncoder(cols=['funder', 'installer', 'basin', 'subvillage', 'region','lga', 'ward', 'scheme_management', 'extraction_type', 'payment', 'water_quality', 'source', 'waterpoint_type'])

    ordinal_encoder = ce.OrdinalEncoder(cols=["quantity"], return_df=True,
                                            mapping=[{'col': 'quantity',
                                            'mapping': {'unknown': 0, 'dry': 1, 'insufficient': 2, 'seasonal': 3, 'enough': 4}}])

    print("encoding nominal categories")
    df_nominal_transformed = nominal_encoder.fit_transform(values_with_labels)
    print("enconding ordinal categories")
    df__cat_clean = ordinal_encoder.fit_transform(df_nominal_transformed)



    #fill missing
    print("dropping NaN")
    df_clean = df__cat_clean.dropna(subset= ['gps_height', 'longitude', 'latitude', 'construction_year'])

    print(df_clean.shape)
    #split data into train and test set
    RSEED = 50
    print("splitting data")
    labels = np.array(df_clean.pop('status_group'))

    train, test, train_labels, test_labels = train_test_split(df_clean,
                                            labels, 
                                            stratify = labels,
                                            test_size = 0.3, 
                                            random_state = RSEED)

    print(train.shape)
    #create model
    print("training model")
    model = RandomForestClassifier(n_estimators=100, random_state=RSEED, max_features='sqrt',n_jobs=1,verbose=1)
    model.fit(train, train_labels)

    test_pred = model.predict(test)

    print('Accuracy: %.3f' % accuracy_score(test_labels, test_pred))

if __name__== "__main__":
    main()