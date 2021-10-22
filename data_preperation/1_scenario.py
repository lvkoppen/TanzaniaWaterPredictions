import pandas as pd
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno


working_directory = os.path.split(os.getcwd())[0]

general_directory = os.path.split(working_directory)[0]


data_location = os.path.join(general_directory, "data")

prepped_data_folder = os.path.join(data_location, "prepped_data")

dataset =  "trainingsetvalues" +".csv"


prepped_file = '1_scenario_data' + ".csv"

prepped_data_file_location = os.path.join(prepped_data_folder, prepped_file)

df = pd.read_csv(os.path.join(data_location, dataset))

#standard columns to drop
df.drop(columns=['wpt_name', 'public_meeting',"num_private", 'recorded_by',
                'permit','scheme_name','payment_type', 'quantity_group'], inplace= True)


#abstraction columns
df.drop(columns=['region_code', 'date_recorded','extraction_type_group','extraction_type_class','quality_group',
                'management','management_group','source_type', 'source_class', 'waterpoint_type_group'], inplace= True)


#replacing missing values with NaN type and incorrect values(zeroes)
df['construction_year'].replace(0, np.nan, inplace= True)

df['longitude'].replace(0, np.nan, inplace= True)

df['latitude'] = df['latitude'].apply(lambda x: np.where(x > -0.98, np.nan, x))

df["gps_height"]=df['gps_height'].apply(lambda x: np.where(x < 1, np.nan, x))




df.to_csv(prepped_data_file_location, index= False)


