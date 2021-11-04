import pandas as pd
import os
import sys
import numpy as np
from thefuzz import fuzz, process
import pprint as pp
import geopy.distance
from tqdm import tqdm
from pandarallel import pandarallel

pp = pp.PrettyPrinter(indent=2)

tqdm.pandas()

pandarallel.initialize(progress_bar=True)
working_directory = os.path.split(os.getcwd())[0]

general_directory = os.path.split(working_directory)[0]


data_location = os.path.join(general_directory, "data")

prepped_data_folder = os.path.join(data_location, "prepped_data")

dataset =  "trainingsetvalues" +".csv"


prepped_file = '4_scenario_data' + ".csv"

gps_file = '4_gps_data' + ".csv"

prepped_data_file_location = os.path.join(prepped_data_folder, prepped_file)

prepped_gps_file= os.path.join(prepped_data_folder, gps_file)

df = pd.read_csv(os.path.join(data_location, dataset))

#standard columns to drop


#concatenation of source_type and extraction_type_group

df['source_extraction_type'] = df['source_type']+df['extraction_type_group']
                


df['gps_height'] = df['gps_height'].abs()

#replacing missing values with NaN type and incorrect values(zeroes)
df['construction_year'].replace(0, np.nan, inplace= True)

df['longitude'].replace(0, np.nan, inplace= True)

df['latitude'] = df['latitude'].apply(lambda x: np.where(x > -0.98, np.nan, x))

df["gps_height"]=df['gps_height'].apply(lambda x: np.where(x < 0.1, np.nan, x))



df['gps_height'] = df.groupby('subvillage', dropna = False)['gps_height'].transform(lambda x: x.fillna(x.mean()))

df['gps_height'] = df.groupby('ward', dropna = False)['gps_height'].transform(lambda x: x.fillna(x.mean()))

df['gps_height'] = df.groupby('region', dropna = False)['gps_height'].transform(lambda x: x.fillna(x.mean()))

df['date_recorded'] = pd.to_datetime(df["date_recorded"], format = '%Y-%m-%d', errors = 'coerce')

df.dropna(subset=['latitude', 'longitude'], inplace=True)


df['construction_year'] = df.groupby(['subvillage','extraction_type'], dropna = False)['construction_year'].transform(lambda x: x.fillna(x.median()))
df['construction_year'] = df.groupby(['ward','extraction_type'], dropna = False)['construction_year'].transform(lambda x: x.fillna(x.median()))
df['construction_year'] = df.groupby(['region','extraction_type'], dropna = False)['construction_year'].transform(lambda x: x.fillna(x.median()))

df["construction_year"]=df['construction_year'].apply(lambda x: np.where(x < 0, np.nan, x))

df['construction_year']= df['construction_year'].round()
df['construction_year'].replace(0, np.nan, inplace= True)
df.dropna(subset= ['construction_year'], inplace=True)




df['waterpoint_age'] = df['date_recorded'].dt.year - df['construction_year']

df['latlon'] = list(zip(df['latitude'], df['longitude']))

square = pd.DataFrame(
    np.zeros((df.shape[0], df.shape[0])),
    index=df.index, columns=df.index
)

# replacing distance.vicenty with distance.distance
def get_distance(col):
    end = df.loc[col.name, 'latlon']
    return df['latlon'].progress_apply(geopy.distance.distance,
                              args=(end,),
                              ellipsoid='WGS-84'
                             )

distances = square.parallel_apply(get_distance, axis=1).T

print(distances.head())

distances.to_csv(prepped_gps_file, index = False)
#df.to_csv(prepped_data_file_location, index= False)


