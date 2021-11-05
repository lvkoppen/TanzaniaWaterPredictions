import pandas as pd
import os
import sys
import numpy as np
import pprint as pp
import geopy.distance
import timeit
from tqdm import tqdm
import json

pp = pp.PrettyPrinter(indent=2)
tqdm()
tqdm.pandas()


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

df['longitude'].replace(0, np.nan, inplace= True)

df['latitude'] = df['latitude'].apply(lambda x: np.where(x > -0.98, np.nan, x))

df.dropna(subset=['latitude', 'longitude'], inplace=True)

df_cols = df[['id','longitude', 'latitude']]
df_cols.set_index('id', inplace=True)


df_partial = df_cols.sort_values(by=['longitude', 'latitude'])





df_partial['latlon'] = list(zip(df_partial['latitude'], df_partial['longitude']))


nearby_dict = {}

def proximity_of(latlon):
    mask = (df_partial['latitude'].between(latlon[0]-1, latlon[0]+1) & df_partial['longitude'].between(latlon[1]-1, latlon[1]+1))
    id_set = df_partial.index[mask]
    return id_set


def get_nearby_waterpoints():
    for id in tqdm(df_partial.index):
        row_data = df_partial.loc[id,:]
        mask = proximity_of(row_data.latlon)
        within_range = set()
        for ix in mask:
            if ix != id:
                r_value = df_partial.loc[ix,:]
                distance = geopy.distance.distance(row_data.latlon ,r_value.latlon)
                if distance < 10:
                    within_range.add(ix)
        nearby_dict[id] = within_range
    with open('nearbydict.json', 'w') as fp:
        json.dump(nearby_dict, fp)

start = timeit.default_timer()
get_nearby_waterpoints()
end = timeit.default_timer()
total = end - start
print('Total time: {}'.format(total))
