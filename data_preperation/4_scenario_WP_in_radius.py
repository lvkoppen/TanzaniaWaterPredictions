import pandas as pd
import os
import sys
import numpy as np
import pprint as pp
import geopy.distance
from tqdm import tqdm
import timeit
import json
from multiprocessing import Manager, Pool
from functools import partial
pp = pp.PrettyPrinter(indent=2)

def get_path_to_data():
    working_directory = os.path.split(os.getcwd())[0]

    general_directory = os.path.split(working_directory)[0]

    data_location = os.path.join(general_directory, "data")

    dataset =  "trainingsetvalues" +".csv"

    return os.path.join(data_location, dataset)

def get_path_to_place_data():
    working_directory = os.path.split(os.getcwd())[0]

    general_directory = os.path.split(working_directory)[0]

    data_location = os.path.join(general_directory, "data")

    prepped_data_folder = os.path.join(data_location, "prepped_data")

    prepped_file = '4_scenario_data' + ".csv"

    return os.path.join(prepped_data_folder, prepped_file)

def get_data(path):
    return pd.read_csv(path)
#standard columns to drop


#concatenation of source_type and extraction_type_group
def prepare_data(df_fresh):
    df = df_fresh
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

    return df.copy()


def prepare_data_for_gps(df):


    df_cols = df[['id','longitude', 'latitude']]
    df_cols.set_index('id', inplace=True)


    df_partial = df_cols.sort_values(by=['longitude', 'latitude'])


    df_partial['latlon'] = list(zip(df_partial['latitude'], df_partial['longitude']))

    return df_partial.copy()

def func_multiple_arguments(n, m, *args, **kwargs):

    return n, m


def mp_execution(df_partial, id):
        row_data = df_partial.loc[id,:]
        mask = (df_partial['latitude'].between(row_data.latlon[0]-1, row_data.latlon[0]+1) & 
                df_partial['longitude'].between(row_data.latlon[1]-1, row_data.latlon[1]+1))
        id_set = df_partial.index[mask]
        within_range = set()
        for ix in id_set:
            if ix != id:
                r_value = df_partial.loc[ix,:]
                distance = geopy.distance.distance(row_data.latlon ,r_value.latlon)
                if distance < 10:
                    within_range.add(ix)
        return id, within_range


def main(df):
    df_partial = df

    def get_nearby_waterpoints():
        pool = Pool(8)
        results = tqdm(pool.imap_unordered(partial(mp_execution, df_partial), list(df_partial.index.values), chunksize=40), total= len(df_partial.index.values))
        for id, l in results:
            nearby_dict[int(id)] = list(l)    
        with open('nearbydict.json', 'w') as fp:
            json.dump(dict(nearby_dict), fp)


    manager = Manager()
    nearby_dict = manager.dict()


    start = timeit.default_timer()
    get_nearby_waterpoints()
    end = timeit.default_timer()
    total = end - start
    print('Total time: {}'.format(total))
    #df.to_csv(prepped_data_file_location, index= False)


if __name__ == "__main__":
    path_to_data = get_path_to_data()

    df = get_data(path_to_data)
    df2 = prepare_data(df)
    df_final = prepare_data_for_gps(df2)
    main(df_final)
