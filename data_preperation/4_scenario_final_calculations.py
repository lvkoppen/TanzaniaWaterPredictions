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


    df_cols = df[['id','longitude', 'latitude', 'waterpoint_age']]
    df_cols.set_index('id', inplace=True)


    df_partial = df_cols.sort_values(by=['longitude', 'latitude'])


    df_partial['latlon'] = list(zip(df_partial['latitude'], df_partial['longitude']))

    return df_partial.copy()

    


def calc_wp_data(id, df_partial = None, list_wp_nearby = None,):


    radius = {'1km' : [], '3km': [], '10km': []}

    average_age_1km = 0
    average_age_3km = 0
    average_age_10km = 0

    count_1km = 0
    count_3km = 0
    count_10km = 0

    row_data = None


    try:
        row_data = df_partial.loc[int(id),:]
    except:
        print('not in test set')
        return (int(id), average_age_1km, average_age_3km, average_age_10km, count_1km, count_3km, count_10km)

    if list_wp_nearby[id]:
        for wp in list_wp_nearby[id]:
            wp = int(wp)
            r_value = None
            try:
                r_value = df_partial.loc[wp,:]
            except:
                print('not in test set')
                continue
            distance = geopy.distance.distance(row_data.latlon ,r_value.latlon)
            if(distance <= 1):
                radius['1km'].append(wp)
            elif(distance <= 3):
                radius['3km'].append(wp)
            elif(distance <= 10):
                radius['10km'].append(wp)

        if(radius['1km']):
            average_age_1km = df_partial.loc[radius['1km'], 'waterpoint_age']
            count_1km = len(radius['1km'])
            average_age_1km = df_partial['waterpoint_age']
        if(radius['3km']):
            average_age_3km = df_partial.loc[radius['3km'], 'waterpoint_age'].mean()
            count_3km = len(radius['3km'])
        if(radius['10km']):
            count_10km = len(radius['10km'])
            average_age_10km = df_partial.loc[radius['10km'], 'waterpoint_age'].mean()

        

    return (int(id), average_age_1km, average_age_3km, average_age_10km, count_1km, count_3km, count_10km)




def get_json_data():
    with open('nearbydict.json') as fp:
        data = json.load(fp)
    return data
        
    

    

def main(df, df_unfiltered):
    df_partial = df
    wp_location_data = get_json_data()

    def calculate_nearby_features():
        pool = Pool(8)
        results = tqdm(pool.imap_unordered(partial(calc_wp_data, df_partial=df_partial, list_wp_nearby= wp_location_data), 
                                            list(wp_location_data.keys()), chunksize=20),total=len(wp_location_data.keys()))
        
        df_results = pd.DataFrame(results, columns= ['id','average_age_1km', 'average_age_3km', 'average_age_10km', 'count_1km', 'count_3km', 'count_10km'])
        df_results.set_index('id', inplace=True)

        return df_results


    start = timeit.default_timer()
    new_features_df = calculate_nearby_features()
    end = timeit.default_timer()
    total = end - start
    print('Total time: {}'.format(total))

    df_total = df_unfiltered.merge(new_features_df, on='id')

    print(df_total.shape)

    get_path_to_place_data()
    df.to_csv(get_path_to_place_data(), index= False)


if __name__ == "__main__":
    path_to_data = get_path_to_data()

    df = get_data(path_to_data)
    st = timeit.default_timer()
    df2 = prepare_data(df)
    end = timeit.default_timer()
    tt = end - st
    print("Time to perform initial data cleaning operations: {}".format(tt))
    df_gps = prepare_data_for_gps(df2)
    main(df_gps, df2)
