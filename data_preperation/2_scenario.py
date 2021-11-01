import pandas as pd
import os
import sys
import numpy as np
from thefuzz import fuzz, process
import pprint as pp

pp = pp.PrettyPrinter(indent=2)
working_directory = os.path.split(os.getcwd())[0]

general_directory = os.path.split(working_directory)[0]


data_location = os.path.join(general_directory, "data")

prepped_data_folder = os.path.join(data_location, "prepped_data")

dataset =  "trainingsetvalues" +".csv"


prepped_file = '2_scenario_data' + ".csv"

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

df["gps_height"]=df['gps_height'].apply(lambda x: np.where(x < 0.1, np.nan, x))



# df['funder'] = df['funder'].str.rstrip()

# df['funder'].replace('0','unknown', inplace=True)

# df['funder'].replace('Rc Church','Roman Catholic Church', inplace=True)

# df['funder'].replace('Roman','Roman Catholic Church', inplace=True)

# df['funder'].replace('Rc', 'Roman Catholic Church', inplace=True)

# df['funder'].replace('Rwssp', 'rural water supply and sanitation', inplace=True)

# df['funder'].replace('Rural Water Supply And Sanitat', 'rural water supply and sanitation', inplace=True)

# df['funder'].replace('Nethalan', 'Netherlands', inplace=True)

# df['funder'].replace('Germany Republi', 'German Republic', inplace=True)

# df['funder'].replace('Dh', 'dhv', inplace=True)

# df['funder'].replace('Holand', 'netherlands', inplace=True)

# df['funder'].replace('Holland', 'netherlands', inplace=True)

# df['funder'] = df['funder'].map(lambda x: x.casefold() if isinstance(x, str) else x)


# df['funder'].replace('world bank/government', 'world bank', inplace=True)

# df['funder'].replace('romam catholic', 'roman catholic church', inplace=True)
# df['funder'].replace('unice', 'unicef', inplace=True)
# df['funder'].replace('roman catholic', 'roman catholic church', inplace=True)
# df['funder'].replace('roman cathoric', 'roman catholic church', inplace=True)




# filtered_set = set()

# add existing program names:
# filtered_set.add('nrwssp')
# filtered_set.add('rwssi')
# filtered_set.add('nwsds')
# filtered_set.add('dwst')
# filtered_set.add('mofea')
# filtered_set.add('france')
# filtered_set.add('germany')

# close_but_not_enough = set()
# swapped_set = set()

# x = df['funder'].value_counts()

# print(x.head(40))
# for i in range(0,40):
#     filtered_set.add(x.index[i])

# print(filtered_set)



# def fuzzy_matching(word, matched_word):
#     x =fuzz.token_sort_ratio(word, matched_word)
#     if (x > 80):
#         return True
#     if (x > 50 and x < 80):
#         close_but_not_enough.add((word, matched_word))
#     return False
# def get_set_values():
#     sett = filtered_set.copy()
#     return sett
# def replace_fuzzy_match(word):
#     fil_set = get_set_values()
#     if word not in fil_set:
#         for mword in fil_set:
#             if word != mword:
#                 if(fuzzy_matching(word, mword)):
#                     df['funder'].replace(word, mword, inplace=True)
#                     swapped_set.add((word, mword))
#                     break
#                 else: 
#                     filtered_set.add(word)
#             else:
#                 break


# df['funder'].apply(lambda x: replace_fuzzy_match(x))


# pp.pprint(df['funder'].value_counts())


# pp.pprint(close_but_not_enough)

# pp.pprint(swapped_set)

df.to_csv(prepped_data_file_location, index= False)


