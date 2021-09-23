import pandas as pd
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import folium
import pprint


pp = pprint.PrettyPrinter(indent =4)
general_directory = os.path.split(os.getcwd())[0]


data_location = os.path.join(general_directory, "data")



dataset =  "trainingsetvalues" +".csv"
datasetlabels = "trainingsetlabels.csv"


df = pd.read_csv(os.path.join(data_location, dataset))
df1 = pd.read_csv(os.path.join(data_location, datasetlabels))

values_with_labels = pd.merge(left=df, right=df1, left_on = "id", right_on= "id")

# g = sns.catplot(data=values_with_labels, kind='bar',
#                 x ='water_quality', y=  ,hue= 'status_group'

# )

for col in values_with_labels.columns:
    if(col != ""):
        print(values_with_labels[col].value_counts())

# m = folium.Map(location= [7, 35],
#                zoom_start=6 
#             )

# values_with_labels['num_status'] = pd.Categorical(values_with_labels["status_group"]).codes
# colordict = {0: 'green', 1: 'yellow', 2: 'red'}

# for lat, lon, status, id in zip(values_with_labels['latitude'], values_with_labels['longitude'], values_with_labels['num_status'], values_with_labels['id']):
#     folium.CircleMarker(
#         [lat, lon],
#         color='b',
#         radius= 3,
#         popup = ('latitude: ' + str(lat) + '<br>'
#             'longitude: ' + str(lon) + '<br>'
#             'id: ' + str(id) +'%'
#         ),
#         threshold_scale=[1,2,3],
#         fill_color=colordict[status],
#         fill=True,
#         fill_opacity=0.7
#         ).add_to(m)
# sw = values_with_labels[['latitude', 'longitude']].min().values.tolist()
# ne = values_with_labels[['latitude', 'longitude']].max().values.tolist()
# m.fit_bounds([sw, ne])
# m.save('index.html')

# x = values_with_labels.loc[values_with_labels['id']== 49651]
#pp.pprint(x)
plt.show()


