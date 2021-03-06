import pandas as pd
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


general_directory = os.path.split(os.getcwd())[0]


data_location = os.path.join(general_directory, "data")



dataset =  "trainingsetvalues" +".csv"
datasetlabels = "trainingsetlabels.csv"


df = pd.read_csv(os.path.join(data_location, dataset))
df1 = pd.read_csv(os.path.join(data_location, datasetlabels))

values_with_labels = pd.merge(left=df, right=df1, left_on = "id", right_on= "id")

values_with_labels['log_gpsheight'] = np.log(values_with_labels['gps_height'], out=np.zeros_like(values_with_labels['gps_height'].astype('d')), where=(values_with_labels['gps_height']!=0))


sns.histplot(values_with_labels['log_gpsheight'])

plt.show()