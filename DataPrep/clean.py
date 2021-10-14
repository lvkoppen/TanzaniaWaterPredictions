import pandas as pd
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno


working_directory = os.path.split(os.getcwd())[0]

general_directory = os.path.split(working_directory)[0]

print(general_directory)

data_location = os.path.join(general_directory, "data")



dataset =  "trainingsetvalues" +".csv"
datasetlabels = "trainingsetlabels.csv"


df = pd.read_csv(os.path.join(data_location, dataset))
df1 = pd.read_csv(os.path.join(data_location, datasetlabels))

values_with_labels = pd.merge(left=df, right=df1, left_on = "id", right_on= "id")


values_with_labels['construction_year'].replace(0, np.NaN)

msno.heatmap(values_with_labels)

plt.show()