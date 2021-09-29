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

#create dataframe where rows are all the different columns
described_df = pd.DataFrame(df.columns, columns = ['column'])

#add data type of a column to the described dataframe
described_df["data_type"] = [df[column].dtype for column in df.columns]

#add unique count of variables in a column to the described dataframe
described_df["unique_count"] = [df[column].nunique() for column in df.columns]

#add nan count of variables in a column to the described df
described_df["nan_count"] = [df[column].isna().sum() for column in df.columns]

#add zero (0) count of variables in a column to the described df
described_df["zero_count"] = [(df[column] == 0).sum(axis=0) for column in df.columns]


described_df.to_excel("description.xlsx")



print(described_df)