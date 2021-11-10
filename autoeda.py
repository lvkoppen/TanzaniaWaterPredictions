import pandas as pd
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from dataprep.eda import create_report
import sweetviz as sv
from autoviz.AutoViz_Class import AutoViz_Class

general_directory = os.path.split(os.getcwd())[0]


data_folder = os.path.join(general_directory, "data")

data_location = os.path.join(data_folder, "prepped_data")


dataset =  "4_scenario_data" +".csv"
#datasetlabels = "trainingsetlabels.csv"


df = pd.read_csv(os.path.join(data_location, dataset))
#df1 = pd.read_csv(os.path.join(data_folder, datasetlabels))

#values_with_labels = pd.merge(left=df, right=df1, left_on = "id", right_on= "id")



report = sv.analyze(df)
report.show_html('4_prepped_data.html')