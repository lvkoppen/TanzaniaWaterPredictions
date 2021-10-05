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


data_location = os.path.join(general_directory, "data")



dataset =  "trainingsetvalues" +".csv"
datasetlabels = "trainingsetlabels.csv"


df = pd.read_csv(os.path.join(data_location, dataset))
df1 = pd.read_csv(os.path.join(data_location, datasetlabels))

values_with_labels = pd.merge(left=df, right=df1, left_on = "id", right_on= "id")

removed_empty_years = values_with_labels[values_with_labels.construction_year > 1960]

report = sv.analyze(removed_empty_years)
report.show_html('nonzeroyears.html')