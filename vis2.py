import pandas as pd
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno



general_directory = os.path.split(os.getcwd())[0]


data_location = os.path.join(general_directory, "data")



dataset =  "trainingsetvalues" +".csv"
datasetlabels = "trainingsetlabels.csv"


df = pd.read_csv(os.path.join(data_location, dataset))
df1 = pd.read_csv(os.path.join(data_location, datasetlabels))

values_with_labels = pd.merge(left=df, right=df1, left_on = "id", right_on= "id")


def create_grouped_df(percentage, group_column):
    if percentage:
        dataframe = values_with_labels.groupby([group_column])["status_group"].value_counts(normalize=True).mul(100).reset_index(name="percentage")
    else:
        dataframe = values_with_labels.groupby([group_column])["status_group"].value_counts().reset_index(name="count")
    return (dataframe, group_column, percentage)



def plot_groupedbar(data, x, percentage):
    ptt = {"functional": "lightgreen", "non functional": "lightcoral", "functional needs repair": "wheat"}
    sns.set_theme(style='darkgrid')
    
    if percentage:  
        g = sns.catplot(x=x , y='percentage', hue= "status_group",  kind='bar', data = data, palette= ptt, legend= False)

        g.ax.set_ylim(0,100)

        for p in g.ax.patches:
            plt.annotate(str(format(p.get_height(), '.1f')+ '%'),
                        (p.get_x() + p.get_width()/2,
                        p.get_height()+1) , ha = 'center', va='center')
        plt.legend(loc='upper left')

    else:
        g = sns.catplot(x=x , y='count', hue= "status_group",  kind='bar', data = data, palette= ptt, legend= False) 
        plt.legend(loc= 'upper center')
    
    sns.despine()
    plt.show()

plotdata = create_grouped_df(False, 'extraction_type_class')
plot_groupedbar(plotdata[0], plotdata[1], plotdata[2])