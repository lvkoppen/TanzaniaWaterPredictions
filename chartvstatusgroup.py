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
        plt.legend(loc='upper right')
        g.set_xticklabels(g.ax.get_xticklabels(), rotation=45)
    else:
        g = sns.catplot(x=x , y='count', hue= "status_group",  kind='bar', data = data, palette= ptt, legend= False) 
        g.set_xticklabels(g.ax.get_xticklabels(), rotation=45)
        plt.legend(loc= 'upper center')
    
    sns.despine()
    plt.show()

#plotdata = create_grouped_df(False, 'permit')
#plot_groupedbar(plotdata[0], plotdata[1], plotdata[2])

def bin_values(data, group_column, bins, percentage, qcut):
    binned_group_column = "binned_" + group_column
    print(binned_group_column)
    
    if qcut:
        values_with_labels[binned_group_column] = pd.qcut(data[group_column], q=10)
    else:
        values_with_labels[binned_group_column] = pd.cut(data[group_column], bins)
 

    if percentage:
        binned_grouped_df = values_with_labels.groupby([binned_group_column])["status_group"].value_counts(normalize=True).mul(100).reset_index(name="percentage")
    else:
        binned_grouped_df = values_with_labels.groupby([binned_group_column])["status_group"].value_counts().reset_index(name="count")
    return (binned_grouped_df, binned_group_column, percentage)
#values_with_labels.groupby()

bins = [-1,0, 25, 150, 300, 500, 1000, 5000, 10000, 20000, 35000]
column = "payment"
#plotdata = bin_values(values_with_labels, column, bins, False, False)
plotdata = create_grouped_df(True, column)
plot_groupedbar(plotdata[0], plotdata[1], plotdata[2])