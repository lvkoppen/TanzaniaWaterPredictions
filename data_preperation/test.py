import pandas as pd

import numpy as np

df = pd.DataFrame({'group' : ['X','Y', 'X','X','Y','X', 'Y','Y','X'], 
                    'name': ['A','A', 'B','B','B','B', 'C','C','C'], 
                    'value': [1, np.nan, np.nan, 2, 3, 1, 3, np.nan, 3]})

print(df.groupby(["group","name"], dropna= False)['value'])


#df["value"] = df.groupby(["group","name"]).transform(lambda x: x.fillna(x.mean()))

#print(x)