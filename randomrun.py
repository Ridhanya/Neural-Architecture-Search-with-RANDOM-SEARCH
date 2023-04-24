import pandas as pd
import pickle
import numpy as np
from randomNAS import *
from randomvariables import *



# read the data
train_data = pd.read_csv('DATA/train.csv')
val_data = pd.read_csv('DATA/val.csv')


# split it into X and y values
x = np.array(train_data.drop(['label','filename','patient_id'], axis=1, inplace=False)).astype('float32')
#y = pd.get_dummies(data['label']).values
y = (train_data['label']).values

#validation dataset
x_val = np.array(val_data.drop(['label','filename','patient_id'], axis=1, inplace=False)).astype('float32')
y_val = (val_data['label']).values

# let the search begin
data = randomsearch(x,y,x_val,y_val)


#log data
with open(nas_data_log, 'wb') as f:
    pickle.dump(data, f)

# # get top n architectures (the n is defined in constants)
# get_top_n_architectures(TOP_N)

# #plot
# get_nas_accuracy_plot()