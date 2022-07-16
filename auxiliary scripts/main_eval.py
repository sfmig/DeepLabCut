# 
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

# Import 'local' deeplabcut ---- use relative path instead?
sys.path.append('/Users/user/Desktop/DeepLabCut/') 
import deeplabcut
print(os.path.abspath(deeplabcut.__file__))
from deeplabcut.utils import auxiliaryfunctions

#--------------------------------------------
# Read files from evaluated network
human_labels_filepath = '/Users/user/Desktop/sabris-mouse/sabris-mouse-nirel-2022-07-06/training-datasets/iteration-0/UnaugmentedDataSet_sabris-mouseJul6/CollectedData_nirel.h5'
model_predictions_filepath = '/Users/user/Desktop/sabris-mouse/sabris-mouse-nirel-2022-07-06/evaluation-results/iteration-0/sabris-mouseJul6-trainset80shuffle1/DLC_resnet50_sabris-mouseJul6shuffle1_2-snapshot-2.h5'

# Read config of trained network
config_path = '/Users/user/Desktop/sabris-mouse/sabris-mouse-nirel-2022-07-06/config.yaml'
NUM_SHUFFLES=1 # this is an input to create_training_dataset but I think it is not saved anywhere

#--------------------------------------------
# Read human labelled data (shuffle 1)
df_human = pd.read_hdf(human_labels_filepath)
# Read predictions
df_model = pd.read_hdf(model_predictions_filepath)

#--------------------------------------------
# Compute error per keypoint and per sample in the test set 
cfg = auxiliaryfunctions.read_config(config_path)
trainingsetfolder = auxiliaryfunctions.GetTrainingSetFolder(cfg)
TrainingFractions = cfg["TrainingFraction"]

# Loop thru shuffles
for shuffle in range(1,NUM_SHUFFLES+1):
    # Loop thru train-fractions
    for trainFraction in TrainingFractions:
        # Get test indices
        _, metadatafn = auxiliaryfunctions.GetDataandMetaDataFilenames(
                trainingsetfolder, trainFraction, shuffle, cfg)
        _,_, testIndices, _ = auxiliaryfunctions.LoadMetadata(
                os.path.join(cfg["project_path"], metadatafn))

        # Get rows from test set only
        df_human_test_only = df_human.iloc[testIndices,:]  # test idcs form images in ascending order?
        df_model_test_only = df_model.iloc[testIndices,:]

        # Drop scorer level
        df_human_test_only = df_human_test_only.droplevel('scorer',axis=1)
        df_model_test_only = df_model_test_only.droplevel('scorer',axis=1)   

        ### Compute deltas in x and y dir between human scorer and model prediction
        df_diff_test_only = df_human_test_only - df_model_test_only
        # Drop llk for model predictions before computing distance
        df_diff_test_only = df_diff_test_only.drop(labels='likelihood',axis=1,level=1)

        # Compute distance btw model and human
        # - nrows = samples in test set
        # - ncols = bodyparts tracked
        df_distance_test_only = df_diff_test_only.pow(2).sum(level='bodyparts',axis=1,skipna=False).pow(0.5)
        # warning: recommends to use 'df_diff_test_only.pow(2).groupby(level='bodyparts',axis=1).sum(axis=1,skipna=False)' instead,
        # but that makes NaNs into 0s!
        # add distance level
        df_distance_test_only.columns = pd.MultiIndex.from_product([df_distance_test_only.columns, ['distance']])

        ## Combine w Likelihood
        df_llk_test_only = df_model_test_only.drop(labels=['x','y'],axis=1,level=1)
        df_results = pd.concat([df_distance_test_only,df_llk_test_only],axis=1).sort_index(level=0,axis=1)


# ------------------------
# Mean and sigma per bodypart
df_summary_per_bodypart = df_results.describe()
print('----------------------')
print('Pixel error per bodypart:')
print(df_summary_per_bodypart)


# Plot histogram per bodypart
# https://mode.com/example-gallery/python_histogram/
# drop 'distance' level column before plotting
np_axes_per_row = df_results.drop(labels='likelihood',axis=1,level=1).droplevel(level=1,axis=1).hist()  # returns: matplotlib.AxesSubplot or numpy.ndarray of them
for ax in np_axes_per_row:
    for i in range(ax.shape[0]):
        # Title
        # ax[i].title
        # Despine
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['left'].set_visible(False)

        # Switch off ticks
        ax[i].tick_params(axis='both',which='both',\
                        bottom='off', top='off',labelbottom='on')
        # Draw horiz lines per ytick
        for yt in ax[i].get_yticks():
                ax[i].axhline(y=yt, linestyle='-',alpha=0.4, color='#eeeeee', zorder=1)

        # Set x and y labels
        ax[i].set_xlabel('error-distance (pixels)', weight='bold', size=12)
        ax[i].set_ylabel('counts',weight='bold',size=12) # labelpad=5,
        
        # Format y-axis label
        ax[i].yaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))
        ax[i].tick_params(axis='x', rotation=0)
#-----------------------------
# Mean and sigma across all bodyparts
px_error_all_bodyparts_and_test_samples = np.nanmean(df_results.drop(labels='likelihood',axis=1,level=1)) # matches result for evaluate fn
print('----------------------')
print('Pixel error all bodyparts and test samples:')
print(px_error_all_bodyparts_and_test_samples)
print('----------------------')