# import sys
# import os
# import pandas as pd

# #########################
# # Import 'local' deeplabcut
# sys.path.append('/Users/user/Desktop/DeepLabCut/') 

# import deeplabcut
# print(os.path.abspath(deeplabcut.__file__))

# from deeplabcut.utils import auxiliaryfunctions

# #################################################
# # Read config of trained network
# config_path = '/Users/user/Desktop/sabris-mouse/sabris-mouse-nirel-2022-07-06/config.yaml'

# #######################################
# # Other params
# NUM_SHUFFLES=1 # this is an input to create_training_dataset but I think it is not saved anywhere

# ############################################
# Evaluate 
# to evaluate all saved snapshots: edit config  before evaluating network!
# auxiliaryfunctions.edit_config(config_path,
#             {'snapshotindex':'all'}) # to evaluate all snapshots saved

# # I hacked it to spit out 'combined' data in h5 file

# deeplabcut.evaluate_network(
#     config_path,
#     plotting=False, # comparisonbodyparts=['nose','rightLeg','leftLeg'],
#     show_errors=True,
#     gputouse=None,
# )

# ########################################
# # Read full predictions from evaluation results?
# # get evaluation folder
# cfg = auxiliaryfunctions.read_config(config_path)
# TrainingFractions = cfg["TrainingFraction"]
# # Loop thru shuffles
# for shuffle in range(1,NUM_SHUFFLES+1):

#     # Loop thru train-fractions
#     for trainFraction in TrainingFractions:

#         # get evaluation folder per shuffle and train-fracton
#         evaluationfolder = os.path.join(cfg["project_path"],
#                                         str(auxiliaryfunctions.get_evaluation_folder(trainFraction, shuffle, cfg)))

#         # get h5 file with evaluation data
#         eval_results_h5_path = os.path.join(evaluationfolder,
#                                             DLCscorer + "-" + str(snapshot) + ".h5")  



# # Read dataframe with combined data
# df_combined = pd.read_hdf(
#     '/Users/user/Desktop/sabris-mouse/sabris-mouse-nirel-2022-07-06/evaluation-results/iteration-0/sabris-mouseJul6-trainset80shuffle1/DLC_resnet50_sabris-mouseJul6shuffle1_1-snapshot-1_combined.h5')
# # conversioncode.guarantee_multiindex_rows(DataMachine) # not sure if this is req

# # Get list of columns
# df_combined.columns.values.tolist()

# # Spit in two dataframes?
# # the second element in the index is the video name
# df_human = df_combined['nirel']
# df_model = df_combined['DLC_resnet50_sabris-mouseJul6shuffle1_1']

# # Compute error per bodypart in a separate dataframe
# list_unique_bodyparts = list(set([x for (x,y) in df_model.columns.tolist()]))
# for bp in list_unique_bodyparts:
#     # compute distance between rows of same index
#     # first compute deltas--square--sum across cols -- sqrt
#     df_deltas_one_bp = df_human[bp][['x','y']] - df_model[bp][['x','y']]  # does pandas ensure matches between indices? check
#     df_distance = df_deltas_one_bp.pow(2).sum(axis=1, skipna=False).pow(0.5)


# ################
# # %%
# # Compute diff
# df_diff = df_human - df_model

# # Get llk from model prediction (can I do this in one go?)
# list_unique_bodyparts = list(set([x for (x,y) in df_diff.columns.tolist()]))
# for bp in list_unique_bodyparts:
#     df_diff[bp]['likelihood'] = df_model[bp]['likelihood']


#####################################################################
# %%
import sys
import os
import pandas as pd
import numpy as np

# Import 'local' deeplabcut
sys.path.append('/Users/user/Desktop/DeepLabCut/') 
import deeplabcut
print(os.path.abspath(deeplabcut.__file__))
from deeplabcut.utils import auxiliaryfunctions

#--------------------------------------------
# Read files from evaluated network
human_labels_filepath = '/Users/user/Desktop/sabris-mouse/sabris-mouse-nirel-2022-07-06/training-datasets/iteration-0/UnaugmentedDataSet_sabris-mouseJul6/CollectedData_nirel.h5'
model_predictions_filepath = '/Users/user/Desktop/sabris-mouse/sabris-mouse-nirel-2022-07-06/evaluation-results/iteration-0/sabris-mouseJul6-trainset80shuffle1/DLC_resnet50_sabris-mouseJul6shuffle1_1-snapshot-1.h5'

# Read config of trained network
config_path = '/Users/user/Desktop/sabris-mouse/sabris-mouse-nirel-2022-07-06/config.yaml'
NUM_SHUFFLES=1 # this is an input to create_training_dataset but I think it is not saved anywhere

#--------------------------------------------
# Read human labelled data (shuffle 1)
df_human = pd.read_hdf(human_labels_filepath)
# Read predictions
df_model = pd.read_hdf(model_predictions_filepath)

#--------------------------------------------
# Compute error per keypoint and test set sample
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

# CHECK: how is the error computed

# ------------------------
# Mean and sigma per bodypart?
df_summary_per_bodypart = df_results.describe()

#-----------------------------
# Mean and sigma across all bodyparts
np.nanmean(df_results.drop(labels='likelihood',axis=1,level=1)) # matches result for evaluate fn
df_results.groupby(level=1).mean()