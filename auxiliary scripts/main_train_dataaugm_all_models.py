"""
This script launches all training jobs for a data augm study
- We use a data augmentation baseline as reference (10 different methods)
- We train a network dropping one method everytime

To run in the background: [eventually this in a bash script?]
-----------------------------
nohup python main_dataaugm_ablation.py & 
nohup python main_dataaugm_ablation.py > log_[date].out & --- I havent tried this

Contributors: Sofia, Jonas, Sabrina
"""

import sys, os, shutil

##############################################
print(sys.path)  # to check interpreter path (befre/after)
sys.path.append('/Users/user/Desktop/DeepLabCut/') # is there a better way to do this?
print(sys.path)
##############################################
import deeplabcut
from deeplabcut.utils.auxiliaryfunctions import read_config, edit_config
from deeplabcut.generate_training_dataset.trainingsetmanipulation import create_training_dataset

##########################################################
### Set config path of project with labelled data
# (we assume create_training_dataset has already been run)
config_path = '/media/data/stinkbugs-DLC-2022-07-15/config.yaml'

# Other params
NUM_SHUFFLES=3
#SHUFFLE_ID=1
TRAINING_SET_INDEX=0 # default
MAX_SNAPSHOTS=3
DISPLAY_ITERS=1 # display loss every N iters; one iter processes one batch
SAVE_ITERS=1 # save snapshots every n iters
MAX_ITERS=1

N_GPUS = 4 # to assing models to one gpu everytime?

##########################################################
### Get config as dict and associated paths
cfg = read_config(config_path)
project_path = cfg["project_path"] # or: os.path.dirname(config_path) #dlc_models_path = os.path.join(project_path, "dlc-models")
training_datasets_path = os.path.join(project_path, "training-datasets")

base_train_pose_config_file_path,\
    _, _ = deeplabcut.return_train_network_path(config_path,
                                                shuffle=SHUFFLE_ID, 
                                                trainingsetindex=0)  # base_train_pose_config_file

# each model subfolder is named with the format: <modelprefix_pre>_<id>_<str_id>
modelprefix_pre = "data_augm" 

##################################################################
### Define parameters for each data augmentation method
# ATT! Parameters must be defined for True or False cases.
# Not defining a set of parameters will result in applying the parameters from the pose_config.yaml template

## Initialise baseline dict with params per data augm type
parameters_dict = dict() # define a class instead of a dict?

### General
parameters_dict['general'] = {'dataset_type': 'imgaug', # OJO! not all the following will be available?
                                'batch_size': 1, # 128
                                'apply_prob': 0.5,
                                'pre_resize': []} # Specify [width, height] if pre-resizing is desired

### Crop----is this applied if we select imgaug? I think so....
parameters_dict['crop'] = {False: {'crop_by': 0.0,
                                  'cropratio': 0.0},
                           True: {'crop_by': 0.15,
                                  'cropratio': 0.4}}#---------- these are only used if height, width passed to pose_imgaug
# from template:
# parameters_dict['crop'] = {'crop_size':[400, 400],  # width, height,
#                   'max_shift': 0.4,
#                   'crop_sampling': 'hybrid',
#                   'cropratio': 0.4}---------- crop ratio is used too

### Rotation
parameters_dict['rotation'] = {False:{'rotation': 0, 
                                      'rotratio': 0},
                                True:{'rotation': 25, 
                                      'rotratio': 0.4}}

### Scale
parameters_dict['scale'] = {False:{'scale_jitter_lo': 1.0,
                                   'scale_jitter_up': 1.0},
                            True:{'scale_jitter_lo': 0.5,
                                  'scale_jitter_up': 1.25}}


### Motion blur
# ATT motion_blur is not expected as a dictionary
parameters_dict['motion_blur'] = {False: {'motion_blur': False}, # motion_blur_params should not be defined if False, but check if ok
                                  True: {'motion_blur': True,
                                         'motion_blur_params':{"k": 7, "angle": (-90, 90)}}}  

### Contrast
# ATT for Contrast a dict should be defined in the yaml file!
# also: log, linear, sigmoid, gamma params...include those too? [I think if they are not defined in the template we are good, they wont be set]
parameters_dict['contrast'] = {False: {'contrast': {'clahe': False,
                                                    'histeq': False}}, # ratios should not be defined if False, but check if ok
                                True:{'contrast': {'clahe': True,
                                                    'claheratio': 0.1,
                                                    'histeq': True,
                                                    'histeqratio': 0.1}}}


### Convolution
# ATT for Convolution a dict should be defined in the yaml file!
parameters_dict['convolution'] = {False: {'convolution': {'sharpen': False,  # ratios should not be defined if False, but check if ok
                                                          'edge': False,  
                                                          'emboss': False}}, # this needs to be fixed in pose_cfg.yaml template?
                                  True: {'convolution':{'sharpen': True, 
                                                        'sharpenratio': 0.3, #---- in template: 0.3, in pose_imgaug default is 0.1
                                                        'edge': True,  
                                                        'edgeratio': 0.1, #--------
                                                        'emboss': True,
                                                        'embossratio': 0.1}}}
### Mirror
parameters_dict['mirror'] = {False: {'mirror': False},
                             True: {'mirror': True}} 

### Grayscale
parameters_dict['grayscale'] = {False: {'grayscale': False},
                                True: {'grayscale': True}}

### Covering
parameters_dict["covering"] = {False: {'covering': False},
                               True: {'covering': True}}

### Elastic transform  
parameters_dict["elastic_transform"] = {False: {'elastic_transform': False},
                                        True: {'elastic_transform': True}}

### Gaussian noise
parameters_dict['gaussian_noise'] = {False: {'gaussian_noise': False},
                                    True: {'gaussian_noise': True}}

############################################################################
## Define baseline 
baseline = {'crop':             True, #----check
            'rotation':         True,
            'scale':            True,
            'mirror':           False,
            'contrast':         True,
            'motion_blur':      True,
            'convolution':      False,
            'grayscale':        False,
            'covering':         True,
            'elastic_transform': True,
            'gaussian_noise':   False}


#################################################
## Create list of strings identifying each model
list_of_data_augm_models_strs = ['baseline'] 
for ky in baseline.keys() :
    list_of_data_augm_models_strs.append(ky) #'wo_' + ky)


#########################################
## Loop to train each model
#list_gpus_to_use = list(range(N_GPUS))

for i,daug_str in enumerate(list_of_data_augm_models_strs):

    ###########################################################
    # Create subdirs for this augmentation method
    model_prefix = '_'.join([modelprefix_pre, str(i), daug_str]) # modelprefix_pre = aug_
    aug_project_path = os.path.join(project_path, model_prefix)
    aug_dlc_models = os.path.join(aug_project_path, "dlc-models", )
    aug_training_datasets = os.path.join(aug_project_path, "training-datasets")
    # create subdir for this model
    try:
        os.mkdir(aug_project_path)
    except OSError as error:
        print(error)
        print("Skipping this one as it already exists")
        continue
    # copy tree 'training-datasets' of dlc project under subdir for the current model
    shutil.copytree(training_datasets_path, aug_training_datasets)

    ###########################################################
    # Copy base train pose config file to the directory of this augmentation method
    for sh in range(NUM_SHUFFLES):
        one_train_pose_config_file_path,\
            _, _ = deeplabcut.return_train_network_path(config_path,
                                                        shuffle=sh, 
                                                        trainingsetindex=TRAINING_SET_INDEX, # default
                                                        modelprefix=model_prefix) 
        
        os.makedirs(str(os.path.dirname(one_train_pose_config_file_path))) # create parentdir 'train'
        shutil.copyfile(base_train_pose_config_file_path, 
                        one_train_pose_config_file_path) #copy base train config file


    #####################################################
    # Create dict with the data augm params for this model

    # initialise dict
    edits_dict = dict()

    # add gral params
    edits_dict.update(parameters_dict['general'])

    for ky in baseline.keys(): 
        if daug_str == ky:
            # Get params that correspond to the opposite state of the method daug_str in the baseline
            d_temp = parameters_dict[ky][not baseline[ky]] 
            # add to edits dict
            edits_dict.update(d_temp)
        else:
            # Get params that correspond to the same state as the baseline
            d_temp = parameters_dict[ky][baseline[ky]] 
            # add to edits dict
            edits_dict.update(d_temp)

    # print
    print('-----------------------------------')
    if daug_str == 'baseline':
        print('Data augmentation model {}: {}'.format(i, daug_str))
    else:
        print('Data augmentation model {}: "{}" opposite to baseline'.format(i, daug_str))
    [print('{}: {}'.format(k,v)) for k,v in edits_dict.items()]
    print('-----------------------------------')
    ##################################################
    # Edit config for this augmentation method
    edit_config(str(one_train_pose_config_file_path), edits_dict)
    edit_config(str(one_train_pose_config_file_path), {'project_path': aug_project_path}) 
    #---should this be aug_project_path? or the parentdir to config.yaml (i.e. project_path)? bc it is copied from parent dir, it is already set to project_path

    #########################################
    # ## Train model
    # deeplabcut.train_network(config_path, # config.yaml, common to all models
    #                         shuffle=SHUFFLE_ID,
    #                         trainingsetindex=TRAINING_SET_INDEX,
    #                         max_snapshots_to_keep=MAX_SNAPSHOTS,
    #                         displayiters=DISPLAY_ITERS,
    #                         maxiters=MAX_ITERS,
    #                         saveiters=SAVE_ITERS,
    #                         gputouse=list_gpus_to_use[i],
    #                         allow_growth=True,
    #                         modelprefix=model_prefix)