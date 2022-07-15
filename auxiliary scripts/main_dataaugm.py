"""
Launch training jobs for a data augm study
- The baseline is all data augm methods applied (8 methods)
- We train a network dropping one method everytime

To run in the background: [eventually this in a bash script?]
-------------------------
nohup python main_dataaugm_ablation.py & 

nohup python main_dataaugm_ablation.py > log_[date].out --- I havent tried this

https://linoxide.com/example-how-to-use-linux-nohup-command/
"""
###########################################################
# Import deeplabcut
import deeplabcut
from deeplabcut.utils.auxiliaryfunctions import read_config, edit_config
from deeplabcut.generate_training_dataset.trainingsetmanipulation import create_training_dataset

# # to check where deeplabcut is coming from: 
# print(os.path.abspath(deeplabcut.__file__))

######################################################
### Input params
## Set config path
# (assuming training dataset already created)
config_path = '/home/sofia/dlc/data/stinkbugs/config.yaml'
#cfg = read_config(config_path)

## Set other params
NUM_SHUFFLES=1
SHUFFLE_ID=1
MAX_SNAPSHOTS=5
DISPLAY_ITERS=1000 # display loss every N iters; one iter processes one batch
SAVE_ITERS=25000 # save snapshots every n iters
MAX_ITERS=150000 #----to test only

#----------------------------------------------------------------------------------
## Define config for training with data augm params
# References:
# - params in pose_cfg.yaml
# - params in pose_imgaug
# - we could use the standard imgaug as ref: 
#   https://imgaug.readthedocs.io/en/latest/source/examples_basics.html#a-simple-and-common-augmentation-sequence
# - OJO maybe not all in imgaug???


## Initialise dict with params per data augm type
params = dict() # define a class instead of a dict?

### General
params['general'] = {'dataset_type': 'imgaug', # OJO! not all the following will be available?
                     'batch_size': 8, # 128
                     'apply_prob': 0.5,
                     'pre_resize': []} # Specify [width, height] if pre-resizing is desired

### Crop----is this applied if we select imgaug?
# params['crop'] = {'crop_size':[400, 400],  # width, height,
#                   'max_shift': 0.4,
#                   'crop_sampling': 'hybrid',
#                   'cropratio': 0.4}

### Rotation
params['rotation'] = {'rotation': 180, #25,
                      'rotratio': 0.4}

### Scale
params['scale'] = {'scale_jitter_lo': 0.5,
                   'scale_jitter_up': 1.25}

### Mirror
params['mirror'] = {'mirror': True} ###----- default is False; is there vertical flip?

### Contrast
# also> log, linear, sigmoid, gamma params...include?
params['contrast'] = {'clahe': True,
                    'claheratio': 0.1,
                    'histeq': True,
                    'histeqratio': 0.1}
### Convolution
params['convolution'] = {'sharpen': True, ### ---default is False
                        'sharpenratio': 0.3,
                        'edge': True,  ### ---default is False
                        'emboss':
                            {'alpha': [0.0, 1.0],
                            'strength': [0.5, 1.5],
                            'embossratio': 0.1}}

### Motion blur
params['motion_blur'] = {'motion_blur': True,
                         'motion_blur_params':{"k": 7, "angle": (-90, 90)}}     

### Grayscale
params['grayscale'] = {'grayscale': True}

### Covering
params["covering"] = {"covering": True}

### Elastic transform  
params["elastic_transform"] = {"elastic_transform": True}

### Gaussian noise
params['gaussian_noise'] = {'gaussian_noise': True}

## Other
# - deterministic?
# - [x] motion blur?
# - gaussian_noise, noise_sigma
# - [x] grayscale
# - [x] constrast?
# - [x] covering
# - [x] elastic transform
# stride?

#----------------------------------------------------------
## Ablation study
list_data_augm_type = list(params.keys())
list_data_augm_type_to_pop = ['']
list_data_augm_type_to_pop.extend(list_data_augm_type)
list_data_augm_type_to_pop.remove('general')

map_iter_to_data_augm_excl_case = dict()
for k, pop_el in enumerate(list_data_augm_type_to_pop):

    ## Treat each case as a different iteration (hack)--100,200,300  -------
    iter_data_augm = (k+1)*100
    print('--------------------------------------------------')
    print('Data augm excl iter: {}'.format(iter_data_augm))
    # add to dict
    map_iter_to_data_augm_excl_case[iter_data_augm] = pop_el

    ## Edit config per iter  
    edit_config(config_path,{'iteration': iter_data_augm}) 
    
    # Create training dataset
    create_training_dataset(config_path,
                            num_shuffles=NUM_SHUFFLES)   
    
    # Return train config for this iter
    # Get path to train config
    train_pose_config_file,\
        _, _ = deeplabcut.return_train_network_path(config_path,
                                                    shuffle=SHUFFLE_ID, 
                                                    trainingsetindex=0, # default
                                                    modelprefix="") # default
    #------------------------------------------
    ## Create edit config for each case
    list_data_augm_type_one_iter = [j for j in list_data_augm_type if j!=pop_el]

    print('Excluding params from {}'.format(pop_el)) #print([i for i in list_data_augm_type_one_iter])
    print('--------------------------------------------------')
    
    edits_dict = dict()
    for el in list_data_augm_type_one_iter:    
        edits_dict.update(params[el])

    #--------
    # Print data augm params to terminal
    # for k,v in edits_dict.items():
    #     print(k,v)
    # print('------')

    #------------------------------------------
    ## Edit train config
    edit_config(str(train_pose_config_file), edits_dict)

    #-------------------------------------------------
    ### Train network for N iters
    deeplabcut.train_network(config_path,
                                shuffle=SHUFFLE_ID,
                                max_snapshots_to_keep=MAX_SNAPSHOTS,
                                displayiters=DISPLAY_ITERS,
                                maxiters=MAX_ITERS,
                                saveiters=SAVE_ITERS,
                                gputouse=2) 

### Print to log the map from iter number to exclusion case
for k,v in map_iter_to_data_augm_excl_case.items():
    print('{} - data augm excluding {}'.format(k,v))
