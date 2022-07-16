### Train using local ('git-cloned') deeplabcut codebase
# use a conda env without deeplabcut
import sys
import os
######################################################
## Set deeplabcut location

# ### By default: the interpreter looks in the current path, so I can change current path [this doesnt work for me]
# print(os.getcwd())
# import deeplabcut

### Alternatively: add depplabcut module from git-cloned repo to path
print(sys.path)  # to check interpreter path (befre/after)
sys.path.append('/Users/user/Desktop/DeepLabCut/') # is there a better way to do this?
print(sys.path)

################################################
## Import deeplabcut
import deeplabcut
from deeplabcut.utils.auxiliaryfunctions import read_config, edit_config
from deeplabcut.generate_training_dataset.trainingsetmanipulation import create_training_dataset
# to check where deeplabcut is coming from: 
print(os.path.abspath(deeplabcut.__file__))

######################################################
### Set config path
config_path = '/Users/user/Desktop/sabris-mouse/sabris-mouse-nirel-2022-07-06/config.yaml'
#cfg = read_config(config_path)

############################################################
## Set other params
NUM_SHUFFLES=1
SHUFFLE_ID=1
MAX_SNAPSHOTS=2 # max snapshots to save
DISPLAY_ITERS=1 # display loss every N iters; one iter processes one batch
SAVE_ITERS=1 # save snapshots every n iters
MAX_ITERS=2
#####################################################
# Create training dataset
create_training_dataset(
    config_path,
    num_shuffles=NUM_SHUFFLES) # augmenter_type=None, posecfg_template=None,

#######################################################
# Edit batchsize in pose config from training
# snapshot folder: '/Users/user/Desktop/sabris-mouse/sabris-mouse-nirel-2022-07-06/dlc-models/iteration-0/sabris-mouseJul6-trainset80shuffle1/train'
train_pose_config_file,\
    test_pose_config_file, \
    snapshot_folder = deeplabcut.return_train_network_path(config_path,
                                                            shuffle=SHUFFLE_ID, 
                                                            trainingsetindex=0, # default
                                                            modelprefix="") # default
#edit_config(str(train_pose_config_file),{'batch_size': 8}) 
edit_config(str(train_pose_config_file),{'contrast': {'clahe': True, 'histeq':True}}) 

# to check if edited correctly:
pose_cfg = read_config(train_pose_config_file)
pose_cfg['batch_size']
######################################################
### Train network
deeplabcut.train_network(config_path,
                            shuffle=SHUFFLE_ID,
                            max_snapshots_to_keep=MAX_SNAPSHOTS,
                            displayiters=DISPLAY_ITERS,
                            saveiters=SAVE_ITERS,
                            maxiters=MAX_ITERS) # allow_growth=True,

