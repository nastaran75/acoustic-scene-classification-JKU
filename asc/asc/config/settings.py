
import os
import getpass

# get user name
user = getpass.getuser()

# get hostname
hostname = os.uname()[1]

# paths on compute nodes
if hostname in ["rechenknecht0.cp.jku.at", "rechenknecht1.cp.jku.at"]:
    DATA_ROOT_LITIS = "/media/rk0/shared/datasets/LITIS/"
    DATA_ROOT_JAMENDO = "/media/rk0/shared/datasets/jamendo/"
    DATA_ROOT_TUT17 = "~/nastaran/Acoustic_Scene_Analysis/DCASE/DCASE2016-Wav-Files/TUT-acoustic-scenes-2016-development/"
    DATA_ROOT_AUDIO2CF = "/media/rk0/shared/datasets/audio2cf/"

    if user == 'matthias':
        EXP_ROOT = "/home/matthias/experiments/asc/"

# set on local machines
else:

    if user == 'matthias':
        DATA_ROOT_LITIS = "/media/matthias/Data/LITIS/"
        DATA_ROOT_JAMENDO = "/media/matthias/Data/jamendo/"
        DATA_ROOT_TUT17 = "/media/matthias/Data/TUT_scene_clf17/TUT-acoustic-scenes-2017-development/"
        EXP_ROOT = "/home/matthias/experiments/asc/"
        DATA_ROOT_AUDIO2CF = "/home/matthias/mounts/home@rechenknecht0/shared/datasets/audio2cf/"
