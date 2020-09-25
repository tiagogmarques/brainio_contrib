import os
import numpy as np
from mkgu_packaging.dicarlo.marques.marques_stim_common import gen_grating_stim, gen_blank_stim, load_stim_info
from brainio_collection.packaging import package_stimulus_set

BLANK_STIM_NAME = 'dicarlo.Marques2020_blank'
RF_STIM_NAME = 'dicarlo.Marques2020_receptive_field'
ORIENTATION_STIM_NAME = 'dicarlo.Marques2020_orientation'
SF_STIM_NAME = 'dicarlo.Marques2020_spatial_frequency'
SIZE_STIM_NAME = 'dicarlo.Marques2020_size'

DATA_DIR = '/braintree/data2/active/users/tmarques/bs_stimuli'
DEGREES = 12
SIZE_PX = 672

## All parameters
RF_POS = np.linspace(-2.5, 2.5, 21, endpoint=True)
STIM_POS = np.array([0.5])
RADIUS = np.logspace(-3 + 0.75, 4 - 0.75, 12, endpoint=True, base=2) / 2
SF = np.logspace(-1.5 + 0.125, 4 - 0.125, 22, endpoint=True, base=2)
ORIENTATION = np.linspace(0, 165, 12, endpoint=True)
PHASE = np.linspace(0, 315, 8, endpoint=True)

STIM_NAMES = [RF_STIM_NAME, ORIENTATION_STIM_NAME, SF_STIM_NAME, SIZE_STIM_NAME]

POS_DICT = {RF_STIM_NAME: RF_POS, ORIENTATION_STIM_NAME: STIM_POS, SF_STIM_NAME: STIM_POS, SIZE_STIM_NAME: STIM_POS}
RADIUS_DICT = {RF_STIM_NAME: np.array([1/6]), ORIENTATION_STIM_NAME: np.array([0.25, 0.5, 1]),
               SF_STIM_NAME: np.array([0.75, 2.25]), SIZE_STIM_NAME: RADIUS}
SF_DICT = {RF_STIM_NAME: np.array([3]), ORIENTATION_STIM_NAME: np.array([1, 2, 4]), SF_STIM_NAME: SF,
           SIZE_STIM_NAME: np.array([1, 2, 4])}
ORIENTATION_DICT = {RF_STIM_NAME: ORIENTATION[[0, 3, 6, 9]], ORIENTATION_STIM_NAME: ORIENTATION,
                    SF_STIM_NAME: ORIENTATION[[0, 2, 4, 6, 8, 10]], SIZE_STIM_NAME: ORIENTATION[[0, 2, 4, 6, 8, 10]]}
PHASE_DICT = {RF_STIM_NAME: PHASE[[0, 4]], ORIENTATION_STIM_NAME: PHASE, SF_STIM_NAME: PHASE, SIZE_STIM_NAME: PHASE}


def main():
    blank_dir = DATA_DIR + os.sep + BLANK_STIM_NAME
    if not (os.path.isdir(blank_dir)):
        gen_blank_stim(degrees=DEGREES, size_px=448, save_dir=blank_dir)
    stimuli = load_stim_info(BLANK_STIM_NAME, blank_dir)
    print('Packaging stimuli:' + stimuli.identifier)
    package_stimulus_set(stimuli, stimulus_set_identifier=stimuli.identifier, bucket_name='brainio.dicarlo')

    for stim_name in STIM_NAMES:
        stim_dir = DATA_DIR + os.sep + stim_name
        if not (os.path.isdir(stim_dir)):
            gen_grating_stim(degrees=DEGREES, size_px=SIZE_PX, stim_name=stim_name, grat_contrast=[1],
                             grat_pos=POS_DICT[stim_name], grat_rad=RADIUS_DICT[stim_name], grat_sf=SF_DICT[stim_name],
                             grat_orientation=ORIENTATION_DICT[stim_name], grat_phase=PHASE_DICT[stim_name],
                             save_dir=stim_dir)
        stimuli = load_stim_info(stim_name, stim_dir)
        print('Packaging stimuli:' + stimuli.identifier)
        package_stimulus_set(stimuli, stimulus_set_identifier=stimuli.identifier, bucket_name='brainio.dicarlo')
    return


if __name__ == '__main__':
    main()
