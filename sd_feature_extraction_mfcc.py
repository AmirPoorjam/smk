#######################################################################################################################
# This module extracts 13 MFCC coefficients from all Mixer6 recordings. It reads data from two Mixer6 tabels          #
# (mx6_calls.csv and mx6_subjs.csv), and make two folders M:male and F:female. In each of the folders it makes two    #
# subfolders Y: smokers and N: non-smokers. Then, it makes a folder for each speakers and saves features of all       #
# recordings from each speaker.                                                                                       #
#                                                                                                                     #
###### Amir H. Poorjam, November 2019 #################################################################################

import os
import numpy as np
import acoustic_feature_extraction as fe
import librosa
import pandas as pd
from os import listdir

MFCCParam = {'NumFilters': 27,'NFFT':512,'FminHz':0,'FMaxHz':4000,'no':12,'FLT':0.030,'FST':0.020,'vad_flag':1}
##### Setting the paths #######
path_to_tabels = input('Please enter the path to the Mixer6 tables: ')
sourcefoldedr  = input('Please enter the path to the Mixer6 raw audio files (.sph): ')
destinationfolder = './features_mfcc/'
if path_to_tabels[-1] != '/':
    path_to_tabels = path_to_tabels + '/'
if sourcefoldedr[-1] != '/':
    sourcefoldedr = sourcefoldedr + '/'

##### Reading data from tables #######
call_id   = np.asarray(pd.read_csv(path_to_tabels + 'mx6_calls.csv', usecols=['call_id']).T.values.tolist()[0])
spk_a     = np.asarray(pd.read_csv(path_to_tabels + 'mx6_calls.csv', usecols=['sid_a']).T.values.tolist()[0])
spk_b     = np.asarray(pd.read_csv(path_to_tabels + 'mx6_calls.csv', usecols=['sid_b']).T.values.tolist()[0])
speakers  = np.asarray(pd.read_csv(path_to_tabels + 'mx6_subjs.csv', usecols=['subjid']).T.values.tolist()[0])
gender    = np.asarray(pd.read_csv(path_to_tabels + 'mx6_subjs.csv', usecols=['sex']).T.values.tolist()[0])
smoke_lbl = np.asarray(pd.read_csv(path_to_tabels + 'mx6_subjs.csv', usecols=['smoker']).T.values.tolist()[0])

##### Finding metadata for each recording, extracting features, and saving with the corresponding names #####
all_files = [f for f in listdir(sourcefoldedr) if f.endswith('.sph')]
total_file = len(all_files)
fidx = 0 # will count the processed files
nmdx = 0 # will count the files that do not have metadata
for filename in all_files:
    print('%d / %d ' % (fidx + 1, total_file))
    signals, fs = librosa.load(sourcefoldedr+filename, mono=False, sr=None)
    file_handle = int(filename[16:-4])
    index = np.where(call_id == file_handle)

    id_spk_a = spk_a[index]
    id_spk_b = spk_b[index]

    ind_spk_a = np.where(speakers == id_spk_a)
    ind_spk_b = np.where(speakers == id_spk_b)

    gender_spk_a = gender[ind_spk_a]
    gender_spk_b = gender[ind_spk_b]

    smoke_spk_a = smoke_lbl[ind_spk_a]
    smoke_spk_b = smoke_lbl[ind_spk_b]

    if gender_spk_a=='nan' or smoke_spk_a=='nan':
        nmdx += 1
        print('No metadata for speaker %s --> count: %d' % (id_spk_a[0],nmdx))
    else:
        directory_a = destinationfolder + gender_spk_a[0] + '/' + smoke_spk_a[0] + '/' + str(id_spk_a[0]) + '/'
        if not os.path.exists(directory_a):
            os.makedirs(directory_a)
        maxamp_a = np.max(abs(signals[0]))
        if maxamp_a != 0:
            norm_sig_a = signals[0] * (1 / maxamp_a)
            mfcc_features = fe.main_mfcc_function(norm_sig_a, fs, MFCCParam)[0]
            np.save(directory_a + str(id_spk_a[0]) + '_' + filename[0:-4]+'.npy',mfcc_features)
    if gender_spk_b=='nan' or smoke_spk_b=='nan':
        nmdx += 1
        print('No metadata for speaker %s --> count: %d' % (id_spk_b[0],nmdx))
    else:
        directory_b = destinationfolder + gender_spk_b[0] + '/' + smoke_spk_b[0] + '/' + str(id_spk_b[0]) + '/'
        if not os.path.exists(directory_b):
            os.makedirs(directory_b)
        maxamp_b = np.max(abs(signals[1]))
        if maxamp_b != 0:
            norm_sig_b = signals[1] * (1 / maxamp_b)
            mfcc_features = fe.main_mfcc_function(norm_sig_b, fs, MFCCParam)[0]
            np.save(directory_b + str(id_spk_b[0]) + '_' + filename[0:-4] + '.npy', mfcc_features)
    fidx += 1

