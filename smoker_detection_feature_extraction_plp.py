import os
import numpy as np
import acoustic_feature_extraction as fe
import librosa
import pandas as pd
from os import listdir


PLP_Param = {'NFFT':512,'lpc_order': 12,'FLT': 0.030,'FST': 0.020, 'vad_flag':1}

path_calls = 'C:/Amir/Data/Mixer6/data/docs/mx6_calls.csv'
path_subjs = 'C:/Amir/Data/Mixer6/data/docs/mx6_subjs.csv'

call_id   = np.asarray(pd.read_csv(path_calls, usecols=['call_id']).T.values.tolist()[0])
spk_a     = np.asarray(pd.read_csv(path_calls, usecols=['sid_a']).T.values.tolist()[0])
spk_b     = np.asarray(pd.read_csv(path_calls, usecols=['sid_b']).T.values.tolist()[0])
speakers  = np.asarray(pd.read_csv(path_subjs, usecols=['subjid']).T.values.tolist()[0])
gender    = np.asarray(pd.read_csv(path_subjs, usecols=['sex']).T.values.tolist()[0])
smoke_lbl = np.asarray(pd.read_csv(path_subjs, usecols=['smoker']).T.values.tolist()[0])

sourcefoldedr = 'C:/Amir/Data/Mixer6/data/ulaw_sphere/'
destinationfolder = 'C:/Amir/Codes/smoker detection/features_plp/'
all_files = [f for f in listdir(sourcefoldedr) if f.endswith('.sph')]
total_file = len(all_files)
fidx = 0
nmdx = 0
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
            plp_features = fe.main_rasta_plp_function(norm_sig_a, fs, PLP_Param)[0]
            np.save(directory_a + str(id_spk_a[0]) + '_' + filename[0:-4]+'.npy',plp_features)


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
            plp_features = fe.main_rasta_plp_function(norm_sig_b, fs, PLP_Param)[0]
            np.save(directory_b + str(id_spk_b[0]) + '_' + filename[0:-4] + '.npy', plp_features)

    fidx += 1

