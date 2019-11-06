import os
import numpy as np
from hmmlearn import hmm
# import mfcc_extraction as fe
# import librosa
from sklearn.metrics import roc_auc_score, confusion_matrix
import random
import winsound
import time
from acoustic_feature_extraction import array2vector as a2v

class HMMTrainer(object):
    def __init__(self,model_name='GMMHMM',n_components=1, n_mix=16 ,cov_type='diag',n_iter=300):
        self.model_name = model_name
        self.n_components = n_components
        self.cov_type = cov_type
        self.n_iter = n_iter
        self.n_mix = n_mix
        self.models = []

        if self.model_name == 'GMMHMM':
            self.model = hmm.GMMHMM(n_components=self.n_components,n_mix=self.n_mix ,covariance_type=self.cov_type,n_iter=self.n_iter)
        elif self.model_name == 'GaussianHMM':
            self.model = hmm.GaussianHMM(n_components=self.n_components, covariance_type=self.cov_type, n_iter=self.n_iter)
        else:
            raise TypeError('Invalid model type')


    def train(self,X):
        np.seterr(all='ignore')
        self.models.append(self.model.fit(X))

    def get_score(self,input_data):
        return self.model.score(input_data)

if __name__ == '__main__':
    tic = time.time()
    input_folder_mfcc = 'C:/Amir/Codes/smoker detection/features_mfcc/M/'
    input_folder_pros = 'C:/Amir/Codes/smoker detection/features_prosody/frm/M/'
    input_folder_form = 'C:/Amir/Codes/smoker detection/features_prosody/rec/M/'

    smokers = next(os.walk(input_folder_mfcc + 'Y'))[1]
    nonsmokers = next(os.walk(input_folder_mfcc + 'N'))[1]
    num_smokers = len(smokers)
    num_nonsmokers = len(nonsmokers)

    hmm_models = []
    folds = 5

    # for itx in range(iteration):

    # test_ind_smokers  = np.sort(random.sample(range(num_smokers), int(0.5 * num_smokers / folds)))
    # train_ind_smokers = np.setdiff1d(np.arange(num_smokers), test_ind_smokers)[0:int(0.5*num_smokers*(1-1/folds))]
    # test_ind_nonsmokers  = np.sort(random.sample(range(num_nonsmokers), int(0.5 * num_smokers / folds)))
    # train_ind_nonsmokers = np.setdiff1d(np.arange(num_nonsmokers), test_ind_nonsmokers)[0:len(train_ind_smokers)]

    test_ind_smokers  = np.sort(random.sample(range(num_smokers), int(num_smokers / folds)))
    train_ind_smokers = np.setdiff1d(np.arange(num_smokers), test_ind_smokers)

    test_ind_nonsmokers  = np.sort(random.sample(range(num_nonsmokers), int(num_smokers / folds)))
    train_ind_nonsmokers = np.setdiff1d(np.arange(num_nonsmokers), test_ind_nonsmokers)[0:len(train_ind_smokers)]

    # collect training features and labels for smokers and training the smoker model
    tic_feat_s = time.time()
    print('Smoker feature preparation... ', end='')
    training_features_smokers = np.array([])
    label_smokers = []
    label_smokers_numeric = []
    for ixs in train_ind_smokers:
        spx = 0  # to choose only one recording for each speaker
        for jxs in [x for x in os.listdir(input_folder_mfcc + 'Y/' + smokers[ixs]) if x.endswith('.npy')][:-1]:
            # if spx < 5:
            pros_feat_file_name = input_folder_form + 'Y/' + jxs
            if os.path.isfile(pros_feat_file_name):
                formant_features = np.load(input_folder_form + 'Y/' + jxs)
                if not np.isnan(np.sum(formant_features)):
                    # pitch_harmonicity = np.load(pros_feat_file_name).T
                    # pitch_harmonicity[pitch_harmonicity == 0] = np.nan
                    # pitch_harmonicity = np.nanmean(pitch_harmonicity, axis=0)
                    # pitch_harmonicity = (pitch_harmonicity - np.tile(np.mean(pitch_harmonicity, axis=0),(pitch_harmonicity.shape[0], 1))) / np.tile(np.std(pitch_harmonicity, axis=0, ddof=1), (pitch_harmonicity.shape[0], 1))
                    mfcc_features = np.load(input_folder_mfcc + 'Y/' + smokers[ixs] + '/' + jxs)
                    mfcc_features = np.mean(mfcc_features,axis=0)
                    acoustic_features = a2v(np.concatenate((mfcc_features,formant_features),axis=0)).T
                    if len(training_features_smokers) == 0:
                        training_features_smokers = acoustic_features
                    else:
                        training_features_smokers = np.append(training_features_smokers,acoustic_features,axis=0)
                    spx += 1
                    label_smokers.append('Y')
                    label_smokers_numeric.append(1)

    label_smokers_numeric = np.array(label_smokers_numeric)
    toc_feat_s = time.time()
    print(' Done! ---> Elapsed Time: %2.2f sec.' % (toc_feat_s - tic_feat_s))

    tic_tr_sm = time.time()
    print('Training the smoker model... ', end='')
    hmm_trainer = HMMTrainer()
    hmm_trainer.train(training_features_smokers)
    hmm_models.append((hmm_trainer, label_smokers))
    hmm_trainer = None
    toc_tr_smk = time.time()
    print('Done! ---> Elapsed Time: %2.2f min.' % ((toc_tr_smk - tic_tr_sm)/60))


    # collect training features and labels for non-smokers and training the non-smoker model
    tic_feat_n = time.time()
    print('Non-smoker feature preparation... ', end='')
    training_features_nonsmokers = np.array([])
    label_nonsmokers = []
    label_nonsmokers_numeric = []
    for ixs in train_ind_nonsmokers:
        spx = 0 # to choose only one recording for each speaker
        for jxs in [x for x in os.listdir(input_folder_mfcc + 'N/' + nonsmokers[ixs]) if x.endswith('.npy')][:-1]:
            # if spx < 5:
            pros_feat_file_name = input_folder_form + 'N/' + jxs
            if os.path.isfile(pros_feat_file_name):
                formant_features = np.load(input_folder_form + 'N/' + jxs)
                if not np.isnan(np.sum(formant_features)):
                    mfcc_features = np.load(input_folder_mfcc + 'N/' + nonsmokers[ixs] + '/' + jxs)
                    mfcc_features = np.mean(mfcc_features, axis=0)
                    acoustic_features = a2v(np.concatenate((mfcc_features,formant_features),axis=0)).T
                    if len(training_features_nonsmokers) == 0:
                        training_features_nonsmokers = acoustic_features
                    else:
                        training_features_nonsmokers = np.append(training_features_nonsmokers, acoustic_features, axis=0)
                    spx += 1
                    label_nonsmokers.append('N')
                    label_nonsmokers_numeric.append(2)

    label_nonsmokers_numeric = np.array(label_nonsmokers_numeric)
    toc_feat_n = time.time()
    print(' Done! ---> Elapsed Time: %2.2f sec.' % (toc_feat_n - tic_feat_n))

    tic_tr_nm = time.time()
    print('Training the non-smoker model... ', end='')
    hmm_trainer = HMMTrainer()
    hmm_trainer.train(training_features_nonsmokers)
    hmm_models.append((hmm_trainer, label_nonsmokers))
    hmm_trainer = None
    toc_tr_nsmk = time.time()
    print('Done! ---> Elapsed Time: %2.2f min.' % ((toc_tr_nsmk - tic_tr_nm) / 60))


    # Testing Phase
    tic_scoring = time.time()
    print('Scoring... ', end='')
    all_scores = [] # np.zeros((len(test_ind_smokers)+len(test_ind_nonsmokers),2))
    test_label = []
    test_label_numeric = []
    for ixs in test_ind_smokers:
        for jxs in [x for x in os.listdir(input_folder_mfcc + 'Y/' + smokers[ixs]) if x.endswith('.npy')][:-1]:
            pros_feat_file_name = input_folder_form + 'Y/' + jxs
            if os.path.isfile(pros_feat_file_name):
                formant_features = np.load(input_folder_form + 'Y/' + jxs)
                if not np.isnan(np.sum(formant_features)):
                    mfcc_features = np.load(input_folder_mfcc + 'Y/' + smokers[ixs] + '/' + jxs)
                    mfcc_features = np.mean(mfcc_features, axis=0)
                    acoustic_features = a2v(np.concatenate((mfcc_features,formant_features),axis=0)).T

                    score = np.array([0,0])
                    cx = 0
                    for item in hmm_models:
                        hmm_model, label = item
                        score[cx] = hmm_model.get_score(acoustic_features)
                        cx += 1
                    all_scores.append(score)
                    test_label.append('Y')
                    test_label_numeric.append(1)


    for ixn in test_ind_nonsmokers:
        for jxn in [x for x in os.listdir(input_folder_mfcc + 'N/' + nonsmokers[ixn]) if x.endswith('.npy')][:-1]:
            pros_feat_file_name = input_folder_form + 'N/' + jxn
            if os.path.isfile(pros_feat_file_name):
                formant_features = np.load(input_folder_form + 'N/' + jxn)
                if not np.isnan(np.sum(formant_features)):
                    mfcc_features = np.load(input_folder_mfcc + 'N/' + nonsmokers[ixn] + '/' + jxn)
                    mfcc_features = np.mean(mfcc_features, axis=0)
                    acoustic_features = a2v(np.concatenate((mfcc_features,formant_features),axis=0)).T
                    score = np.array([0,0])
                    cx = 0
                    for item in hmm_models:
                        hmm_model, label = item
                        score[cx] = hmm_model.get_score(acoustic_features)
                        cx += 1
                    all_scores.append(score)
                    test_label.append('N')
                    test_label_numeric.append(2)

    test_label_numeric = np.array(test_label_numeric)
    all_scores = np.array(all_scores)
    auc_0 = roc_auc_score(test_label_numeric, all_scores[:,0])
    auc_1 = roc_auc_score(test_label_numeric, all_scores[:, 1])
    predict_labels = 1 + np.argmax(all_scores,axis=1)
    cm = confusion_matrix(test_label_numeric, predict_labels)
    toc_scoring = time.time()
    print('Done! ---> Elapsed Time: %2.2f sec.' % (toc_scoring - tic_scoring))
    print('AUC_0 = %1.2f ' % auc_0)
    print('AUC_1 = %1.2f ' % auc_1)
    print('Accuracy = %2.2f ' % (100*np.sum(np.diag(cm))/(np.sum(cm))))
    print(cm)

    toc_end = time.time()
    print('Total Processing Time: %2.2f min.' % ((toc_end - tic)/60))
    winsound.Beep(233, 500)
    winsound.Beep(293, 500)
    winsound.Beep(349, 500)






