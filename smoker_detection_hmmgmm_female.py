import os
import numpy as np
from hmmlearn import hmm
# import mfcc_extraction as fe
# import librosa
from sklearn.metrics import roc_auc_score, confusion_matrix
import random
import winsound
import time

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
    input_folder = 'C:/Amir/Codes/smoker detection/features_mfcc/F/'
    smokers = next(os.walk(input_folder + 'Y'))[1]
    nonsmokers = next(os.walk(input_folder + 'N'))[1]
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
        for jxs in [x for x in os.listdir(input_folder + 'Y/' + smokers[ixs]) if x.endswith('.npy')][:-1]:
            if spx < 1:
                mfcc_features = np.load(input_folder + 'Y/' + smokers[ixs]+'/'+ jxs)
                # mfcc_features = mfcc_features[:,1:]
                # mfcc_features = (mfcc_features - np.tile(np.mean(mfcc_features,axis=0), (mfcc_features.shape[0],1))) / np.tile(np.std(mfcc_features,axis=0,ddof=1),(mfcc_features.shape[0], 1))
                if len(training_features_smokers) == 0:
                    training_features_smokers = mfcc_features
                else:
                    training_features_smokers = np.append(training_features_smokers,mfcc_features,axis=0)
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
        for jxs in [x for x in os.listdir(input_folder + 'N/' + nonsmokers[ixs]) if x.endswith('.npy')][:-1]:
            if spx < 1:
                mfcc_features = np.load(input_folder + 'N/' + nonsmokers[ixs] + '/' + jxs)
                # mfcc_features = mfcc_features[:,1:]
                # mfcc_features = (mfcc_features - np.tile(np.mean(mfcc_features, axis=0),(mfcc_features.shape[0], 1))) / np.tile(np.std(mfcc_features, axis=0, ddof=1), (mfcc_features.shape[0], 1))

                if len(training_features_nonsmokers) == 0:
                    training_features_nonsmokers = mfcc_features
                else:
                    training_features_nonsmokers = np.append(training_features_nonsmokers, mfcc_features, axis=0)
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
        for jxs in [x for x in os.listdir(input_folder + 'Y/' + smokers[ixs]) if x.endswith('.npy')][:-1]:
            mfcc_features = np.load(input_folder + 'Y/' + smokers[ixs]+'/'+ jxs)
            # mfcc_features = mfcc_features[:,1:]
            # mfcc_features = (mfcc_features - np.tile(np.mean(mfcc_features, axis=0),(mfcc_features.shape[0], 1))) / np.tile(np.std(mfcc_features, axis=0, ddof=1), (mfcc_features.shape[0], 1))

            score = np.array([0,0])
            cx = 0
            for item in hmm_models:
                hmm_model, label = item
                score[cx] = hmm_model.get_score(mfcc_features)
                cx += 1
            all_scores.append(score)
            test_label.append('Y')
            test_label_numeric.append(1)


    for ixn in test_ind_nonsmokers:
        for jxn in [x for x in os.listdir(input_folder + 'N/' + nonsmokers[ixn]) if x.endswith('.npy')][:-1]:
            mfcc_features = np.load(input_folder + 'N/' + nonsmokers[ixn]+'/'+ jxn)
            # mfcc_features = mfcc_features[:,1:]
            # mfcc_features = (mfcc_features - np.tile(np.mean(mfcc_features, axis=0),(mfcc_features.shape[0], 1))) / np.tile(np.std(mfcc_features, axis=0, ddof=1), (mfcc_features.shape[0], 1))

            score = np.array([0,0])
            cx = 0
            for item in hmm_models:
                hmm_model, label = item
                score[cx] = hmm_model.get_score(mfcc_features)
                cx += 1
            all_scores.append(score)
            test_label.append('N')
            test_label_numeric.append(2)

    test_label_numeric = np.array(test_label_numeric)
    all_scores = np.array(all_scores)
    auc = roc_auc_score(test_label_numeric, all_scores[:,0])
    predict_labels = 1 + np.argmax(all_scores,axis=1)
    cm = confusion_matrix(test_label_numeric, predict_labels)
    toc_scoring = time.time()
    print('Done! ---> Elapsed Time: %2.2f sec.' % (toc_scoring - tic_scoring))
    print('AUC = %1.2f ' % auc)
    print('Accuracy = %2.2f ' % (100*np.sum(np.diag(cm))/(np.sum(cm))))
    print(cm)

    toc_end = time.time()
    print('Total Processing Time: %2.2f min.' % ((toc_end - tic)/60))
    winsound.Beep(233, 500)
    winsound.Beep(293, 500)
    winsound.Beep(349, 500)
