import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve,auc
import librosa
from acoustic_feature_extraction import array2vector as a2v
from acoustic_feature_extraction import calculate_CI as CI
import acoustic_feature_extraction as fe
import pandas as pd
import matplotlib.pyplot as plt
from os import listdir


path_training_filelist = './filelists/training_filelists/' # path to training file lists
path_test_filelist     = './filelists/test_filelists/' # path to test file lists
sourcefoldedr = './Mixer6/data/ulaw_sphere/' # path to audio files

MFCCParam = {'NumFilters':27,'NFFT':512,'FminHz':0,'FMaxHz':4000,'no':12,'FLT':0.030,'FST':0.020,'vad_flag':1} # MFCC parameters
#### Preparing lists to save the restlts of each iteration ##########
accuracy = []
accuracy_top5p = []
accuracy_top10p = []
accuracy_top15p = []
accuracy_top20p = []
accuracy_top25p = []
accuracy_top30p = []
accuracy_top35p = []
accuracy_top40p = []
accuracy_top45p = []
accuracy_top50p = []
tprs = []
aucs = []
mean_fpr  = np.linspace(0, 1, 100)
mean_pmls = np.linspace(0,50,100) # PMLS: percentage of most likely smokers
top_x_accuracy = []
top_x_accuracies = []
#####################

training_filelists = [f for f in listdir(path_training_filelist) if f.endswith('.csv')] # list of all training file lists
test_filelists     = [f for f in listdir(path_test_filelist) if f.endswith('.csv')] # list of all test file lists
total_training_file = len(training_filelists) # total number of file lists (number of iterations)

for itx in range(total_training_file): # load file lists one-by-one and process recordings according to the corresponding file list
    trn_filenames = np.asarray(pd.read_csv(path_training_filelist + training_filelists[itx], usecols=['file name']).T.values.tolist()[0]) # audio file names in the file lists
    trn_labels    = np.asarray(pd.read_csv(path_training_filelist + training_filelists[itx], usecols=['smoking label']).T.values.tolist()[0]) # training labels
    trn_channels  = np.asarray(pd.read_csv(path_training_filelist + training_filelists[itx], usecols=['channel']).T.values.tolist()[0]) # channels of the signals to be processed

    training_features_mfcc = np.array([]) # preparing an array to save features of all training data
    trn_label_numeric = [] # prepare a list to save the numeric training labels (SMK:1, NSMK:2)
    for rx in range(len(trn_filenames)): # load and process recordings one-by-one
        audio_file_name = sourcefoldedr + trn_filenames[rx] # audio file name: directory/file
        signal, fs=librosa.load(audio_file_name, mono=False, sr=None) # loading audio signal (stereo)
        signal = signal[trn_channels[rx]] # peak the corresponding channel (according to the Mixer6 tabels)
        maxamp = np.max(abs(signal))
        norm_sig = signal / maxamp # normalize signal amplitude
        mfcc_features = fe.main_mfcc_function(norm_sig, fs, MFCCParam)[0] # extract MFCC. It has framed signal [1] and VAD indices [2], but we are interested in features [0]
        mfcc_mean = np.mean(mfcc_features, axis=0) # average over frames
        acoustic_features_mfcc = a2v(mfcc_mean).T # convert it to vector to be able to append to other features (of other training recordings)
        if len(training_features_mfcc) == 0: # when the array is empty (first time appending)
            training_features_mfcc = acoustic_features_mfcc
        else: # when the array is no longer empty
            training_features_mfcc = np.append(training_features_mfcc, acoustic_features_mfcc, axis=0)
        if trn_labels[rx] == 'Y': # make numeric labels (Y:smoker --> 1, N:nonsmoker --> 2)
            trn_label_numeric.append(1)
        elif trn_labels[rx] == 'N': # make numeric labels (Y:smoker --> 1, N:nonsmoker --> 2)
            trn_label_numeric.append(2)
    trn_label_numeric = np.array(trn_label_numeric) # convert the list to array

    lr_model_mfcc = LogisticRegression(solver='liblinear', class_weight='balanced', max_iter=1000).fit(training_features_mfcc, trn_label_numeric) # training the logistic regression

    ######## Testing Phase ###################################################################
    tst_filenames = np.asarray(pd.read_csv(path_test_filelist + test_filelists[itx], usecols=['file name']).T.values.tolist()[0])
    tst_labels    = np.asarray(pd.read_csv(path_test_filelist + test_filelists[itx], usecols=['smoking label']).T.values.tolist()[0])
    tst_channels  = np.asarray(pd.read_csv(path_test_filelist + test_filelists[itx], usecols=['channel']).T.values.tolist()[0])
    #######################################################################
    test_features_mfcc = np.array([])
    tst_label_numeric = []
    for rx in range(len(tst_filenames)):
        audio_file_name = sourcefoldedr + tst_filenames[rx]
        signal, fs = librosa.load(audio_file_name, mono=False, sr=None)
        signal = signal[tst_channels[rx]]
        maxamp = np.max(abs(signal))
        norm_sig = signal / maxamp
        mfcc_features = fe.main_mfcc_function(norm_sig, fs, MFCCParam)[0]
        mfcc_mean = np.mean(mfcc_features, axis=0)
        acoustic_features_mfcc = a2v(mfcc_mean).T
        if len(test_features_mfcc) == 0:
            test_features_mfcc = acoustic_features_mfcc
        else:
            test_features_mfcc = np.append(test_features_mfcc, acoustic_features_mfcc, axis=0)
        if tst_labels[rx] == 'Y':
            tst_label_numeric.append(1)
        elif tst_labels[rx] == 'N':
            tst_label_numeric.append(2)

    tst_label_numeric = np.array(tst_label_numeric)

    predicted_labels = lr_model_mfcc.predict(test_features_mfcc) # hard deciison labels
    probability_prediction = lr_model_mfcc.predict_proba(test_features_mfcc) # soft scores. Prediction probabilities

################ Performance Analysis ##############
    ranked_scores = np.flipud(np.argsort(probability_prediction[:, 0]))
    top5p  = ranked_scores[0:int(0.05 * len(ranked_scores))]
    top10p = ranked_scores[0:int(0.10 * len(ranked_scores))]
    top15p = ranked_scores[0:int(0.15 * len(ranked_scores))]
    top20p = ranked_scores[0:int(0.20 * len(ranked_scores))]
    top25p = ranked_scores[0:int(0.25 * len(ranked_scores))]
    top30p = ranked_scores[0:int(0.30 * len(ranked_scores))]
    top35p = ranked_scores[0:int(0.35 * len(ranked_scores))]
    top40p = ranked_scores[0:int(0.40 * len(ranked_scores))]
    top45p = ranked_scores[0:int(0.45 * len(ranked_scores))]
    top50p = ranked_scores[0:int(0.50 * len(ranked_scores))]

    test_labels_top5p  = tst_label_numeric[top5p]
    test_labels_top10p = tst_label_numeric[top10p]
    test_labels_top15p = tst_label_numeric[top15p]
    test_labels_top20p = tst_label_numeric[top20p]
    test_labels_top25p = tst_label_numeric[top25p]
    test_labels_top30p = tst_label_numeric[top30p]
    test_labels_top35p = tst_label_numeric[top35p]
    test_labels_top40p = tst_label_numeric[top40p]
    test_labels_top45p = tst_label_numeric[top45p]
    test_labels_top50p = tst_label_numeric[top50p]

    predicted_labels_top5p  = predicted_labels[top5p]
    predicted_labels_top10p = predicted_labels[top10p]
    predicted_labels_top15p = predicted_labels[top15p]
    predicted_labels_top20p = predicted_labels[top20p]
    predicted_labels_top25p = predicted_labels[top25p]
    predicted_labels_top30p = predicted_labels[top30p]
    predicted_labels_top35p = predicted_labels[top35p]
    predicted_labels_top40p = predicted_labels[top40p]
    predicted_labels_top45p = predicted_labels[top45p]
    predicted_labels_top50p = predicted_labels[top50p]

    acc_top5p  = accuracy_score(test_labels_top5p, predicted_labels_top5p)
    acc_top10p = accuracy_score(test_labels_top10p, predicted_labels_top10p)
    acc_top15p = accuracy_score(test_labels_top15p, predicted_labels_top15p)
    acc_top20p = accuracy_score(test_labels_top20p, predicted_labels_top20p)
    acc_top25p = accuracy_score(test_labels_top25p, predicted_labels_top25p)
    acc_top30p = accuracy_score(test_labels_top30p, predicted_labels_top30p)
    acc_top35p = accuracy_score(test_labels_top35p, predicted_labels_top35p)
    acc_top40p = accuracy_score(test_labels_top40p, predicted_labels_top40p)
    acc_top45p = accuracy_score(test_labels_top45p, predicted_labels_top45p)
    acc_top50p = accuracy_score(test_labels_top50p, predicted_labels_top50p)

    fpr, tpr, thresholds = roc_curve(tst_label_numeric, probability_prediction[:, 0], pos_label=1)
    roc_auc = auc(fpr, tpr)
    acc = accuracy_score(tst_label_numeric, predicted_labels)

    tprs.append(np.interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    aucs.append(roc_auc)
    accuracy = np.append(accuracy, acc)
    top_x_acc = np.array(
        [acc_top5p, acc_top10p, acc_top15p, acc_top20p, acc_top25p, acc_top30p, acc_top35p, acc_top40p, acc_top45p,
         acc_top50p])
    top_x_accuracy.append(top_x_acc)
    top_x_accuracies.append(np.interp(mean_pmls, np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 50]), top_x_acc))
    print(
        'acc: %2.1f --> AUC: %2.1f --> top5: %2.1f --> top10: %2.1f --> top15: %2.1f --> top20: %2.1f --> top25: %2.1f --> top30: %2.1f' % (
            100 * acc, roc_auc, 100 * acc_top5p, 100 * acc_top10p, 100 * acc_top15p, 100 * acc_top20p,
            100 * acc_top25p,
            100 * acc_top30p))
    accuracy_top5p = np.append(accuracy_top5p, acc_top5p)
    accuracy_top10p = np.append(accuracy_top10p, acc_top10p)
    accuracy_top15p = np.append(accuracy_top15p, acc_top15p)
    accuracy_top20p = np.append(accuracy_top20p, acc_top20p)
    accuracy_top25p = np.append(accuracy_top25p, acc_top25p)
    accuracy_top30p = np.append(accuracy_top30p, acc_top30p)

print('Accuracy overal = %2.1f -+ %2.1f' % (100 * np.mean(np.array(accuracy)), 100 * np.std(np.array(accuracy), ddof=1)))
print('Accuracy top 5  = %2.1f -+ %2.1f' % (100 * np.mean(np.array(accuracy_top5p)), 100 * np.std(np.array(accuracy_top5p), ddof=1)))
print('Accuracy top 10 = %2.1f -+ %2.1f' % (100 * np.mean(np.array(accuracy_top10p)), 100 * np.std(np.array(accuracy_top10p), ddof=1)))
print('Accuracy top 15 = %2.1f -+ %2.1f' % (100 * np.mean(np.array(accuracy_top15p)), 100 * np.std(np.array(accuracy_top15p), ddof=1)))
print('Accuracy top 20 = %2.1f -+ %2.1f' % (100 * np.mean(np.array(accuracy_top20p)), 100 * np.std(np.array(accuracy_top20p), ddof=1)))
print('Accuracy top 25 = %2.1f -+ %2.1f' % (100 * np.mean(np.array(accuracy_top25p)), 100 * np.std(np.array(accuracy_top25p), ddof=1)))
print('Accuracy top 30 = %2.1f -+ %2.1f' % (100 * np.mean(np.array(accuracy_top30p)), 100 * np.std(np.array(accuracy_top30p), ddof=1)))

########## Plot ROC ###########
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs, ddof=1)
ci_auc = CI(mean_auc, std_auc, total_training_file, 95)[0]
std_tpr = np.std(tprs, axis=0, ddof=1)
tprs_std_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_std_lower = np.maximum(mean_tpr - std_tpr, 0)
ci_tprs, tprs_lower, tprs_upper = CI(mean_tpr, std_tpr, total_training_file, 95)
plt.figure(0)
plt.rcParams.update({'font.size': 24})
plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f (CI))' % (mean_auc, ci_auc), lw=2,alpha=.8)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='red', alpha=.2, label=r'$\pm$ 95% CI')
plt.fill_between(mean_fpr, tprs_std_lower, tprs_std_upper, color='blue', alpha=.2, label=r'$\pm$ 1 STD')
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
plt.xlim([0, 1.01])
plt.ylim([0, 1.01])
plt.xlabel('False Positive Rate', fontsize=24)
plt.ylabel('True Positive Rate', fontsize=24)
plt.title('Receiver Operating Characteristic', fontsize=26)
plt.legend(loc="lower right")
plt.show()
########## Plot Lift Chart ###########
mean_top_x_accuracies = np.mean(top_x_accuracies, axis=0)
std_top_x_accuracies = np.std(top_x_accuracies, axis=0, ddof=1)
std_top_x_accuracies_upper = np.minimum(mean_top_x_accuracies + std_top_x_accuracies, 1)
std_top_x_accuracies_lower = np.maximum(mean_top_x_accuracies - std_top_x_accuracies, 0)
ci_mean_top_x_accuracies, mean_top_x_accuracies_lower, mean_top_x_accuracies_upper = CI(mean_top_x_accuracies,std_top_x_accuracies, total_training_file,95)
plt.figure(1)
plt.plot(mean_pmls, 100 * mean_top_x_accuracies, color='b', label=r'Mean Accuracy', lw=2, alpha=.8)
plt.fill_between(mean_pmls, 100 * mean_top_x_accuracies_lower, 100 * mean_top_x_accuracies_upper, color='red', alpha=.2,label=r'$\pm$ 95% CI')
plt.fill_between(mean_pmls, 100 * std_top_x_accuracies_lower, 100 * std_top_x_accuracies_upper, color='blue', alpha=.2,label=r'$\pm$ 1 STD')
plt.xlim([0, 50.01])
plt.ylim([20, 100])
plt.xlabel('Percentage Most Likely Smokers', fontsize=24)
plt.ylabel('Percentage Actual Smokers', fontsize=24)
plt.title('Lift Chart', fontsize=26)
plt.legend(loc="lower right")
plt.grid(True)
plt.show()
print('done')