import os
import numpy as np
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, precision_recall_fscore_support
import random
import matplotlib.pyplot as plt
from acoustic_feature_extraction import array2vector as a2v
from scipy import stats
import seaborn as sns
import pandas as pd


if __name__ == '__main__':
    input_folder_mfcc = 'C:/Amir/Codes/smoker detection/features_mfcc/M/'
    input_folder_pros = 'C:/Amir/Codes/smoker detection/features_prosody/frm/M/'
    input_folder_form = 'C:/Amir/Codes/smoker detection/features_prosody/rec/M/'

    smokers = next(os.walk(input_folder_mfcc + 'Y'))[1]
    nonsmokers = next(os.walk(input_folder_mfcc + 'N'))[1]
    num_smokers = len(smokers)
    num_nonsmokers = len(nonsmokers)

    folds = 5
    itr = 20
    FRAME_AVG = 50
    auc0 = []
    auc1 = []
    accuracy = []
    precision = []
    recall = []
    fscore = []
    accuracy_top5p = []
    accuracy_top10p = []
    accuracy_top15p = []
    accuracy_top20p = []
    accuracy_top25p = []
    accuracy_top30p = []

    for irx in range(itr):
        print('%d / %d --> ' % (irx + 1, itr),end='')
        test_ind_smokers  = np.sort(random.sample(range(num_smokers), int(num_smokers / folds)))
        train_ind_smokers = np.setdiff1d(np.arange(num_smokers), test_ind_smokers)

        test_ind_nonsmokers  = np.sort(random.sample(range(num_nonsmokers), int(num_smokers / folds)))
        train_ind_nonsmokers = np.setdiff1d(np.arange(num_nonsmokers), test_ind_nonsmokers)[0:len(train_ind_smokers)]

        # collect training features and labels for smokers and training the smoker model
        training_features = np.array([])
        training_labels_frm = np.array([])
        training_labels_rec = []
        for ixs in train_ind_smokers:
            spx = 0  # to choose only one recording for each speaker
            for jxs in [x for x in os.listdir(input_folder_mfcc + 'Y/' + smokers[ixs]) if x.endswith('.npy')][:-1]:
                if spx < 1:
                    mfcc = np.load(input_folder_mfcc + 'Y/' + smokers[ixs] + '/' + jxs)
                    if mfcc.shape[0]>FRAME_AVG:
                        mfcc_truncated = mfcc[:FRAME_AVG * int(mfcc.shape[0] / FRAME_AVG), :]
                        frames_indx = np.arange(mfcc_truncated.shape[0])
                        cell_len = int(mfcc_truncated.shape[0] / FRAME_AVG)
                        Frm_indx = np.reshape(frames_indx, (FRAME_AVG, cell_len), order='F')
                        MFCCs_matrix = a2v(np.mean(mfcc_truncated[Frm_indx[:, 0], :], axis=0))
                        for i in range(1, cell_len):
                            new_column = a2v(np.nanmean(mfcc_truncated[Frm_indx[:, i], :], axis=0))
                            MFCCs_matrix = np.concatenate((MFCCs_matrix, new_column), axis=1)
                        acoustic_features = MFCCs_matrix.T
                        if len(training_features) == 0:
                            training_features = acoustic_features
                            training_labels_frm = 1*np.ones((1,acoustic_features.shape[0]))
                        else:
                            training_features = np.append(training_features,acoustic_features,axis=0)
                            training_labels_frm = np.append(training_labels_frm, 1*np.ones((1,acoustic_features.shape[0])), axis=1)
                        spx += 1
                        training_labels_rec.append(1)

        # collect training features and labels for non-smokers and training the non-smoker model
        for ixs in train_ind_nonsmokers:
            spx = 0 # to choose only one recording for each speaker
            for jxs in [x for x in os.listdir(input_folder_mfcc + 'N/' + nonsmokers[ixs]) if x.endswith('.npy')][:-1]:
                if spx < 1:
                    mfcc = np.load(input_folder_mfcc + 'N/' + nonsmokers[ixs] + '/' + jxs)
                    if mfcc.shape[0]>FRAME_AVG:
                        mfcc_truncated = mfcc[:FRAME_AVG * int(mfcc.shape[0] / FRAME_AVG), :]
                        frames_indx = np.arange(mfcc_truncated.shape[0])
                        cell_len = int(mfcc_truncated.shape[0] / FRAME_AVG)
                        Frm_indx = np.reshape(frames_indx, (FRAME_AVG, cell_len), order='F')
                        MFCCs_matrix = a2v(np.mean(mfcc_truncated[Frm_indx[:, 0], :], axis=0))
                        for i in range(1, cell_len):
                            new_column = a2v(np.nanmean(mfcc_truncated[Frm_indx[:, i], :], axis=0))
                            MFCCs_matrix = np.concatenate((MFCCs_matrix, new_column), axis=1)
                        acoustic_features = MFCCs_matrix.T
                        training_features = np.append(training_features, acoustic_features, axis=0)
                        training_labels_frm = np.append(training_labels_frm, 2 * np.ones((1, acoustic_features.shape[0])), axis=1)
                        spx += 1
                        training_labels_rec.append(2)

        training_labels_rec = np.array(training_labels_rec)
        training_labels = np.ravel(np.array(training_labels_frm))
        # model_classifier = svm.SVC(gamma='scale', kernel='linear', probability=True).fit(training_features,training_labels) # ,class_weight='balanced'
        # model_classifier = LogisticRegression(solver='liblinear',class_weight='balanced', max_iter=1000).fit(training_features,training_labels)
        model_classifier = RandomForestClassifier(n_estimators=50, max_depth=None,class_weight='balanced').fit(training_features,training_labels) #
        ################################################################################################
        # Testing Phase
        test_all_label_numeric = []
        test_all_predicted_labels = []
        test_all_probability_prediction = np.array([])
        for ixs in test_ind_smokers:
            for jxs in [x for x in os.listdir(input_folder_mfcc + 'Y/' + smokers[ixs]) if x.endswith('.npy')][:-1]:
                mfcc = np.load(input_folder_mfcc + 'Y/' + smokers[ixs] + '/' + jxs)
                if mfcc.shape[0]>FRAME_AVG:
                    mfcc_truncated = mfcc[:FRAME_AVG * int(mfcc.shape[0] / FRAME_AVG), :]
                    frames_indx = np.arange(mfcc_truncated.shape[0])
                    cell_len = int(mfcc_truncated.shape[0] / FRAME_AVG)
                    Frm_indx = np.reshape(frames_indx, (FRAME_AVG, cell_len), order='F')
                    MFCCs_matrix = a2v(np.mean(mfcc_truncated[Frm_indx[:, 0], :], axis=0))
                    for i in range(1, cell_len):
                        new_column = a2v(np.nanmean(mfcc_truncated[Frm_indx[:, i], :], axis=0))
                        MFCCs_matrix = np.concatenate((MFCCs_matrix, new_column), axis=1)
                    acoustic_features = MFCCs_matrix.T
                    predicted_labels_frm = model_classifier.predict(acoustic_features)
                    test_all_predicted_labels.append(int(stats.mode(predicted_labels_frm)[0][0]))
                    probability_prediction_frm = model_classifier.predict_proba(acoustic_features)
                    probability_prediction_frm[(probability_prediction_frm > 0.35) & (probability_prediction_frm < 0.65)] = np.nan
                    if len(test_all_probability_prediction) == 0:
                        test_all_probability_prediction = a2v(np.nanmean(probability_prediction_frm,axis=0)).T
                    else:
                        test_all_probability_prediction = np.append(test_all_probability_prediction, a2v(np.nanmean(probability_prediction_frm,axis=0)).T, axis=0)
                    test_all_label_numeric.append(1)


        for ixn in test_ind_nonsmokers:
            for jxn in [x for x in os.listdir(input_folder_mfcc + 'N/' + nonsmokers[ixn]) if x.endswith('.npy')][:-1]:
                mfcc = np.load(input_folder_mfcc + 'N/' + nonsmokers[ixn] + '/' + jxn)
                if mfcc.shape[0]>FRAME_AVG:
                    mfcc_truncated = mfcc[:FRAME_AVG * int(mfcc.shape[0] / FRAME_AVG), :]
                    frames_indx = np.arange(mfcc_truncated.shape[0])
                    cell_len = int(mfcc_truncated.shape[0] / FRAME_AVG)
                    Frm_indx = np.reshape(frames_indx, (FRAME_AVG, cell_len), order='F')
                    MFCCs_matrix = a2v(np.mean(mfcc_truncated[Frm_indx[:, 0], :], axis=0))
                    for i in range(1, cell_len):
                        new_column = a2v(np.nanmean(mfcc_truncated[Frm_indx[:, i], :], axis=0))
                        MFCCs_matrix = np.concatenate((MFCCs_matrix, new_column), axis=1)
                    acoustic_features = MFCCs_matrix.T
                    predicted_labels_frm = model_classifier.predict(acoustic_features)
                    test_all_predicted_labels.append(int(stats.mode(predicted_labels_frm)[0][0]))
                    probability_prediction_frm = model_classifier.predict_proba(acoustic_features)
                    probability_prediction_frm[(probability_prediction_frm > 0.35) & (probability_prediction_frm < 0.65)] = np.nan
                    test_all_probability_prediction = np.append(test_all_probability_prediction,a2v(np.nanmean(probability_prediction_frm,axis=0)).T, axis=0)
                    test_all_label_numeric.append(2)

        test_all_label_numeric = np.array(test_all_label_numeric)


        acc = accuracy_score(test_all_label_numeric, test_all_predicted_labels)
        prec, rec, fsc, _ = precision_recall_fscore_support(test_all_label_numeric, test_all_predicted_labels)
        auc_0 = roc_auc_score(test_all_label_numeric, test_all_probability_prediction[:, 0])
        auc_1 = roc_auc_score(test_all_label_numeric, test_all_probability_prediction[:, 1])


        accuracy = np.append(accuracy,acc)
        precision = np.append(precision,prec[0])
        recall = np.append(recall, rec[0])
        fscore = np.append(fscore, fsc[0])
        auc0 = np.append(auc0, auc_0)
        auc1 = np.append(auc1, auc_1)

        ranked_scores = np.flipud(np.argsort(test_all_probability_prediction[:,0]))
        top5p = ranked_scores[0:int(0.05*len(ranked_scores))]
        top10p = ranked_scores[0:int(0.1 * len(ranked_scores))]
        top15p = ranked_scores[0:int(0.15 * len(ranked_scores))]
        top20p = ranked_scores[0:int(0.2 * len(ranked_scores))]
        top25p = ranked_scores[0:int(0.25 * len(ranked_scores))]
        top30p = ranked_scores[0:int(0.3 * len(ranked_scores))]

        test_labels_top5p  = test_all_label_numeric[top5p]
        test_labels_top10p = test_all_label_numeric[top10p]
        test_labels_top15p = test_all_label_numeric[top15p]
        test_labels_top20p = test_all_label_numeric[top20p]
        test_labels_top25p = test_all_label_numeric[top25p]
        test_labels_top30p = test_all_label_numeric[top30p]

        predicted_labels_top5p  = np.array(test_all_predicted_labels)[top5p]
        predicted_labels_top10p = np.array(test_all_predicted_labels)[top10p]
        predicted_labels_top15p = np.array(test_all_predicted_labels)[top15p]
        predicted_labels_top20p = np.array(test_all_predicted_labels)[top20p]
        predicted_labels_top25p = np.array(test_all_predicted_labels)[top25p]
        predicted_labels_top30p = np.array(test_all_predicted_labels)[top30p]

        acc_top5p = accuracy_score(test_labels_top5p, predicted_labels_top5p)
        acc_top10p = accuracy_score(test_labels_top10p, predicted_labels_top10p)
        acc_top15p = accuracy_score(test_labels_top15p, predicted_labels_top15p)
        acc_top20p = accuracy_score(test_labels_top20p, predicted_labels_top20p)
        acc_top25p = accuracy_score(test_labels_top25p, predicted_labels_top25p)
        acc_top30p = accuracy_score(test_labels_top30p, predicted_labels_top30p)
        print(
            'acc: %2.1f --> top5: %2.1f --> top10: %2.1f --> top15: %2.1f --> top20: %2.1f --> top25: %2.1f --> top30: %2.1f' % (
            100 * acc, 100 * acc_top5p, 100 * acc_top10p, 100 * acc_top15p, 100 * acc_top20p, 100 * acc_top25p,
            100 * acc_top30p))
        accuracy_top5p = np.append(accuracy_top5p, acc_top5p)
        accuracy_top10p = np.append(accuracy_top10p, acc_top10p)
        accuracy_top15p = np.append(accuracy_top15p, acc_top15p)
        accuracy_top20p = np.append(accuracy_top20p, acc_top20p)
        accuracy_top25p = np.append(accuracy_top25p, acc_top25p)
        accuracy_top30p = np.append(accuracy_top30p, acc_top30p)

        # auc_1 = roc_auc_score(test_labels, probability_prediction[:, 1])
    print('Accuracy overal = %2.1f -+ %2.1f' % (100*np.mean(np.array(accuracy)),100*np.std(np.array(accuracy),ddof=1)))
    print('Accuracy top 5  = %2.1f -+ %2.1f' % (100 * np.mean(np.array(accuracy_top5p)), 100 * np.std(np.array(accuracy_top5p), ddof=1)))
    print('Accuracy top 10 = %2.1f -+ %2.1f' % (100 * np.mean(np.array(accuracy_top10p)), 100 * np.std(np.array(accuracy_top10p), ddof=1)))
    print('Accuracy top 15 = %2.1f -+ %2.1f' % (100 * np.mean(np.array(accuracy_top15p)), 100 * np.std(np.array(accuracy_top15p), ddof=1)))
    print('Accuracy top 20 = %2.1f -+ %2.1f' % (100 * np.mean(np.array(accuracy_top20p)), 100 * np.std(np.array(accuracy_top20p), ddof=1)))
    print('Accuracy top 25 = %2.1f -+ %2.1f' % (100 * np.mean(np.array(accuracy_top25p)), 100 * np.std(np.array(accuracy_top25p), ddof=1)))
    print('Accuracy top 30 = %2.1f -+ %2.1f' % (100 * np.mean(np.array(accuracy_top30p)), 100 * np.std(np.array(accuracy_top30p), ddof=1)))

    # df = np.concatenate((a2v(accuracy),a2v(precision),a2v(recall),a2v(fscore),a2v(auc1),a2v(accuracy_top5p),a2v(accuracy_top10p),a2v(accuracy_top15p),a2v(accuracy_top20p)),axis=1)
    # pdf = pd.DataFrame(df, columns=['accuracy', 'precision', 'recall', 'fscore', 'auc', 'accuracy_top5p', 'accuracy_top10p', 'accuracy_top15p', 'accuracy_top20p'])
    # sns.pairplot(pdf)
    # plt.show()





