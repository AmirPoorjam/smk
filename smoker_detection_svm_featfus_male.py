import os
import numpy as np
from sklearn import svm
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, precision_recall_fscore_support
import random
import matplotlib.pyplot as plt
from acoustic_feature_extraction import array2vector as a2v
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
    itr = 100
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
                        mfcc_features = np.load(input_folder_mfcc + 'Y/' + smokers[ixs] + '/' + jxs)
                        mfcc_features = np.mean(mfcc_features,axis=0)
                        # acoustic_features = a2v(np.concatenate((mfcc_features,formant_features[1:]),axis=0)).T
                        acoustic_features = a2v(mfcc_features).T
                        if len(training_features_smokers) == 0:
                            training_features_smokers = acoustic_features
                        else:
                            training_features_smokers = np.append(training_features_smokers,acoustic_features,axis=0)
                        spx += 1
                        label_smokers.append('Y')
                        label_smokers_numeric.append(1)

        label_smokers_numeric = np.array(label_smokers_numeric)

        # collect training features and labels for non-smokers and training the non-smoker model
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
                        # acoustic_features = a2v(np.concatenate((mfcc_features,formant_features[1:]),axis=0)).T
                        acoustic_features = a2v(mfcc_features).T
                        if len(training_features_nonsmokers) == 0:
                            training_features_nonsmokers = acoustic_features
                        else:
                            training_features_nonsmokers = np.append(training_features_nonsmokers, acoustic_features, axis=0)
                        spx += 1
                        label_nonsmokers.append('N')
                        label_nonsmokers_numeric.append(2)

        label_nonsmokers_numeric = np.array(label_nonsmokers_numeric)

        training_features = np.concatenate((training_features_smokers, training_features_nonsmokers), axis=0)
        training_labels = np.concatenate((label_smokers_numeric, label_nonsmokers_numeric), axis=0)
        svm_model = svm.SVC(gamma='scale', kernel='rbf', probability=True).fit(training_features,training_labels) # ,class_weight='balanced'

        ################################################################################################
        # Testing Phase
        all_scores = [] # np.zeros((len(test_ind_smokers)+len(test_ind_nonsmokers),2))
        test_label_smokers_numeric = []
        test_features_smokers = np.array([])
        for ixs in test_ind_smokers:
            for jxs in [x for x in os.listdir(input_folder_mfcc + 'Y/' + smokers[ixs]) if x.endswith('.npy')][:-1]:
                pros_feat_file_name = input_folder_form + 'Y/' + jxs
                if os.path.isfile(pros_feat_file_name):
                    formant_features = np.load(input_folder_form + 'Y/' + jxs)
                    if not np.isnan(np.sum(formant_features)):
                        mfcc_features = np.load(input_folder_mfcc + 'Y/' + smokers[ixs] + '/' + jxs)
                        mfcc_features = np.mean(mfcc_features, axis=0)
                        # acoustic_features = a2v(np.concatenate((mfcc_features,formant_features[1:]),axis=0)).T
                        acoustic_features = a2v(mfcc_features).T
                        if len(test_features_smokers) == 0:
                            test_features_smokers = acoustic_features
                        else:
                            test_features_smokers = np.append(test_features_smokers, acoustic_features, axis=0)

                        test_label_smokers_numeric.append(1)
        test_label_smokers_numeric = np.array(test_label_smokers_numeric)

        test_label_nonsmokers_numeric = []
        test_features_nonsmokers = np.array([])
        for ixn in test_ind_nonsmokers:
            for jxn in [x for x in os.listdir(input_folder_mfcc + 'N/' + nonsmokers[ixn]) if x.endswith('.npy')][:-1]:
                pros_feat_file_name = input_folder_form + 'N/' + jxn
                if os.path.isfile(pros_feat_file_name):
                    formant_features = np.load(input_folder_form + 'N/' + jxn)
                    if not np.isnan(np.sum(formant_features)):
                        mfcc_features = np.load(input_folder_mfcc + 'N/' + nonsmokers[ixn] + '/' + jxn)
                        mfcc_features = np.mean(mfcc_features, axis=0)
                        # acoustic_features = a2v(np.concatenate((mfcc_features,formant_features[1:]),axis=0)).T
                        acoustic_features = a2v(mfcc_features).T
                        if len(test_features_nonsmokers) == 0:
                            test_features_nonsmokers = acoustic_features
                        else:
                            test_features_nonsmokers = np.append(test_features_nonsmokers, acoustic_features, axis=0)

                        test_label_nonsmokers_numeric.append(2)
        test_label_nonsmokers_numeric = np.array(test_label_nonsmokers_numeric)

        test_features = np.concatenate((test_features_smokers, test_features_nonsmokers), axis=0)
        test_labels = np.concatenate((test_label_smokers_numeric, test_label_nonsmokers_numeric), axis=0)

        predicted_labels = svm_model.predict(test_features)
        probability_prediction = svm_model.predict_proba(test_features)

        acc = accuracy_score(test_labels, predicted_labels)
        prec, rec, fsc, _ = precision_recall_fscore_support(test_labels, predicted_labels)
        auc_0 = roc_auc_score(test_labels, probability_prediction[:, 0])
        auc_1 = roc_auc_score(test_labels, probability_prediction[:, 1])


        accuracy = np.append(accuracy,acc)
        precision = np.append(precision,prec[0])
        recall = np.append(recall, rec[0])
        fscore = np.append(fscore, fsc[0])
        auc0 = np.append(auc0, auc_0)
        auc1 = np.append(auc1, auc_1)

        ranked_scores = np.flipud(np.argsort(probability_prediction[:,0]))
        top5p = ranked_scores[0:int(0.05*len(ranked_scores))]
        top10p = ranked_scores[0:int(0.1 * len(ranked_scores))]
        top15p = ranked_scores[0:int(0.15 * len(ranked_scores))]
        top20p = ranked_scores[0:int(0.2 * len(ranked_scores))]
        top25p = ranked_scores[0:int(0.25 * len(ranked_scores))]
        top30p = ranked_scores[0:int(0.3 * len(ranked_scores))]

        test_labels_top5p = test_labels[top5p]
        test_labels_top10p = test_labels[top10p]
        test_labels_top15p = test_labels[top15p]
        test_labels_top20p = test_labels[top20p]
        test_labels_top25p = test_labels[top25p]
        test_labels_top30p = test_labels[top30p]

        predicted_labels_top5p = predicted_labels[top5p]
        predicted_labels_top10p = predicted_labels[top10p]
        predicted_labels_top15p = predicted_labels[top15p]
        predicted_labels_top20p = predicted_labels[top20p]
        predicted_labels_top25p = predicted_labels[top25p]
        predicted_labels_top30p = predicted_labels[top30p]

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


    #     toc_scoring = time.time()
    # print('Done! ---> Elapsed Time: %2.2f sec.' % (toc_scoring - tic_scoring))
    # print('AUC_0 = %1.2f ' % auc_0)
    # print('AUC_1 = %1.2f ' % auc_1)
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





