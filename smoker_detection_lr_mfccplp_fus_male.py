import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, precision_recall_fscore_support,roc_curve,auc
import random
import matplotlib.pyplot as plt
from acoustic_feature_extraction import array2vector as a2v
from acoustic_feature_extraction import calculate_CI as CI

if __name__ == '__main__':
    input_folder_plp = 'C:/Amir/Codes/smoker detection/features_plp/M/'
    input_folder_mfcc = 'C:/Amir/Codes/smoker detection/features_mfcc/M/'

    smokers = next(os.walk(input_folder_plp + 'Y'))[1]
    nonsmokers = next(os.walk(input_folder_plp + 'N'))[1]
    num_smokers = len(smokers)
    num_nonsmokers = len(nonsmokers)

    folds = 5
    itr = 50
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
    normalacc  = 0
    verylowacc = 0
    mean_fpr  = np.linspace(0, 1, 100)
    mean_pmls = np.linspace(0,50,100) # PMLS: percentage of most likely smokers
    top_x_accuracy = []
    top_x_accuracies = []


    for irx in range(itr):
        print('%d / %d --> ' % (irx + 1, itr),end='')
        test_ind_smokers  = np.sort(random.sample(range(num_smokers), int(num_smokers / folds)))
        train_ind_smokers = np.setdiff1d(np.arange(num_smokers), test_ind_smokers)

        test_ind_nonsmokers  = np.sort(random.sample(range(num_nonsmokers), int(num_smokers / folds)))
        train_ind_nonsmokers = np.setdiff1d(np.arange(num_nonsmokers), test_ind_nonsmokers)[0:len(train_ind_smokers)]

        # collect training features and labels for smokers and training the smoker model
        training_features_smokers_plp  = np.array([])
        training_features_smokers_mfcc = np.array([])
        label_smokers = []
        label_smokers_numeric = []
        for ixs in train_ind_smokers:
            spx = 0  # to choose only one recording for each speaker
            for jxs in [x for x in os.listdir(input_folder_plp + 'Y/' + smokers[ixs]) if x.endswith('.npy')][:-1]:
                # if spx < 5:
                plp_features = np.load(input_folder_plp + 'Y/' + smokers[ixs] + '/' + jxs)
                plp_mean = np.mean(plp_features[:, 0:13], axis=0)
                mfcc_features = np.load(input_folder_mfcc + 'Y/' + smokers[ixs] + '/' + jxs)
                mfcc_mean = np.mean(mfcc_features, axis=0)
                # acoustic_features_plp = a2v(np.concatenate((plp_mean,plp_std),axis=0)).T
                acoustic_features_plp  = a2v(plp_mean).T
                acoustic_features_mfcc = a2v(mfcc_mean).T
                if len(training_features_smokers_plp) == 0:
                    training_features_smokers_plp = acoustic_features_plp
                    training_features_smokers_mfcc = acoustic_features_mfcc
                else:
                    training_features_smokers_plp = np.append(training_features_smokers_plp,acoustic_features_plp,axis=0)
                    training_features_smokers_mfcc = np.append(training_features_smokers_mfcc, acoustic_features_mfcc,axis=0)
                spx += 1
                label_smokers.append('Y')
                label_smokers_numeric.append(1)

        label_smokers_numeric = np.array(label_smokers_numeric)

        # collect training features and labels for non-smokers and training the non-smoker model
        training_features_nonsmokers_plp  = np.array([])
        training_features_nonsmokers_mfcc = np.array([])
        label_nonsmokers = []
        label_nonsmokers_numeric = []
        for ixs in train_ind_nonsmokers:
            spx = 0 # to choose only one recording for each speaker
            for jxs in [x for x in os.listdir(input_folder_plp + 'N/' + nonsmokers[ixs]) if x.endswith('.npy')][:-1]:
                # if spx < 5:
                plp_features = np.load(input_folder_plp + 'N/' + nonsmokers[ixs] + '/' + jxs)
                plp_mean = np.mean(plp_features[:,0:13], axis=0)
                mfcc_features = np.load(input_folder_mfcc + 'N/' + nonsmokers[ixs] + '/' + jxs)
                mfcc_mean = np.mean(mfcc_features, axis=0)
                acoustic_features_plp = a2v(plp_mean).T
                acoustic_features_mfcc = a2v(mfcc_mean).T
                if len(training_features_nonsmokers_plp) == 0:
                    training_features_nonsmokers_plp = acoustic_features_plp
                    training_features_nonsmokers_mfcc = acoustic_features_mfcc
                else:
                    training_features_nonsmokers_plp  = np.append(training_features_nonsmokers_plp, acoustic_features_plp, axis=0)
                    training_features_nonsmokers_mfcc = np.append(training_features_nonsmokers_mfcc,acoustic_features_mfcc, axis=0)
                spx += 1
                label_nonsmokers.append('N')
                label_nonsmokers_numeric.append(2)

        label_nonsmokers_numeric = np.array(label_nonsmokers_numeric)

        training_features_plp = np.concatenate((training_features_smokers_plp, training_features_nonsmokers_plp), axis=0)
        training_features_mfcc = np.concatenate((training_features_smokers_mfcc, training_features_nonsmokers_mfcc),axis=0)
        training_labels = np.concatenate((label_smokers_numeric, label_nonsmokers_numeric), axis=0)
        lr_model_plp  = LogisticRegression(solver='liblinear',class_weight='balanced', max_iter=1000).fit(training_features_plp,training_labels)
        lr_model_mfcc = LogisticRegression(solver='liblinear', class_weight='balanced', max_iter=1000).fit(training_features_mfcc, training_labels)

        ################################################################################################
        # Testing Phase
        all_scores = [] # np.zeros((len(test_ind_smokers)+len(test_ind_nonsmokers),2))
        test_label_smokers_numeric = []
        test_features_smokers_plp  = np.array([])
        test_features_smokers_mfcc = np.array([])
        for ixs in test_ind_smokers:
            for jxs in [x for x in os.listdir(input_folder_plp + 'Y/' + smokers[ixs]) if x.endswith('.npy')][:-1]:
                plp_features = np.load(input_folder_plp + 'Y/' + smokers[ixs] + '/' + jxs)
                plp_mean = np.mean(plp_features[:,0:13], axis=0)
                mfcc_features = np.load(input_folder_mfcc + 'Y/' + smokers[ixs] + '/' + jxs)
                mfcc_mean = np.mean(mfcc_features, axis=0)
                acoustic_features_plp = a2v(plp_mean).T
                acoustic_features_mfcc = a2v(mfcc_mean).T
                if len(test_features_smokers_plp) == 0:
                    test_features_smokers_plp = acoustic_features_plp
                    test_features_smokers_mfcc = acoustic_features_mfcc
                else:
                    test_features_smokers_plp  = np.append(test_features_smokers_plp, acoustic_features_plp, axis=0)
                    test_features_smokers_mfcc = np.append(test_features_smokers_mfcc, acoustic_features_mfcc, axis=0)
                test_label_smokers_numeric.append(1)
        test_label_smokers_numeric = np.array(test_label_smokers_numeric)

        test_label_nonsmokers_numeric = []
        test_features_nonsmokers_plp  = np.array([])
        test_features_nonsmokers_mfcc = np.array([])
        for ixn in test_ind_nonsmokers:
            for jxn in [x for x in os.listdir(input_folder_plp + 'N/' + nonsmokers[ixn]) if x.endswith('.npy')][:-1]:
                plp_features = np.load(input_folder_plp + 'N/' + nonsmokers[ixn] + '/' + jxn)
                plp_mean = np.mean(plp_features[:,0:13], axis=0)
                mfcc_features = np.load(input_folder_mfcc + 'N/' + nonsmokers[ixn] + '/' + jxn)
                mfcc_mean = np.mean(mfcc_features, axis=0)
                acoustic_features_plp = a2v(plp_mean).T
                acoustic_features_mfcc = a2v(mfcc_mean).T
                if len(test_features_nonsmokers_plp) == 0:
                    test_features_nonsmokers_plp  = acoustic_features_plp
                    test_features_nonsmokers_mfcc = acoustic_features_mfcc
                else:
                    test_features_nonsmokers_plp  = np.append(test_features_nonsmokers_plp, acoustic_features_plp, axis=0)
                    test_features_nonsmokers_mfcc = np.append(test_features_nonsmokers_mfcc, acoustic_features_mfcc,axis=0)
                test_label_nonsmokers_numeric.append(2)
        test_label_nonsmokers_numeric = np.array(test_label_nonsmokers_numeric)

        test_features_plp  = np.concatenate((test_features_smokers_plp, test_features_nonsmokers_plp), axis=0)
        test_features_mfcc = np.concatenate((test_features_smokers_mfcc, test_features_nonsmokers_mfcc), axis=0)
        test_labels = np.concatenate((test_label_smokers_numeric, test_label_nonsmokers_numeric), axis=0)

        predicted_labels_plp = lr_model_plp.predict(test_features_plp)
        probability_prediction_plp = lr_model_plp.predict_proba(test_features_plp)
        predicted_labels_mfcc = lr_model_mfcc.predict(test_features_mfcc)
        probability_prediction_mfcc = lr_model_mfcc.predict_proba(test_features_mfcc)

        predicted_labels = np.mean((predicted_labels_plp,predicted_labels_mfcc),axis=0)
        predicted_labels[(1 < predicted_labels) & (predicted_labels < 2)] = 2
        probability_prediction = np.mean((probability_prediction_plp,probability_prediction_mfcc),axis=0)

        ranked_scores = np.flipud(np.argsort(probability_prediction[:, 0]))
        top5p = ranked_scores[0:int(0.05 * len(ranked_scores))]
        top10p = ranked_scores[0:int(0.10 * len(ranked_scores))]
        top15p = ranked_scores[0:int(0.15 * len(ranked_scores))]
        top20p = ranked_scores[0:int(0.20 * len(ranked_scores))]
        top25p = ranked_scores[0:int(0.25 * len(ranked_scores))]
        top30p = ranked_scores[0:int(0.30 * len(ranked_scores))]
        top35p = ranked_scores[0:int(0.35 * len(ranked_scores))]
        top40p = ranked_scores[0:int(0.40 * len(ranked_scores))]
        top45p = ranked_scores[0:int(0.45 * len(ranked_scores))]
        top50p = ranked_scores[0:int(0.50 * len(ranked_scores))]

        test_labels_top5p = test_labels[top5p]
        test_labels_top10p = test_labels[top10p]
        test_labels_top15p = test_labels[top15p]
        test_labels_top20p = test_labels[top20p]
        test_labels_top25p = test_labels[top25p]
        test_labels_top30p = test_labels[top30p]
        test_labels_top35p = test_labels[top35p]
        test_labels_top40p = test_labels[top40p]
        test_labels_top45p = test_labels[top45p]
        test_labels_top50p = test_labels[top50p]

        predicted_labels_top5p = predicted_labels[top5p]
        predicted_labels_top10p = predicted_labels[top10p]
        predicted_labels_top15p = predicted_labels[top15p]
        predicted_labels_top20p = predicted_labels[top20p]
        predicted_labels_top25p = predicted_labels[top25p]
        predicted_labels_top30p = predicted_labels[top30p]
        predicted_labels_top35p = predicted_labels[top35p]
        predicted_labels_top40p = predicted_labels[top40p]
        predicted_labels_top45p = predicted_labels[top45p]
        predicted_labels_top50p = predicted_labels[top50p]

        acc_top5p = accuracy_score(test_labels_top5p, predicted_labels_top5p)
        acc_top10p = accuracy_score(test_labels_top10p, predicted_labels_top10p)
        acc_top15p = accuracy_score(test_labels_top15p, predicted_labels_top15p)
        acc_top20p = accuracy_score(test_labels_top20p, predicted_labels_top20p)
        acc_top25p = accuracy_score(test_labels_top25p, predicted_labels_top25p)
        acc_top30p = accuracy_score(test_labels_top30p, predicted_labels_top30p)
        acc_top35p = accuracy_score(test_labels_top35p, predicted_labels_top35p)
        acc_top40p = accuracy_score(test_labels_top40p, predicted_labels_top40p)
        acc_top45p = accuracy_score(test_labels_top45p, predicted_labels_top45p)
        acc_top50p = accuracy_score(test_labels_top50p, predicted_labels_top50p)

        if acc_top5p > 0:
            normalacc += 1
            fpr, tpr, thresholds = roc_curve(test_labels, probability_prediction[:, 0],pos_label=1)
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            acc = accuracy_score(test_labels, predicted_labels)
            accuracy = np.append(accuracy,acc)
            top_x_acc  = np.array([acc_top5p,acc_top10p,acc_top15p,acc_top20p,acc_top25p,acc_top30p,acc_top35p,acc_top40p,acc_top45p,acc_top50p])
            top_x_accuracy.append(top_x_acc)
            top_x_accuracies.append(np.interp(mean_pmls, np.array([5,10,15,20,25,30,35,40,45,50]),top_x_acc))
            print(
                'acc: %2.1f --> AUC: %2.1f --> top5: %2.1f --> top10: %2.1f --> top15: %2.1f --> top20: %2.1f --> top25: %2.1f --> top30: %2.1f' % (
                100 * acc, roc_auc, 100 * acc_top5p, 100 * acc_top10p, 100 * acc_top15p, 100 * acc_top20p, 100 * acc_top25p,
                100 * acc_top30p))
            accuracy_top5p = np.append(accuracy_top5p, acc_top5p)
            accuracy_top10p = np.append(accuracy_top10p, acc_top10p)
            accuracy_top15p = np.append(accuracy_top15p, acc_top15p)
            accuracy_top20p = np.append(accuracy_top20p, acc_top20p)
            accuracy_top25p = np.append(accuracy_top25p, acc_top25p)
            accuracy_top30p = np.append(accuracy_top30p, acc_top30p)
        else:
            verylowacc += 1
            print('------------------------------------> %d' % verylowacc)

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

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs, ddof=1)
    ci_auc = CI(mean_auc, std_auc, normalacc, 95)[0]

    std_tpr = np.std(tprs, axis=0, ddof=1)
    ci_tprs,tprs_lower,tprs_upper = CI(mean_tpr, std_tpr, normalacc, 95)
    plt.figure(0)
    plt.rcParams.update({'font.size': 24})
    plt.plot(mean_fpr, mean_tpr, color='b',label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f (CI))' % (mean_auc, ci_auc),lw=2,alpha=.8)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,label=r'$\pm$ 95% CI')
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
    plt.xlim([0, 1.01])
    plt.ylim([0, 1.01])
    plt.xlabel('False Positive Rate', fontsize=24)
    plt.ylabel('True Positive Rate', fontsize=24)
    plt.title('Receiver Operating Characteristic', fontsize=26)
    plt.legend(loc="lower right")
    # plt.rcParams.update({'font.size': 24})
    plt.show()
#################################
    mean_top_x_accuracies = np.mean(top_x_accuracies, axis=0)
    std_top_x_accuracies = np.std(top_x_accuracies, axis=0, ddof=1)
    ci_mean_top_x_accuracies, mean_top_x_accuracies_lower, mean_top_x_accuracies_upper = CI(mean_top_x_accuracies, std_top_x_accuracies, normalacc, 95)
    plt.figure(1)
    plt.plot(mean_pmls, 100*mean_top_x_accuracies, color='b', label=r'Mean Accuracy', lw=2, alpha=.8)
    plt.fill_between(mean_pmls, 100*mean_top_x_accuracies_lower, 100*mean_top_x_accuracies_upper, color='grey',  alpha=.2,label=r'$\pm$ 95% CI')
    plt.xlim([0, 50.01])
    plt.ylim([20, 100])
    plt.xlabel('Percentage Most Likely Smokers', fontsize=24)
    plt.ylabel('Percentage Actual Smokers', fontsize=24)
    plt.title('Lift Chart', fontsize=26)
    plt.legend(loc="lower right")
    plt.grid(True)
    # plt.rcParams.update({'font.size': 24})
    plt.show()
    print('done')
