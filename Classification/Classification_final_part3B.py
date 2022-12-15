import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.utils import resample
import seaborn as sns
import numpy as np


# # every function is implementing a different classification method with grid search, providing the best parameters
# # used, evaluation metrics, roc curve and computes also the area under curve. if rub all together it also plots
# # all the roc curves in one plot.


def decision_tree_(x_train, x_test, y_train, y_test):
    clf_dt = tree.DecisionTreeClassifier()

    dt_params = {'criterion': ['gini', 'entropy'],
                 'max_depth': [10, 20, 25, None]}

    gs_dt = GridSearchCV(clf_dt, dt_params, n_jobs=-1, cv=5, scoring='f1_macro')

    gs_dt.fit(x_train, y_train)

    print('\nDecision Tree best parameter:', gs_dt.best_params_)

    y_train_pred_dt = gs_dt.predict(x_train)
    y_test_pred_dt = gs_dt.predict(x_test)

    cmDT_test = metrics.confusion_matrix(y_test, y_test_pred_dt, labels=None)
    cmDT_train = metrics.confusion_matrix(y_train, y_train_pred_dt, labels=None)

    print('\nDecision Tree confusion matrix for train:')
    print(cmDT_train)
    print('\nDecision Tree confusion matrix for test:')
    print(cmDT_test)

    print('\nDecision Tree metrics:')
    print('Macro Precision, recall, f1-score')
    print(metrics.precision_recall_fscore_support(y_test, y_test_pred_dt, average='macro'))

    pred_dt = gs_dt.predict_proba(x_test)
    fpdt, tpdt, threshold_dt = roc_curve(y_test, pred_dt[:, 1])

    plt.figure(1, figsize=(10, 8))
    plt.plot(fpdt, tpdt, lw=2, label='DT')
    plt.plot([0, 1], [0, 1], lw=2, linestyle='--')
    plt.ylabel('True Positive')
    plt.xlabel('False Positive')
    plt.grid(color='black', linestyle='--', linewidth=0.1)

    roc_auc = roc_auc_score(y_test, pred_dt[:, 1])
    print('\nLogistic ROC AUC DT %.3f' % roc_auc)


def random_forest_(x_train, x_test, y_train, y_test):
    clf_rf = RandomForestClassifier()

    rf_params = {'criterion': ['gini', 'entropy'],
                 'n_estimators': [50, 100, 200],
                 'max_depth': [10, 15, 25, None]}

    gs_rf = GridSearchCV(clf_rf, rf_params, n_jobs=-1, cv=5, scoring='f1_macro')

    gs_rf.fit(x_train, y_train)

    print('\nRandom Forest best parameter:', gs_rf.best_params_)

    y_train_pred_rf = gs_rf.predict(x_train)
    y_test_pred_rf = gs_rf.predict(x_test)

    cmRF_test = metrics.confusion_matrix(y_test, y_test_pred_rf, labels=None)
    cmRF_train = metrics.confusion_matrix(y_train, y_train_pred_rf, labels=None)

    print('\nRandom forest confusion matrix for train:')
    print(cmRF_train)
    print('\nRandom forest confusion matrix for test:')
    print(cmRF_test)

    print('\nRandom Forest metrics:')
    print('Macro Precision, recall, f1-score')
    print(metrics.precision_recall_fscore_support(y_test, y_test_pred_rf, average='macro'))

    pred_rf = gs_rf.predict_proba(x_test)
    fprf, tprf, threshold_rf = roc_curve(y_test, pred_rf[:, 1])
    plt.plot(fprf, tprf, lw=2, label='RF')

    roc_auc = roc_auc_score(y_test, pred_rf[:, 1])
    print('\nLogistic ROC AUC RF %.3f' % roc_auc)


def naive_bayes_(x_train, x_test, y_train, y_test):
    clf_nb = GaussianNB()

    clf_nb.fit(x_train, y_train)

    y_train_pred_nb = clf_nb.predict(x_train)
    y_test_pred_nb = clf_nb.predict(x_test)

    cmNB_test = metrics.confusion_matrix(y_test, y_test_pred_nb, labels=None)
    cmNB_train = metrics.confusion_matrix(y_train, y_train_pred_nb, labels=None)

    print('\nNaive Bayes confusion matrix for train:')
    print(cmNB_train)
    print('\nNaive Bayes confusion matrix for test:')
    print(cmNB_test)

    print('\nNaive Bayes metrics:')
    print('Macro Precision, recall, f1-score')
    print(metrics.precision_recall_fscore_support(y_test, y_test_pred_nb, average='macro'))

    pred_nb = clf_nb.predict_proba(x_test)
    fpnb, tpnb, threshold_nb = roc_curve(y_test, pred_nb[:, 1])
    plt.plot(fpnb, tpnb, lw=2, label='NB')

    roc_auc = roc_auc_score(y_test, pred_nb[:, 1])
    print('\nLogistic ROC AUC NB %.3f' % roc_auc)


# # raise regularization parameter (C) for better results
def support_vector_machine_(x_train, x_test, y_train, y_test):
    clf_svm = svm.SVC()

    svm_params = {'kernel': ['rbf'],
                  'C': [2000, 5000],
                  'probability': [True]
                  }

    gs_svm = GridSearchCV(clf_svm, svm_params, n_jobs=-1, cv=5, scoring='f1_macro')

    gs_svm.fit(x_train, y_train)

    print('\nSupport Vector Machine best parameter:', gs_svm.best_params_)

    y_train_pred_svm = gs_svm.predict(x_train)
    y_test_pred_svm = gs_svm.predict(x_test)

    cmSVM_test = metrics.confusion_matrix(y_test, y_test_pred_svm, labels=None)
    cmSVM_train = metrics.confusion_matrix(y_train, y_train_pred_svm, labels=None)

    print('\nSupport Vector Machine confusion matrix for train:')
    print(cmSVM_train)
    print('\nSupport Vector Machine confusion matrix for test:')
    print(cmSVM_test)

    print('\nSupport Vector Machine metrics:')
    print('Macro Precision, recall, f1-score')
    print(metrics.precision_recall_fscore_support(y_test, y_test_pred_svm, average='macro'))

    pred_svm = gs_svm.predict_proba(x_test)
    fpsvm, tpsvm, threshold_svm = roc_curve(y_test, pred_svm[:, 1])
    plt.plot(fpsvm, tpsvm, lw=2, label='SVM')

    roc_auc = roc_auc_score(y_test, pred_svm[:, 1])
    print('\nLogistic ROC AUC SVM %.3f' % roc_auc)


def support_vector_machine_linear_(x_train, x_test, y_train, y_test):
    clf_svm = svm.SVC()

    svm_params = {'kernel': ['linear'],
                  'C': [1, 10, 20],
                  'probability': [True]
                  }

    gs_svm = GridSearchCV(clf_svm, svm_params, n_jobs=-1, cv=5, scoring='f1_macro')

    gs_svm.fit(x_train, y_train)

    print('\nSupport Vector Machine best parameter:', gs_svm.best_params_)

    y_train_pred_svm = gs_svm.predict(x_train)
    y_test_pred_svm = gs_svm.predict(x_test)

    cmSVM_test = metrics.confusion_matrix(y_test, y_test_pred_svm, labels=None)
    cmSVM_train = metrics.confusion_matrix(y_train, y_train_pred_svm, labels=None)

    print('\nSupport Vector Machine confusion matrix for train:')
    print(cmSVM_train)
    print('\nSupport Vector Machine confusion matrix for test:')
    print(cmSVM_test)

    print('\nSupport Vector Machine metrics:')
    print('Macro Precision, recall, f1-score')
    print(metrics.precision_recall_fscore_support(y_test, y_test_pred_svm, average='macro'))

    pred_svm = gs_svm.predict_proba(x_test)
    fpsvm, tpsvm, threshold_svm = roc_curve(y_test, pred_svm[:, 1])
    plt.plot(fpsvm, tpsvm, lw=2, label='SVM_linear')

    roc_auc = roc_auc_score(y_test, pred_svm[:, 1])
    print('\nLogistic ROC AUC SVM %.3f' % roc_auc)


def neural_networks_(x_train, x_test, y_train, y_test):
    clf_ann = MLPClassifier()

    ann_params = {
        'hidden_layer_sizes': [(5, 10, 5)],
        'activation': ['tanh', 'relu'],
        'solver': ['adam', 'lbfgs'],
        'tol': [1e-7],
        'alpha': [0.0001],
        'learning_rate': ['constant'],
        'batch_size': [100, 200]
    }

    gs_ann = GridSearchCV(clf_ann, ann_params, n_jobs=-1, cv=5, scoring='f1_macro')

    gs_ann.fit(x_train, y_train)

    print('\nArtificial Neural Network best parameter:', gs_ann.best_params_)

    y_train_pred_ann = gs_ann.predict(x_train)
    y_test_pred_ann = gs_ann.predict(x_test)

    cmANN_test = metrics.confusion_matrix(y_test, y_test_pred_ann, labels=None)
    cmANN_train = metrics.confusion_matrix(y_train, y_train_pred_ann, labels=None)

    print('\nArtificial Neural Network confusion matrix for train:')
    print(cmANN_train)
    print('\nArtificial Neural Network confusion matrix for train:')
    print(cmANN_test)

    print('\nArtificial Neural Network metrics:')
    print('Macro Precision, recall, f1-score')
    print(metrics.precision_recall_fscore_support(y_test, y_test_pred_ann, average='macro'))

    pred_ann = gs_ann.predict_proba(x_test)
    fpann, tpann, threshold_ann = roc_curve(y_test, pred_ann[:, 1])

    roc_auc = roc_auc_score(y_test, pred_ann[:, 1])
    print('\nLogistic ROC AUC ANN %.3f' % roc_auc)

    plt.plot(fpann, tpann, lw=2, label='ANN')
    plt.legend(loc="lower right")
    plt.savefig('Roc_Curve.png')
    plt.show()


# #----------------------------------------------------START-----------------------------------------------# #
data1 = pd.read_csv('Classification_Project_mean.csv')

# Correlation Matrix of the attributes that produced on the previous py file
plt.figure(figsize=(10, 7))
c = data1.corr()
mask = np.triu(np.ones_like(c, dtype=bool))
sns.heatmap(c, mask=mask, annot=True, cmap='coolwarm', linecolor='white', linewidths=0.1)
plt.show()

# visualize new features in a pairplot
data2 = data1.sample(frac=0.1, random_state=40)
pairs = sns.pairplot(data2, hue='action_type')
pairs.fig.suptitle('Features')
plt.show()

# random sample
data_random = data1.sample(frac=0.01, random_state=40)

# over sample 0 class. 0 class to be half of the 1 class
# increasing more the 0 class gives better results but decrease the speed of the running
data_1 = data1[data1['action_type'] == 1]
data_0 = data1[data1['action_type'] == 0]
data_0 = resample(data_0, replace=True, n_samples=round(len(data_1) / 2), random_state=1)
data_over = pd.concat([data_0, data_1])

# under sample 1 class. This is 1% of the dataset and was used like this to see also SVM
data_2 = data1[data1['action_type'] == 0].sample(n=5000)
data_3 = data1[data1['action_type'] == 1].sample(n=5000)
data_under = pd.concat([data_2, data_3])

# # Change to 1 the model you want to run. The rest have to be 0. If you run as it is it will run the best model
whole_dataset = 0
sample_dataset = 0
over_sampled_dataset = 1
under_sampled_dataset = 0

# # every if statement run for a different dataframe. It preprocess the data and runs the functions with test size 0.3


if whole_dataset == 1:
    data = data1
    X = data.drop(['action_type'], axis=1).values
    Y = data.action_type.values

    X = preprocessing.scale(X)

    test_size = 0.3
    random_state = 1

    x_train_, x_test_, y_train_, y_test_ = train_test_split(X, Y, test_size=test_size, random_state=random_state)

    decision_tree_(x_train_, x_test_, y_train_, y_test_)
    print('\nDT finished\n')
    random_forest_(x_train_, x_test_, y_train_, y_test_)
    print('\nRF finished\n')
    naive_bayes_(x_train_, x_test_, y_train_, y_test_)
    print('\nNB finished\n')
    neural_networks_(x_train_, x_test_, y_train_, y_test_)
    print('\nANN finished\n')

if sample_dataset == 1:
    data = data_random
    X = data.drop(['action_type'], axis=1).values
    Y = data.action_type.values

    X = preprocessing.scale(X)

    test_size = 0.3
    random_state = 1

    x_train_, x_test_, y_train_, y_test_ = train_test_split(X, Y, test_size=test_size, random_state=random_state)

    decision_tree_(x_train_, x_test_, y_train_, y_test_)
    print('\nDT finished\n')
    random_forest_(x_train_, x_test_, y_train_, y_test_)
    print('\nRF finished\n')
    naive_bayes_(x_train_, x_test_, y_train_, y_test_)
    print('\nNB finished\n')
    support_vector_machine_(x_train_, x_test_, y_train_, y_test_)
    print('\nSVM finished\n')
    support_vector_machine_linear_(x_train_, x_test_, y_train_, y_test_)
    print('\nSVM_linear finished\n')
    neural_networks_(x_train_, x_test_, y_train_, y_test_)
    print('\nANN finished\n')

if over_sampled_dataset == 1:
    data = data_over
    X = data.drop(['action_type'], axis=1).values
    Y = data.action_type.values

    X = preprocessing.scale(X)

    test_size = 0.3
    random_state = 1

    x_train_, x_test_, y_train_, y_test_ = train_test_split(X, Y, test_size=test_size, random_state=random_state)

    decision_tree_(x_train_, x_test_, y_train_, y_test_)
    print('\nDT finished\n')
    random_forest_(x_train_, x_test_, y_train_, y_test_)
    print('\nRF finished\n')
    naive_bayes_(x_train_, x_test_, y_train_, y_test_)
    print('\nNB finished\n')
    neural_networks_(x_train_, x_test_, y_train_, y_test_)
    print('\nANN finished\n')

if under_sampled_dataset == 1:
    data = data_under
    X = data.drop(['action_type'], axis=1).values
    Y = data.action_type.values

    X = preprocessing.scale(X)

    test_size = 0.3
    random_state = 1

    x_train_, x_test_, y_train_, y_test_ = train_test_split(X, Y, test_size=test_size, random_state=random_state)

    decision_tree_(x_train_, x_test_, y_train_, y_test_)
    print('\nDT finished\n')
    random_forest_(x_train_, x_test_, y_train_, y_test_)
    print('\nRF finished\n')
    naive_bayes_(x_train_, x_test_, y_train_, y_test_)
    print('\nNB finished\n')
    support_vector_machine_(x_train_, x_test_, y_train_, y_test_)
    print('\nSVM finished\n')
    support_vector_machine_linear_(x_train_, x_test_, y_train_, y_test_)
    print('\nSVM_linear finished\n')
    neural_networks_(x_train_, x_test_, y_train_, y_test_)
    print('\nANN finished\n')
