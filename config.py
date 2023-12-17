import argparse
from collections import Counter

import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler, SMOTE
from scipy.signal import savgol_filter
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
from nutrient_class import nutrient_classification
from similarity_compute import simalirity
from sklearn.metrics import precision_recall_fscore_support
from selftrain import SelfLearningModel
from CARSPLSDA import PLS_DA

def parse_option():
    parser = argparse.ArgumentParser('parameter choosing')

    parser.add_argument('--filepath', type=str, default=r'1478mean.xlsx',
                        help='read file path')

    parser.add_argument('--pixel_path', type=str, default=r'1278PIXELS',
                        help='read pixel path')

    parser.add_argument('--element', type=str, default='N',
                        help='could be N and K')

    parser.add_argument('--beta', type=int, default=4,
                        help='unlabel data/label data ratio')

    parser.add_argument('--split_seed', type=int, default=10,
                        help='random state seed for dataset split')

    parser.add_argument('--testsize', type=float, default=0.25,
                        help='test size')

    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')

    parser.add_argument('--SG', type=list, default=[23, 3],
                        help='Window size and step size for SG-smooth')

    parser.add_argument('--Resampling_seed', type=int, default=16,
                        help='random state seed for resampling')

    parser.add_argument('--Resampling_strategy', type=str, default='SMOTE',
                        help='strategy for resampling')

    parser.add_argument('--n_estimators', type=int, default=200,
                        help='estimators')

    parser.add_argument('--max_depth', type=int, default=8,
                        help='max_depth')

    parser.add_argument('--min_samples_leaf', type=int, default=2,
                        help='leaf')

    parser.add_argument('--rfc_rs', type=int, default=16,
                        help='random seed for RFC')

    parser.add_argument('--ssl_prob', type=float, default=0.75,
                        help='select prob threshold for Self-training')

    opt = parser.parse_args()

    return opt

def lda():
    #######LDA
    ldaf = LinearDiscriminantAnalysis(n_components=4)  # , tol=0.00016
    ldaf.fit(x_rtrain, y_rtrain)
    predd = ldaf.predict(x_test)
    valiaccu = f1_score(y_vali, ldaf.predict(x_vali), average='weighted')
    trainacc = f1_score(y_rtrain, ldaf.predict(x_rtrain), average='weighted')
    weight_ldaacc = f1_score(y_test, predd, average='macro')
    ldasc = accuracy_score(y_test, predd)
    ldacm = confusion_matrix(y_test, predd)
    ldacr = classification_report(y_test, predd)
    ldatrain, ldavali, ldatest = ldaf.transform(x_rtrain), ldaf.transform(x_vali), ldaf.transform(x_test)
    return



opt = parse_option()
df = pd.read_excel(opt.filepath)
Dataframe = np.array(df)
data = Dataframe[:, 2:].astype('float64')
Y, X = data[:, -3:].astype('float64'), data[:, :-3].astype('float64')
elements, beta = opt.element, opt.beta
yclass = nutrient_classification(data=Y, elements=elements)
class_count = Counter(yclass)
X = savgol_filter(X, opt.SG[0], opt.SG[1], mode='nearest')
Ranges = list(np.arange(0, 1200)) + list(np.arange(1278, 1478))
samples_name = Dataframe[:, 1]
trvali_idx, test_idx = train_test_split(Ranges, test_size=opt.testsize, random_state=opt.split_seed) #10 #10
train_idx, vali_idx = train_test_split(trvali_idx, test_size=0.3, random_state=opt.split_seed) #10
neglabels = np.full((beta * len(train_idx),), -1)
x_train, x_test, y_train, y_test = X[train_idx, :], X[test_idx, :], yclass[train_idx], yclass[test_idx]
x_vali, y_vali = X[vali_idx, :], yclass[vali_idx]
unlabeldata = simalirity(path1=opt.pixel_path, names=samples_name,
                         idx=train_idx, beta=beta, df=df, smooth=opt.SG)
rs = opt.Resampling_seed  # 44
ros = RandomOverSampler(random_state=rs)
x_rtrain, y_rtrain = ros.fit_resample(x_train, y_train)
x_with_unlabel, ywu = np.vstack((x_rtrain, unlabeldata)), np.hstack((y_rtrain, neglabels))

