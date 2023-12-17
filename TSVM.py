import copy
import random
from collections import Counter

import numpy as np
import sklearn.svm as svm

from scipy.signal import savgol_filter
from sklearn.datasets import make_classification
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.tree import DecisionTreeClassifier
import numpy as np


def distEclud(vecA, vecB):
    '''
    输入：向量A和B
    输出：A和B间的欧式距离
    '''
    return np.sqrt(sum(np.power(vecA - vecB, 2)))


def newCent(L):
    '''
    输入：有标签数据集L
    输出：根据L确定初始聚类中心
    '''
    centroids = []
    label_list = np.unique(L[:, -1])
    for i in label_list:
        L_i = L[(L[:, -1]) == i]
        cent_i = np.mean(L_i, 0)
        centroids.append(cent_i[:-1])
    return np.array(centroids)

def best_map(L1,L2):
    # L1 should be the labels and L2 should be the clustering number we got
    from munkres import Munkres,print_matrix
    Label1 = np.unique(L1)  # 去除重复的元素，由小大大排列
    nClass1 = len(Label1)  # 标签的大小
    Label2 = np.unique(L2)
    nClass2 = len(Label2)
    nClass = np.maximum(nClass1, nClass2)
    G = np.zeros((nClass, nClass))
    for i in range(nClass1):
        ind_cla1 = L1 == Label1[i]
        ind_cla1 = ind_cla1.astype(float)
        for j in range(nClass2):
            ind_cla2 = L2 == Label2[j]
            ind_cla2 = ind_cla2.astype(float)
            G[i, j] = np.sum(ind_cla2 * ind_cla1)
    m = Munkres()
    index = m.compute(-G.T)
    index = np.array(index)
    c = index[:, 1]
    newL2 = np.zeros(L2.shape)
    for i in range(nClass2):
        newL2[L2 == Label2[i]] = Label1[c[i]]
    return newL2

def err_rate(gt_s,s):
    print(gt_s)
    print(s)
    c_x = best_map(gt_s, s)
    err_x = np.sum(gt_s[:] != c_x[:])
    missrate = err_x.astype(float) / (gt_s.shape[0])
    return missrate

def semi_kMeans(L, U, distMeas=distEclud, initial_centriod=newCent):
    '''
    输入：有标签数据集L（最后一列为类别标签）、无标签数据集U（无类别标签）
    输出：聚类结果
    '''
    dataSet = np.vstack((L[:, :-1], U))  # 合并L和U
    label_list = np.unique(L[:, -1])
    k = len(label_list)  # L中类别个数
    m = np.shape(dataSet)[0]

    clusterAssment = np.zeros(m)  # 初始化样本的分配
    centroids = initial_centriod(L)  # 确定初始聚类中心
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):  # 将每个样本分配给最近的聚类中心
            minDist = np.inf;
            minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j, :], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI;
                    minIndex = j
            if clusterAssment[i] != minIndex: clusterChanged = True
            clusterAssment[i] = minIndex
    return clusterAssment



if __name__ == '__main__':
    from selftrain import SelfLearningModel
    from sklearn.neighbors import KNeighborsClassifier
    import pandas as pd
    from CARSPLSDA import PLS_DA
    from sklearn.semi_supervised import SelfTrainingClassifier
    from sklearn.metrics import classification_report, f1_score
    from nutrient_class import nutrient_classification
    from similarity_compute import simalirity
    from imblearn.over_sampling import RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours
    from imblearn.over_sampling import SMOTE
    from sklearn.neighbors import KNeighborsClassifier
    import matplotlib.pyplot as plt
    #from heat_map_subplots import lda_learning_plot
    #data1d = np.load()
    df = pd.read_excel(r'F:\SSL for imbalance\1478mean.xlsx')
    # data1d = data1d.reshape(1171 * 5, 224)
    #df = df.drop(df.columns[0], axis=1, inplace=True)
    #x_rtrain, y_rtrain = x_train, y_train
    #(x_rtrain, y_rtrain), (x_vali, y_vali), (x_test,y_test) = ros.fit_resample(x_train, y_train), \
                                                                  #ros.fit_resample(x_vali, y_vali), ros.fit_resample(x_test, y_test)
    Data = np.array(df)
    data = Data[:, 2:].astype('float64')
    y = data[:, -3:].astype('float64')
    data1d = data[:, :-3].astype('float64')
    elements = 'N'
    yclass = nutrient_classification(data=y, elements=elements)
    Beta = 4
    from selftrain import SelfLearningModel
    # tur = np.array(np.where((yclass == 0) | (yclass == 2) | (yclass == 4))).flatten()
    data1d = savgol_filter(data1d, 23, 3, mode='nearest') #23N, tol=0.00016
    labeldata, labels = data1d, yclass #np.vstack((data1d[:700, :], data1d[1100:, :]))
    addrange = list(np.arange(0, 1200)) + list(np.arange(1278, 1478))
    samples_name = Data[:, 1]
    trvali_idx, test_idx = train_test_split(addrange, test_size=0.25, random_state=10) #10 #10
    train_idx, vali_idx = train_test_split(trvali_idx, test_size=0.3, random_state=10) #10
    unlabeldata = simalirity(path1=r'F:\Leaves pixels\1278PIXELS', path2=r'F:\Leaves pixels', names=samples_name, idx=train_idx, beta=Beta, df=df, smooth=[23, 3])
    #np.load(r'F:\Leaves pixels\Train_unlabel_spctrum.npy')
    neglabels = np.full((Beta*len(train_idx),), -1)
    x_train, x_test, y_train, y_test = data1d[train_idx, :], data1d[test_idx, :], yclass[train_idx], yclass[test_idx]
    x_vali, y_vali = data1d[vali_idx, :], yclass[vali_idx]
    y1, y2, y3 = y_train, y_test, y_vali
    label_count1, label_count2, label_count3, set1, set2, set3 = [], [], [], set(y1), set(y2), set(y3)
    for classes in [0,1,2,3,4]:
        label_count1.append(np.shape(np.where(y1 == classes)[0])[0])
        label_count2.append(np.shape(np.where(y2 == classes)[0])[0])
        label_count3.append(np.shape(np.where(y3 == classes)[0])[0])



    rs = 16 #44
    ros = RandomOverSampler(random_state=rs)
    #x_rtrain, y_rtrain = ros.fit_resample(x_train, y_train)
    #x_test, y_test = ros.fit_resample(x_test, y_test)
    x_rtrain, y_rtrain = x_train, y_train
    x_with_unlabel, ywu = np.vstack((x_rtrain, unlabeldata)), np.hstack((y_rtrain, neglabels))

    #######LDA
    ldaf = LinearDiscriminantAnalysis(n_components=4)  #, tol=0.00016
    ldaf.fit(x_rtrain, y_rtrain)
    predd = ldaf.predict(x_test)
    valiaccu = f1_score(y_vali, ldaf.predict(x_vali), average='weighted')
    trainacc = f1_score(y_rtrain, ldaf.predict(x_rtrain), average='weighted')
    weight_ldaacc = f1_score(y_test, predd, average='macro')
    ldasc = accuracy_score(y_test, predd)
    ldacm = confusion_matrix(y_test, predd)
    ldacr = classification_report(y_test, predd)
    ldatrain, ldavali, ldatest = ldaf.transform(x_rtrain), ldaf.transform(x_vali), ldaf.transform(x_test)
    #lda_learning_plot(ldatrain, y_rtrain, ldaf, [], [], iter=8, ss='sm'+'lda' + str(rs), size=15, nomodel=True)

    #######RFC
    rfc = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=2, min_samples_split=4,random_state=16)
    #
    #  RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=6, random_state=16)
    rfc.fit(ldatrain, y_rtrain)
    rfcvali = f1_score(y_vali, rfc.predict(ldavali), average='weighted')
    rfctrain = f1_score(y_rtrain, rfc.predict(ldatrain), average='weighted')
    weight_rfctestacc = f1_score(y_test, rfc.predict(ldatest), average='macro')
    rfctestsc = accuracy_score(y_test, rfc.predict(ldatest))
    rfcCM = confusion_matrix(y_test, rfc.predict(ldatest))
    rfccr = classification_report(y_test, rfc.predict(ldatest))

    #####SSL
    selftr = SelfLearningModel(basemodel=rfc, rs=rs, prob_threshold=0.75)
    selftr.fit_with_resampling(ldaf.transform(x_with_unlabel), ywu, sampling_strategy='mean')
    resamply0 = selftr.predict(ldatest)
    resampacc0 = f1_score(y_test, resamply0, average='macro')
    resamp_training = f1_score(y_rtrain, selftr.predict(ldatrain), average='macro')
    resamplcr0 = classification_report(y_test, resamply0)
    resampcm0 = confusion_matrix(y_test, resamply0)


    #####PLSDA
    elmclf = PLS_DA(n_components=40)
    elmclf.fit(x_rtrain, y_rtrain)
    elmpred = elmclf.predict(x_test)
    elmcm = confusion_matrix(y_test, elmpred)
    elmcr = classification_report(y_test, elmpred)
    elmf1score = f1_score(y_test, elmpred, average='macro')

    ###dnn
    import sklearn.neural_network as nn

    NN = nn.MLPClassifier(hidden_layer_sizes=(2048, 5), random_state=0)
    NN.fit(ldatrain, y_rtrain)
    dnnpredict = NN.predict(ldatest)
    dnncm = confusion_matrix(y_test, dnnpredict)
    dnncr = classification_report(y_test, dnnpredict)
    dnnscore = f1_score(y_test, dnnpredict, average='macro')
    dnnscore1 = f1_score(y_test, dnnpredict, average='weighted')

    # lda_learning_plot(ldatrain, rfc.predict(ldatrain), rfc, iter=0, ss='non'+'rfc'+str(rs), pX=[], pY=[], size=15, nomodel=True)


    #####contrast expriment
    dctree = DecisionTreeClassifier(max_depth=10, min_samples_leaf=6)
    parameters = {'nest': 500, 'lr': 0.0001, 'maxd': 4, 'mcweight': 0.8, 'gamma': 0.1, 'njob': 1, 'spweight': 0.6,
                  'subtario': 0.9}
    from ensemble_predict import Ensemble_framework

    ensenmble_map, ensemble_wap, ensemble_cm, ensenmble_cr, _ = Ensemble_framework(ldatrain, ldatest, y_rtrain, y_test,
                                                                  params=parameters, model='Adaboosting', baseclf=dctree)


    selflda = selftr.clf()
    from semi_feature_selection import Backward_selection
    datawu = np.hstack((x_with_unlabel, ywu.reshape(-1, 1)))
    BS = Backward_selection(basemodel=PLS_DA, xtr=x_train, ytr=y_train, xva=x_vali, yva=y_vali, datawithunlabel=datawu,
                            intervals=5, up_ncomp=25, elements=elements)
    rfcmodel, rfcwave, semiplsdamodel, semiplsda_wave, ldamodel, ldawave, \
    (basetest, semipldatest, semirfctest, semildatest) = BS.semi_LDA_selection(ldamodel=LinearDiscriminantAnalysis, rfcmodel=RandomForestClassifier,
                                                                               n_compenents=4, xtest=x_test, ytest=y_test)
    baseplsmodel, basewave = BS.base_model(), BS.base_opt_wave()
    selftrainmodel = SelfLearningModel(basemodel=ldamodel)
    selftrainmodel.fit_with_resampling(x_with_unlabel[:, ldawave], ywu, sampling_strategy='reverse')
    resamply = selftrainmodel.predict(x_test[:, ldawave])
    resampacc = f1_score(y_test, resamply, average='macro')
    resamplcr = classification_report(y_test, resamply)
    resampcm = confusion_matrix(y_test, resamply)
    selftrainmodelpls = SelfLearningModel(basemodel=baseplsmodel)
    selftrainmodelpls.fit_with_resampling(x_with_unlabel[:, basewave], ywu, sampling_strategy='reverse')
    plsresamply = selftrainmodelpls.predict(x_test[:, basewave])
    plsresampacc = f1_score(y_test, plsresamply, average='macro')
    plsresamplcr = classification_report(y_test, plsresamply)
    plsresampcm = confusion_matrix(y_test, plsresamply)

    # BS.semi_plsda_filter()
    #basecalisc, semicaliscore, semiplsdamodel, semitestscore, basetestsc = BS.semi_plsda_filter(x_test, y_test)

    #rfcbestmodel, rfcwave, semiplsdamodel, plsdaselected_wave, ldabestmodel, ldaselected_wave = BS.semi_LDA_selection(ldamodel=LinearDiscriminantAnalysis,
                                                                                                                      #rfcmodel=RandomForestClassifier, n_compenents=4)
    elmclf = PLS_DA(n_components=30)
    elmclf.fit(x_rtrain, y_rtrain)
    elmpred = elmclf.predict(x_test)
    elmcm = confusion_matrix(y_test, elmpred)
    elmcr = classification_report(y_test, elmpred)
    elmf1score = f1_score(y_test, elmpred, average='macro')

    ######KNN
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(ldatrain, y_rtrain)
    knnpred = knn.predict(ldatest)
    knnscore = f1_score(y_test, knnpred, average='macro')
    knnestsc = accuracy_score(y_test, knnpred)
    knnCM = confusion_matrix(y_test, knnpred)


    baseoptwave = BS.base_opt_wave()
    baseacc, basecm = BS.base_score(x_test, y_test)
    semiplsr, splsrcm = BS.semi_plsda_score(x_test, y_test)
    semirfcacc, semicm = BS.semi_rfc_score(x_test, y_test)
    semildaacc, semildacm = BS.semi_lda_score(x_test, y_test)
    depthnum, semirfcAcc, semiRFC, semilda = semi_RFC_params_tune(model=RandomForestClassifier, xc=x_with_unlabel, yc=ywu, xv=x_vali, yv=y_vali, decomp=ldaf,
                                                                  depth=16, max_estimators=46, intervals=0)
    semirfctest, srfccm = accuracy_score(y_test, semiRFC.predict(semilda.transform(x_test))), confusion_matrix(y_test, semiRFC.predict(semilda.transform(x_test)))
    plsdda = PLS_DA(n_components=20)
    rfcacc = rfcmodel.score(x_test[:, rfcwave], y_test)
    plsdda.fit(x_train, y_train)
    testpred = plsdda.predict(x_vali)
    plsdaacc = accuracy_score(y_vali, testpred)
    selfmodel = SelfLearningModel(plsdda)
    #data = ldaf.transform(data1d)
    yclass[test_idx] = -1
    selfmodel.fit(x_with_unlabel, ywu)
    pldapred = selfmodel.predict(x_vali)
    plsaccuracy = accuracy_score(y_vali, pldapred)
    semidata, l = selfmodel.remain_unlabel_data()
     # priors=[t0, t1, t2]


    # test_LabelSpreading(data, yclass, test_idx)
    # w = LDA(x_train, y_train, k=3)
    x_trainer, x_tester = ldaf.transform(x_train), ldaf.transform(x_test)
    qda.fit(x_trainer, y_train)
    qdapred = qda.predict(x_tester)
    qdacc = accuracy_score(y_test, qdapred)
    dataldf = ldaf.transform(data1d)
    GMM_label_pred, pi, mu, sigma, det_sigma, sigma_inv = ss_GaussianMixtureModels(x_trainer, y_train, x_tester, y_test,
                                                                                   beta=0.4, max_iterations=300,
                                                                                   early_stop=True, tol=1)
    accuracy = accuracy_score(y_test, GMM_label_pred)


    print()