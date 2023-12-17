import numpy as np
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import PLSRegression
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVR

class PLS_DA:
    def __init__(self, n_components=25, max_iter=500):
        self.nc = n_components
        self.model = PLSRegression(n_components=self.nc, max_iter=max_iter)
    def fit(self, x, y):
        import pandas as pd
        y = pd.get_dummies(y)
        self.model.fit(x, y)

    def predict_proba(self, x):
        return self.model.predict(x)

    def predict(self, x):
        pred = self.model.predict(x)
        preds = np.array([np.argmax(i) for i in pred])
        return preds

    def coef_(self):
        return self.model.coef_

    def score(self, x, y):
        pred = PLS_DA.predict(self, x)
        return accuracy_score(y, pred)

def spxy(x, y, test_size=0.3):
    """
    :param x: shape (n_samples, n_features)
    :param y: shape (n_sample, )
    :param test_size: the ratio of test_size
    :return: spec_train :(n_samples, n_features)
             spec_test: (n_samples, n_features)
             target_train: (n_sample, )
             target_test: (n_sample, )
    """
    x_backup = x
    y_backup = y
    M = x.shape[0]
    N = round((1 - test_size) * M)
    samples = np.arange(M)

    y = (y - np.mean(y)) / np.std(y)
    D = np.zeros((M, M))
    Dy = np.zeros((M, M))

    for i in range(M - 1):
        xa = x[i, :]
        ya = y[i]
        for j in range((i + 1), M):
            xb = x[j, :]
            yb = y[j]
            D[i, j] = np.linalg.norm(xa - xb)
            Dy[i, j] = np.linalg.norm(ya - yb)

    Dmax = np.max(D)
    Dymax = np.max(Dy)
    D = D / Dmax + Dy / Dymax

    maxD = D.max(axis=0)
    index_row = D.argmax(axis=0)
    index_column = maxD.argmax()

    m = np.zeros(N)
    m[0] = index_row[index_column]
    m[1] = index_column
    m = m.astype(int)

    dminmax = np.zeros(N)
    dminmax[1] = D[m[0], m[1]]

    for i in range(2, N):
        pool = np.delete(samples, m[:i])
        dmin = np.zeros(M - i)
        for j in range(M - i):
            indexa = pool[j]
            d = np.zeros(i)
            for k in range(i):
                indexb = m[k]
                if indexa < indexb:
                    d[k] = D[indexa, indexb]
                else:
                    d[k] = D[indexb, indexa]
            dmin[j] = np.min(d)
        dminmax[i] = np.max(dmin)
        index = np.argmax(dmin)
        m[i] = pool[index]

    m_complement = np.delete(np.arange(x.shape[0]), m)

    spec_train = x[m, :]
    target_train = y_backup[m]
    spec_test = x[m_complement, :]
    target_test = y_backup[m_complement]

    return spec_train, spec_test, target_train, target_test

def CARS_Cloud(xcal, xval, ycal, yval, N=50, f=40, cv=5):
    import copy
    p = 1.0
    m, n = xcal.shape
    u = np.power((n/2), (1/(N-1)))
    k = (1/(N-1)) * np.log(n/2)
    cal_num = np.round(m * p)
    # val_num = m - cal_num
    b2 = np.arange(n)
    x = copy.deepcopy(xcal)
    y = copy.deepcopy(ycal)
    D = np.vstack((np.array(b2).reshape(1, -1), xcal))
    WaveData = []
    # Coeff = []
    WaveNum =[]
    RMSECV = []
    R2 = []
    r = []
    rIndex = []
    cal_index = np.arange(0, cal_num).astype('int64')
    for i in range(1, N+1):
        r.append(u*np.exp(-1*k*i))
        wave_num = int(np.round(r[i-1]*n))
        WaveNum = np.hstack((WaveNum, wave_num))
        wave_index = b2[:wave_num].reshape(1, -1)[0]
        xcal = x[np.ix_(list(cal_index), list(wave_index))]
        #xcal = xcal[:,wave_index].reshape(-1,wave_num)
        ycal = y[cal_index]
        xval = xval[:, wave_index]
        x = x[:, wave_index]
        D = D[:, wave_index]
        d = D[0, :].reshape(1, -1)
        wnum = n - wave_num
        if wnum > 0:
            d = np.hstack((d, np.full((1, wnum), -1)))
        if len(WaveData) == 0:
            WaveData = d
        else:
            WaveData  = np.vstack((WaveData, d.reshape(1, -1)))

        if wave_num < f:
            f = wave_num

        pls = PLSRegression(n_components=f, max_iter=200)
        pls.fit(xcal, ycal)
        beta = pls.coef_
        b = np.abs(beta)
        b2 = np.argsort(-b, axis=0)
        coef = copy.deepcopy(beta)
        coeff = coef[b2, :].reshape(len(b2), -1)
        # cb = coeff[:wave_num]
        #
        # if wnum > 0:
        #     cb = np.vstack((cb, np.full((wnum, 1), -1)))
        # if len(Coeff) == 0:
        #     Coeff = copy.deepcopy(cb)
        # else:
        #     Coeff = np.hstack((Coeff, cb))
        ACCtest, ACCcali, pc = PLSDA(xcal, xval, ycal, yval, component=f)
        RMSECV.append(ACCcali)
        R2.append(ACCtest)
        rIndex.append(pc)

    WAVE = []
    # COEFF = []

    for i in range(WaveData.shape[0]):
        wd = WaveData[i, :]
        # cd = CoeffData[i, :]
        WD = np.ones((len(wd)))
        # CO = np.ones((len(wd)))
        for j in range(len(wd)):
            ind = np.where(wd == j)
            if len(ind[0]) == 0:
                WD[j] = 0
                # CO[j] = 0
            else:
                WD[j] = wd[ind[0]]
                # CO[j] = cd[ind[0]]
        if len(WAVE) == 0:
            WAVE = copy.deepcopy(WD)
        else:
            WAVE = np.vstack((WAVE, WD.reshape(1, -1)))
        # if len(COEFF) == 0:
        #     COEFF = copy.deepcopy(CO)
        # else:
        #     COEFF = np.vstack((WAVE, CO.reshape(1, -1)))
    chooseIndex = np.argsort(RMSECV)
    chooseIndex = chooseIndex[:3]
    MaxR2 = np.argmax(R2)
    MinIndex = np.argmin(RMSECV)
    Optimal = WAVE[MinIndex, :]
    Optimal1 = WAVE[MaxR2, :]
    boindex = np.where(Optimal != 0)
    boindex1 = np.where(Optimal1 != 0)
    optWave = boindex[0]
    R2wave = boindex1[0]
    NeiborIndex = []
    for wave in chooseIndex.tolist():
        Optimal2 = WAVE[wave, :]
        boindex = np.where(Optimal2 != 0)
        OptWave = boindex[0]
        NeiborIndex.append(OptWave)


    fig = plt.figure()
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    fonts = 16
    plt.subplot(211)
    plt.xlabel('蒙特卡洛迭代次数', fontsize=fonts)
    plt.ylabel('被选择的波长数量', fontsize=fonts)
    plt.title('最佳迭代次数：' + str(MinIndex) + '次', fontsize=fonts)
    plt.plot(np.arange(N), WaveNum)

    plt.subplot(223)
    plt.xlabel('蒙特卡洛迭代次数', fontsize=fonts)
    plt.ylabel('R2CV', fontsize=fonts)
    plt.plot(np.arange(N), R2)

    plt.subplot(224)
    plt.xlabel('蒙特卡洛迭代次数', fontsize=fonts)
    plt.ylabel('RMSECV', fontsize=fonts)
    plt.plot(np.arange(N), RMSECV)
    # # plt.subplot(313)
    # # plt.xlabel('蒙特卡洛迭代次数', fontsize=fonts)
    # # plt.ylabel('各变量系数值', fontsize=fonts)
    # # plt.plot(COEFF)
    # #plt.vlines(MinIndex, -1e3, 1e3, colors='r')
    plt.show()

    return optWave, R2wave, rIndex[np.argmax(R2)], np.max(R2)
def PLSDA(xc,xv,yc,yv,component):
    from sklearn.model_selection import KFold
    from sklearn.cross_decomposition import PLSRegression
    import pandas as pd
    k_range = np.linspace(1, component, component)
    x_traincali, x_cali, y_traincali, y_cali = train_test_split(xc, yc, test_size=0.24, random_state=100)
    accuracy_validation = np.zeros((1, component))
    accuracy_train = np.zeros((1, component))
    for j in range(component):
        p = 0
        acc = 0
        model_pls = PLSRegression(n_components=j + 1)
        yc_labels = pd.get_dummies(y_traincali)
        #yv_labels = pd.get_dummies(yv)
        model_pls.fit(x_traincali, yc_labels)
        y_pred = model_pls.predict(x_cali)
        y_pred = np.array([np.argmax(i) for i in y_pred])
        accuracy_train[:, j] = accuracy_score(y_cali, y_pred)
        YC_labels = pd.get_dummies(yv)
        model_1 = PLSRegression(n_components=j + 1)
        Y_pred = model_pls.predict(xv)
        Y_pred = np.array([np.argmax(i1) for i1 in Y_pred])
        acc = accuracy_score(yv, Y_pred) + acc
        p = p + 1
        accuracy_validation[:, j] = acc / p
    #print(accuracy_validation)
    plt.plot(k_range, accuracy_validation.T, 'o-', label="Test", color="r")
    plt.plot(k_range, accuracy_train.T, 'o-', label="Cali", color="b")
    plt.xlabel("N components")
    plt.ylabel("Accuracy")
    plt.legend(loc="best")
    plt.rc('font', family='Times New Roman')
    plt.rcParams['font.size'] = 10
    plt.show()
    return np.max(accuracy_validation.T), np.max(accuracy_train.T), 1+np.argmax(accuracy_validation.T)
def getindex(ytrain, ytest, y):
    y = y.tolist()
    trainer = np.zeros((len(ytrain)))
    tester = np.zeros((len(ytest)))
    for k, value in enumerate(ytrain):
        trainer[k] = y.index(value)
    for j, valuer in enumerate(ytest):
        tester[j] = y.index(valuer)
    return trainer, tester
def genreateindex(traininde, testinde):
    newtrain, newtest = np.array([]), np.array([])
    tol = 109
    for i in range(7):
        a = traininde+i*tol
        b = testinde+i*tol
        newtrain = np.append(newtrain, a)
        newtest = np.append(newtest, b)
    return newtrain, newtest


if __name__ == '__main__' :
    CARS_Cloud()
    PLSDA()