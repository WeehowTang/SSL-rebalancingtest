from selftrain import SelfLearningModel
import numpy as np
from imblearn.over_sampling import RandomOverSampler, SMOTE

class Reverse_sampling():
    def __init__(self, random_states):
        self.rs = random_states
        self.sample_indices_ = None
    def fit_resample(self, labeldata, labely):
        classes = list(set(labely))
        classes_total = len(labely)
        classes_num = [len(np.where(labely == i)[0]) for i in classes]
        class_ratios = [float(count/classes_total) for count in classes_num]
        tol = np.sum([1 / k for k in classes_num])
        ratios = [float(1 / classes_num[j]) / tol for j in range(5)]
        new_ratios = [float(1/(1+np.log(1+alpha))) for alpha in class_ratios]
        xnew, ynew, rusidx = np.zeros((1, np.shape(labeldata)[1])), np.zeros((1,)), np.zeros((1,))
        for j, i in enumerate(classes):
            class_idx = np.where(labely == i)[0]
            rt = ratios[j]
            idx = np.random.choice(a=class_idx, size=np.minimum(int(rt * len(labely)), len(class_idx)))
            rusidx = np.hstack((rusidx, idx)).astype('int')
            data, label = labeldata[idx], labely[idx]
            xnew, ynew = np.vstack((xnew, data)), np.hstack((ynew, label))
        Xnew, Ynew = xnew[1:, :], ynew[1:]
        self.sample_indices_ = rusidx[1:]
        return Xnew, Ynew


def resampling(labeldata, labely, unlabeldata, unlabely, conf, random_states, model, sampling_strategy=str, plot=False):
    assert sampling_strategy == 'mean' or sampling_strategy == 'reverse' or sampling_strategy == 'smote'
    global rus
    """
    Choose resampling strategy
    """
    if sampling_strategy == 'mean':
        rus = RandomOverSampler(random_state=random_states)
    elif sampling_strategy == 'reverse':
        rus = Reverse_sampling(random_states=random_states)
    elif sampling_strategy == 'smote':
        rus = SMOTE(random_state=random_states)
    x_conf, y_conf = unlabeldata[conf, :], unlabely[conf]
    samples_idx, resample_idx = np.arange(0, len(unlabely)), np.arange(0, len(y_conf))
    unconf_idx = np.delete(samples_idx, conf)
    xunconf = unlabeldata[unconf_idx, :]
    class_num = set(y_conf)

    if len(class_num) <= 4:
        print('It is not balance!')
        ## constrain balance
        return labeldata, labely, unlabeldata, unconf_idx
    else:
        if sampling_strategy == 'smote':
            newlabeldata, newlabely = np.vstack((labeldata, x_conf)), np.hstack((labely, y_conf))
            newlabeldata, newlabely = rus.fit_resample(newlabeldata, newlabely)
            return newlabeldata, newlabely, xunconf, []
        else:
            x_conf_rus, y_conf_rus = rus.fit_resample(x_conf, y_conf)
            rus_idx = rus.sample_indices_
            unresamples_idx = np.delete(resample_idx, rus_idx)
            x_unrus = x_conf[unresamples_idx, :]
            newlabeldata, newlabely = np.vstack((labeldata, x_conf_rus)), np.hstack((labely, y_conf_rus))
            newunlabeldata = np.vstack((xunconf, x_unrus))
            return newlabeldata, newlabely, newunlabeldata, unconf_idx





