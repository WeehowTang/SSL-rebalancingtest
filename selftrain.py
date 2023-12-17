from collections import Counter

import numpy as np
from sklearn.base import BaseEstimator
import sklearn.metrics
import sys
import numpy
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import accuracy_score

class SelfLearningModel(BaseEstimator):
    """
    Self Learning framework for semi-supervised learning

    This class takes a base model (any scikit learn estimator),
    trains it on the labeled examples, and then iteratively
    labeles the unlabeled examples with the trained model and then
    re-trains it using the confidently self-labeled instances
    (those with above-threshold probability) until convergence.

    See e.g. http://pages.cs.wisc.edu/~jerryzhu/pub/sslicml07.pdf

    Parameters
    ----------
    basemodel : BaseEstimator instance
        Base model to be iteratively self trained

    max_iter : int, optional (default=200)
        Maximum number of iterations

    prob_threshold : float, optional (default=0.8)
        Probability threshold for self-labeled instances
    """

    def __init__(self, basemodel, prob_threshold, max_iter=100,  decomp_model=None, rs=0):
        self.model = basemodel
        self.max_iter = max_iter
        self.prob_threshold = prob_threshold
        self.unconfdata = None
        self.newlabeldata = None
        self.decomposition = decomp_model
        self.rs = rs

    from collections import Counter


    # 示例使用

    def fit(self, X, y):
        import pandas as pd
        import numpy as np
        # -1 for unlabeled
        """Fit base model to the data in a semi-supervised fashion
        using self training

        All the input data is provided matrix X (labeled and unlabeled)
        and corresponding label matrix y with a dedicated marker value (-1) for
        unlabeled samples.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            A {n_samples by n_samples} size matrix will be created from this

        y : array_like, shape = [n_samples]
            n_labeled_samples (unlabeled points are marked as -1)

        Returns
        -------
        self : returns an instance of self.
        """
        unlabeledX = X[y == -1, :]
        labeledX = X[y != -1, :]
        labeledy = y[y != -1]
        m, n = np.shape(labeledX)
        if self.decomposition:
            ldalabeledX, ldaunlabeledX = self.decomposition.fit_transform(labeledX, labeledy), self.decomposition.transform(unlabeledX)
            self.model.fit(ldalabeledX, labeledy)
            ldaunlabeledprob = self.model.predict_proba(ldalabeledX)
            ldaunlabeledy = self.model.predict(ldaunlabeledX)
            unlabeledy_old = []
            # re-train, labeling unlabeled instances with model predictions, until convergence
            i = 0
            while (len(unlabeledy_old) == 0 or numpy.any(ldaunlabeledy != unlabeledy_old)) and i < self.max_iter:
                unlabeledy_old = numpy.copy(ldaunlabeledy)
                conf_idx = \
                    numpy.where(
                        (ldaunlabeledprob[:, 0] > self.prob_threshold) | (ldaunlabeledprob[:, 1] > self.prob_threshold) | (
                                    ldaunlabeledprob[:, 2] > self.prob_threshold)
                        | (ldaunlabeledprob[:, 3] > self.prob_threshold) | (ldaunlabeledprob[:, 4] > self.prob_threshold))[0]
                samples_idx = np.arange(0, len(unlabeledy_old))
                unconf_idx = np.delete(samples_idx, conf_idx)
                labeledy = numpy.hstack((labeledy, unlabeledy_old[conf_idx]))
                labeledX = numpy.vstack((labeledX, unlabeledX[conf_idx, :]))
                totallabel = labeledy
                # if self.getdummy:
                # totallabel = pd.get_dummies(labeledy)
                # else:
                # totallabel = labeledy
                ldalabeledX = self.decomposition.fit_transform(labeledX, labeledy)
                self.model.fit(ldalabeledX, totallabel)
                # unlabeledy = self.predict(unlabeledX)
                unlabeledX = unlabeledX[unconf_idx, :]
                if len(unconf_idx) > 0:
                    ldaunlabeledprob = self.predict_proba(self.decomposition.transform(unlabeledX))
                    ldaunlabeledy = self.predict(unlabeledX, decomp=self.decomposition)
                    i += 1
                else:
                    break
            self.unconfdata, self.newlabeldata = unlabeledX, np.hstack((labeledX, labeledy.reshape(-1, 1)))
        else:
            self.model.fit(labeledX, labeledy)
            unlabeledprob = self.model.predict_proba(unlabeledX)
            unlabeledy = self.model.predict(unlabeledX)
            unlabeledy_old = []
            # re-train, labeling unlabeled instances with model predictions, until convergence
            i = 0
            while (len(unlabeledy_old) == 0 or numpy.any(unlabeledy != unlabeledy_old)) and i < self.max_iter:
                unlabeledy_old = numpy.copy(unlabeledy)
                conf_idx = \
                numpy.where((unlabeledprob[:, 0] > self.prob_threshold) | (unlabeledprob[:, 1] > self.prob_threshold)|(unlabeledprob[:, 2] > self.prob_threshold)
                            | (unlabeledprob[:, 3] > self.prob_threshold) | (unlabeledprob[:, 4] > self.prob_threshold))[0]
                labeledy = numpy.hstack((labeledy, unlabeledy_old[conf_idx]))
                labeledX = numpy.vstack((labeledX, unlabeledX[conf_idx, :]))
                totallabel = labeledy
                #if self.getdummy:
                    #totallabel = pd.get_dummies(labeledy)
                #else:
                    #totallabel = labeledy
                self.model.fit(labeledX, totallabel)
                #unlabeledy = self.predict(unlabeledX)
                samples_idx = np.arange(0, len(unlabeledy_old))
                unconf_idx = np.delete(samples_idx, conf_idx)
                unlabeledX = unlabeledX[unconf_idx, :]
                if len(unconf_idx) > 0:
                    unlabeledprob = self.predict_proba(unlabeledX)
                    unlabeledy = self.predict(unlabeledX)
                    i += 1
                else:
                    break
            if not getattr(self.model, "predict_proba", None):
                # Platt scaling if the model cannot generate predictions itself
                self.plattlr = LR()
                preds = self.model.predict(labeledX)
                preds = np.array([np.argmax(i) for i in preds])
                self.plattlr.fit(preds.reshape(-1, 1), labeledy)
            self.unconfdata, self.newlabeldata = unlabeledX, np.hstack((labeledX, labeledy.reshape(-1, 1)))
        return self

    def fit_with_resampling(self, X, y, sampling_strategy, plot=False):
        def count_frequency(data):
            # 使用Counter统计数组中元素的频次
            counter = Counter(data)
            print(counter)


        from imblearn.over_sampling import RandomOverSampler, SMOTE
        from  imblearn.under_sampling import RandomUnderSampler
        from sampling_based_selftraining import resampling
        #from heat_map_subplots import lda_learning_plot
        import matplotlib.pyplot as plt
        unlabeledX = X[y == -1, :]
        labeledX = X[y != -1, :]
        labeledy = y[y != -1]
        ros = SMOTE(random_state=self.rs)
        #labeledX, labeledy = ros.fit_resample(labeledX, labeledy)
        self.model.fit(labeledX, labeledy)
        unlabeledprob = self.model.predict_proba(unlabeledX)
        unlabeledy = self.model.predict(unlabeledX)

        unlabeledy_old = []
        # re-train, labeling unlabeled instances with model predictions, until convergence
        i = 0
        while (len(unlabeledy_old) == 0 or numpy.any(unlabeledy != unlabeledy_old)) and i < self.max_iter:
            label_count = []
            unlabeledy_old = numpy.copy(unlabeledy)
            conf_idx = \
                numpy.where(
                    (unlabeledprob[:, 0] > self.prob_threshold) | (unlabeledprob[:, 1] > self.prob_threshold) | (
                                unlabeledprob[:, 2] > self.prob_threshold)
                    | (unlabeledprob[:, 3] > self.prob_threshold) | (unlabeledprob[:, 4] > self.prob_threshold))[0]
            samples_idx = np.arange(0, len(unlabeledy_old))
            unconf_idx = np.delete(samples_idx, conf_idx)
            #count_frequency(unlabeledy_old[conf_idx])

            labeledX, labeledy = np.vstack((labeledX, unlabeledX[conf_idx, :])), np.append(labeledy, unlabeledy_old[conf_idx])
            unlabeledX = unlabeledX[unconf_idx, :]
            labeledX, labeledy = ros.fit_resample(labeledX, labeledy)

            for classes in [0, 1, 2, 3, 4]:
                label_count.append(np.shape(np.where(labeledy == classes)[0])[0])
            plt.bar(x=[0,1,2,3,4], height=label_count)
            plt.savefig('F:\\plant phenomics\\2nd\\huibao\\' + 'FIG' + str(i) + '.png')
            self.model.fit(labeledX, labeledy)

            if len(conf_idx) > 0:
                unlabeledprob = self.predict_proba(unlabeledX)
                unlabeledy = self.predict(unlabeledX)
                #self.unconfdata = unlabeledX
                i += 1
            else:
                break
            """ 
             unlabeledX = unlabeledX[unconf_idx, :]

            unlabeledprob = self.predict_proba(unlabeledX)
            unlabeledy = self.predict(unlabeledX)
labeledX, labeledy, unlabeledX, unconf_idx = \
                resampling(labeledX, labeledy, unlabeledX, unlabeledy, conf_idx, random_states=self.rs,
                           sampling_strategy=sampling_strategy, iter=i, model=self.model)
            i = i+1
            if len(conf_idx) == 0:
                break
            if plot and i==0:
                pass
            #lda_learning_plot(labeledX, labeledy, self.model, [], [], 'initial', ss=sampling_strategy)
            elif plot:
                pass
            #lda_learning_plot(labeledX, labeledy, self.model, iter=i, pX=[], pY=[], ss=sampling_strategy)
            
              if len(unconf_idx) > 0:
                unlabeledprob = self.predict_proba(unlabeledX)
                unlabeledy = self.predict(unlabeledX)
                self.unconfdata = unlabeledX
                i += 1
            
            """

        return self

    def predict_proba(self, X):
        """Compute probabilities of possible outcomes for samples in X.

        The model need to have probability information computed at training
        time: fit with attribute `probability` set to True.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        T : array-like, shape = [n_samples, n_classes]
            Returns the probability of the sample for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute `classes_`.
        """

        if getattr(self.model, "predict_proba", None):
            return self.model.predict_proba(X)
        else:
            preds = self.model.predict(X)
            return self.plattlr.predict_proba(preds.reshape(-1, 1))

    def predict(self, X, decomp=None):
        """Perform classification on samples in X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        y_pred : array, shape = [n_samples]
            Class labels for samples in X.
        """
        if decomp:
            X = decomp.transform(X)
            return self.model.predict(X)
        else:
            return self.model.predict(X)

    def remain_unlabel_data(self):
        l = np.shape(self.unconfdata)[0]
        unlabely = np.full(l, -1, dtype=int)
        unlabeldata = np.hstack((self.unconfdata, unlabely.reshape(-1, 1)))
        alldata = np.vstack((self.newlabeldata, unlabeldata))
        return alldata, l

    def clf(self):
        if self.decomposition:
            return self.decomposition, self.model
        else:
            return self.model


