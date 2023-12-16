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


"""
    def output(self, X, y, sample_weight=None):
        import numpy as np
        output = self.predict(X)
        #output = np.array(np.argmax(i) for i in output)
        return output
Created on August 12, 2019
@author: mwdunham
Tested with Python versions: 3.6.6 - 3.7.3

SEMI-SUPERVISED GAUSSIAN MIXTURE MODELS (ssGMM)
	This code is based on the implementation given by:
		@Article{Yan_2017,
		  author   = {Yan, H. and Zhou, J. and Pang, C. K.},
		  title    = {Gaussian mixture model using semisupervised learning for probabilistic fault diagnosis under new data categories},
		  journal  = {IEEE Transactions on Instrumentation and Measurement},
		  year     = {2017},
		  volume   = {66},
		  number   = {4},
		  pages    = {723-733},
		  doi      = {10.1109/TIM.2017.2654552},
		  file     = {:Papers\\Yan et al. (2017) - GMM using semi-supervised learning for probabilistic fault diagnosis.pdf:PDF},
		  keywords = {semi-supervised, ssGMM, machine fault diagnosis}}
"""


# NOTE: A test-train-split of the data needs to be performed before calling this function

# Information regarding INPUTS:
# Xtrain: Training data features (d-dimensional)
# ytrain: Training data labels
# Xtest: Testing data features (d-dimensional)
# ytest: Testing data labels
# beta: tradeoff parameter between unlabeled and labeled data. Must be (0 < beta < 1). beta = 1 is equivalent to 100% supervised, beta = 0 is equivalent to 100% unsupervised
# tol: the tolerance for the ssGMM objective function; it represents the 'percent' change in the objective function. If you wish the algorithm to stop once the obj is only changing by <=1%, then tol=1.0
# max_iterations: maximum number of iterations to perform for optimzing the ssGMM objective function
# early_stop: a boolean variable, i.e. 'True' or 'False'.
## if 'True': at any given iteration, if the ssGMM objective function becomes smaller (worse), the algorithm will stop and will use the information recovered from the PREVIOUS iteration
## if 'False': the code will run until the tol or max_iterations is reached

# Information regarding OUTPUTS (given by return statement at the bottom):
# GMM_label_pred: thresholded predictions for each unlabeled data point derived from the GAMMA matrix
# GAMMA[L:(L+U),:]: probability matrix for each unlabeled data point belonging to each class, size = (len(U), K)
# Objective: array containing the value of the objective function at each iteration, the first entry is the starting value of the objective function prior to using the EM algorithm

def ss_GaussianMixtureModels(Xtrain, ytrain, Xtest, ytest, beta, tol, max_iterations, early_stop=False):
    cond_tolerance = 1E-5  ##the cutoff for singular values - the default for pinv is 1E-15 which was discovered to be too low through testing

    from sklearn.metrics import accuracy_score
    import numpy as np

    ## Custom designed Gaussian Naive Bayes classifier that outputs pi, mu, sigma
    def Bayes(X, y):
        (n, d) = np.shape(X)
        uniq = np.unique(y)
        ####################################################################
        ####          SOLVING FOR THE CLASS PRIOR PROBABILITY          #####
        ####################################################################
        pi = []
        for j in uniq:
            sum = 0
            for i in range(0, n, 1):
                if y[i] == j:
                    sum += 1
            pi.append(sum / n)

            ####################################################################
        ####       SOLVING FOR THE CLASS SPECIFIC GAUSSIAN MEAN         ####
        ####################################################################
        mu_y = np.zeros((len(uniq), d))
        for j in range(0, len(uniq), 1):
            sum = 0
            counter = 0
            for i in range(0, n, 1):
                if y[i] == uniq[j]:
                    sum = sum + X[i, :]
                    counter += 1
            mu_y[j, :] = (1 / counter) * sum

            ####################################################################
        ####     SOLVING FOR THE CLASS SPECIFIC GAUSSAIN COVARIANCE    #####
        ####################################################################
        sigma_dic = {}
        for i in uniq:
            sigma_dic["SIGMA_K_" + str(i)] = np.eye(d)
        # Access an entry in the dictionary using a string key as follows:
        # sigma_ID = "SIGMA_K_" + str(j)
        # sigma_dic[sigma_ID]

        for j in range(0, len(uniq), 1):
            sum = 0
            counter = 0
            sigma_ID = "SIGMA_K_" + str(uniq[j])
            for i in range(0, n, 1):
                if y[i] == uniq[j]:
                    sum = sum + np.outer(np.transpose(X[i, :] - mu_y[j, :]), (X[i, :] - mu_y[j, :]))
                    counter += 1
            sigma_dic[sigma_ID] = (1 / counter) * sum

        return pi, mu_y, sigma_dic

    ###########################################################################
    ###########################################################################

    ##########################
    #### Data preparation ####
    ##########################

    L = np.size(ytrain)  # %%% EQUATION 1 %%%#
    uniq = np.unique(ytrain)
    K = len(uniq)  # The number of classes
    uniq = uniq.tolist()

    U = np.size(ytest)  # %%% EQUATION 2 %%%#
    print('Number of labeled data: ' + str(L))
    print('Number of unlabeled data: ' + str(U))
    D = np.concatenate((Xtrain, Xtest), axis=0)  # %%% EQUATION 4 %%%#
    (n, d) = np.shape(D)

    # ssGMM needs starting values for the Gaussian means & covariances for each class, so a Bayes classifier on the LABELED data is used to determine these
    pi, mu, sigma = Bayes(Xtrain, ytrain)  # %%% EQUATION 8 %%%#

    #### Using a limited number of training data can cause the covariance matrices of the resulting classes
    #### to be singular. The code below uses an SVD decomposition to compute the determinant and the inverse
    #### of the covariance matrices. These are needed to compute probability density function.
    sigma_inv = {}
    det_sigma = []

    for j in range(0, len(uniq), 1):
        sigma_ID = "SIGMA_K_" + str(uniq[j])

        [u, s, v] = np.linalg.svd(sigma[sigma_ID])
        rank = len(s[s > cond_tolerance])
        det_sigma.append(np.prod(s[:rank]))
        try:
            # Code that will (maybe) throw an exception
            sigma_inv[sigma_ID] = np.linalg.inv(sigma[sigma_ID])
            # det_sigma.append(np.linalg.det(sigma[sigma_ID]))
        except np.linalg.LinAlgError:
            print("The covariance matrix associated with Class " + str(uniq[j]) + " is still SINGULAR")
            sigma_inv[sigma_ID] = np.linalg.pinv(sigma[sigma_ID], rcond=cond_tolerance)
        except:
            print("Unexpected error")
            raise

            ###########################################################################

    #########   MULTI-VARIATE GAUSSIAN PROBABILITY DENSITY FUNCTION   #########
    ###########################################################################
    # For the EM algorithm below, the objective function has to be computed MANY times, and the built in function (scipy.stats.multivariate_normal.pdf) recomputes the covariance matrix inverses/determinants, which is computationally inefficient.
    # This function below incorporates the pre-computed covariance matrix inverses and covariance determinants, and this drastically improves the computation time (5-6x faster)

    def gaussian_PDF(x, mu, sigma, det_sigma, sigma_inv):
        return (1 / np.sqrt((2 * np.pi) ** (d) * det_sigma)) * np.exp(
            -0.5 * np.matmul((x - mu).T, np.matmul(sigma_inv, (x - mu))))

    ###########################################################################
    ###################   OBJECTIVE FUNCTION FOR ssGMM   ######################
    ###################            Equation 7            ######################
    ###########################################################################

    def objective_func(L, U, D, ytrain, pi, mu, sigma, det_sigma, sigma_inv):
        sum_label = 0
        ## FOR THE LABELED PART OF THE OBJECTIVE FUNCTION
        for i in range(0, L, 1):
            sigma_ID = "SIGMA_K_" + str(ytrain[i])
            ind = uniq.index(ytrain[i])
            sum_label = sum_label + np.log(
                pi[ind] * gaussian_PDF(D[i, :], mu[ind, :], sigma[sigma_ID], det_sigma[ind], sigma_inv[sigma_ID]))

        ## FOR THE UNLABELED PART OF THE OBJECTIVE FUNCTION
        sum_noLabel = 0
        for i in range(L, L + U, 1):
            inner_sum = 0
            for j in range(0, len(uniq), 1):
                sigma_ID = "SIGMA_K_" + str(uniq[j])
                inner_sum = inner_sum + pi[j] * gaussian_PDF(D[i, :], mu[j, :], sigma[sigma_ID], det_sigma[j],
                                                             sigma_inv[sigma_ID])
            sum_noLabel = sum_noLabel + np.log(inner_sum)

        return beta * sum_label + (1 - beta) * sum_noLabel

    Objective = []
    # This is the starting objective function value
    Objective.append(objective_func(L, U, D, ytrain, pi, mu, sigma, det_sigma, sigma_inv))

    GAMMA = np.zeros((n, K))
    obj_change = tol + 1
    t = 0

    ###########################################################################
    ####### Solving for the soft labels on unalabeled data using EM ###########
    ###########################################################################
    while (obj_change > tol):

        GAMMA_old = np.array(GAMMA)  # Saving the previous GAMMA

        ##########################
        ######## E-STEP ##########
        ##########################
        for i in range(0, n, 1):
            # %%% EQUATION 9 %%%#
            ## For LABELED instances
            if i < L:
                for j in range(0, len(uniq), 1):
                    if ytrain[i] == uniq[j]:
                        GAMMA[i, j] = 1.0

            ## For UNLABELED instances
            else:
                sum = 0
                for j in range(0, len(uniq), 1):
                    sigma_ID = "SIGMA_K_" + str(uniq[j])
                    GAMMA[i, j] = pi[j] * gaussian_PDF(D[i, :], mu[j, :], sigma[sigma_ID], det_sigma[j],
                                                       sigma_inv[sigma_ID])
                    sum = sum + GAMMA[i, j]
                GAMMA[i, :] = (1 / sum) * GAMMA[i, :]

        ##########################
        ######## M-STEP ##########
        ##########################

        for j in range(0, len(uniq), 1):
            # %%% EQUATIONS FROM STEP 3 %%%#
            nl = 0
            nu = 0
            for i in range(0, L, 1):
                nl = nl + GAMMA[i, j]
            for i in range(L, L + U, 1):
                nu = nu + GAMMA[i, j]
            C = (beta * nl + (1 - beta) * nu)  # this is a factor that is common in each of the three parameters below

            #### Updating the cluster prior probabilities, pi ####
            pi[j] = (C) / (beta * L + (1 - beta) * U)

            #### Updating the cluster means, mu ####
            mean_sumL = 0
            mean_sumU = 0
            for i in range(0, L, 1):
                mean_sumL = mean_sumL + GAMMA[i, j] * D[i, :]
            for i in range(L, L + U, 1):
                mean_sumU = mean_sumU + GAMMA[i, j] * D[i, :]
            mu[j, :] = (beta * mean_sumL + (1 - beta) * mean_sumU) / (C)

            #### Updating the cluster covariance matrices, sigma ####
            sigma_ID = "SIGMA_K_" + str(uniq[j])

            sigma_sumL = 0
            sigma_sumU = 0
            for i in range(0, L, 1):
                sigma_sumL = sigma_sumL + GAMMA[i, j] * np.outer(np.transpose(D[i, :] - mu[j, :]), (D[i, :] - mu[j, :]))
            for i in range(L, L + U, 1):
                sigma_sumU = sigma_sumU + GAMMA[i, j] * np.outer(np.transpose(D[i, :] - mu[j, :]), (D[i, :] - mu[j, :]))

            sigma[sigma_ID] = (beta * sigma_sumL + (1 - beta) * sigma_sumU) / (C)

            #### Updating the covariance matrix determinants and covariance inverses ####
            try:  # Code that will (maybe) throw an exception
                sigma_inv[sigma_ID] = np.linalg.inv(sigma[sigma_ID])
                [u, s, v] = np.linalg.svd(sigma[sigma_ID])
                rank = len(s[s > cond_tolerance])
                det_sigma[j] = np.prod(s[:rank])
            except np.linalg.LinAlgError:
                print("The covariance matrix associated with Class " + str(
                    uniq[j]) + " has singular values, so its determinant and inverse has issues")
                sigma_inv[sigma_ID] = np.linalg.pinv(sigma[sigma_ID], rcond=cond_tolerance)
            except:
                print("Unexpected error")
                raise

                ##############################################################
        ######## Compute Objective Function: Log-likelihood ##########
        ##############################################################

        Objective.append(objective_func(L, U, D, ytrain, pi, mu, sigma, det_sigma, sigma_inv))

        ## The early stopping criteria
        if early_stop == 'True':
            if (Objective[t] - Objective[t + 1]) > 0:
                print(
                    'Objective function is INCREASING... stopping early and using the GAMMA from the previous iteration')
                GAMMA = np.array(GAMMA_old)
                break

        obj_change = abs((Objective[t + 1] - Objective[t]) / (Objective[t])) * 100
        t = t + 1

        if t == max_iterations:
            print("Max number of iterations reached")
            break

    print("The number of iterations used: ", t)
    print("The objective function: \n", Objective)

    ## Using a threshold to assign labels to the unlabeled points with the highest probability
    GMM_label_pred = np.ones(U) * 99.99
    k = 0
    for i in range(L, L + U, 1):
        # %%% EQUATION 10 %%%#
        cl = GAMMA[i, :].argmax()
        GMM_label_pred[k] = uniq[cl]
        k = k + 1

    semi_GMM_accuracy = accuracy_score(ytest, GMM_label_pred)
    print("Standard accuracy metric of Semi-supervised GMM using beta = " + str(beta) + ", and tol = " + str(
        tol) + ": " + str(semi_GMM_accuracy))
    miss_class_points = accuracy_score(ytest, GMM_label_pred, normalize=False)
    print("Number of misclassified points: " + str(U - miss_class_points) + "/" + str(U))
    print("")

    return GMM_label_pred, pi, mu, sigma, det_sigma, sigma_inv



def ssGuassian_predict(xtest, ytest, pi, mu, sigma, det_sigma, sigma_inv):
    sum, D = 0, xtest
    m, n = np.shape(D)
    def gaussian_PDF(x, mu, sigma, det_sigma, sigma_inv):
        return (1 / np.sqrt((2 * np.pi) ** (n) * det_sigma)) * np.exp(
            -0.5 * np.matmul((x - mu).T, np.matmul(sigma_inv, (x - mu))))

    uniq = np.unique(ytest)
    K = len(uniq)  # The number of classes
    uniq = uniq.tolist()

    GAMMA = np.zeros((m, K))
    for i in range(m):
        for j in range(0, len(uniq), 1):
            sigma_ID = "SIGMA_K_" + str(uniq[j])
            GAMMA[i, j] = pi[j] * gaussian_PDF(D[i, :], mu[j, :], sigma[sigma_ID], det_sigma[j],
                                               sigma_inv[sigma_ID])
            sum = sum + GAMMA[i, j]
        GAMMA[i, :] = (1 / sum) * GAMMA[i, :]
    GMM_label_pred = np.ones(m) * 99.99
    k = 0
    for i in range(m):
        # %%% EQUATION 10 %%%#
        cl = GAMMA[i, :].argmax()
        GMM_label_pred[k] = uniq[cl]
        k = k + 1

    acc = accuracy_score(ytest, GMM_label_pred)
    return GMM_label_pred, acc