from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from config import *

def train():
    ###LDA
    ldaf = LinearDiscriminantAnalysis(n_components=4)  # , tol=0.00016
    ldaf.fit(x_rtrain, y_rtrain)
    ldatrain, ldavali, ldatest = ldaf.transform(x_rtrain), ldaf.transform(x_vali), ldaf.transform(x_test)

    ##rfc
    rfc = RandomForestClassifier(n_estimators=opt.n_estimators, max_depth=opt.max_depth,
                                 min_samples_leaf=opt.min_samples_leaf, random_state=opt.rfc_rs)
    rfc.fit(ldatrain, y_rtrain)

    ##PLSDA
    plsda = PLS_DA(n_components=40)
    plsda.fit(x_rtrain, y_rtrain)

    ##Self-training
    selftraining = SelfLearningModel(basemodel=rfc, rs=opt.Resampling_seed, prob_threshold=opt.ssl_prob)
    selftraining.fit_with_resampling(ldaf.transform(x_with_unlabel), ywu, sampling_strategy=opt.Resampling_strategy)

    return ldaf, rfc, plsda, selftraining, ldatrain, ldavali, ldatest


def vali(model, xv, yv):
    vali_prediction = model.predict(xv)
    WAP_vali = f1_score(yv, vali_prediction, average='weighted')
    MAP_vali = f1_score(yv, vali_prediction, average='macro')
    Recall_vali = precision_recall_fscore_support(yv, vali_prediction, average='weighted')
    CM_vali = confusion_matrix(yv, vali_prediction)
    CR_vali = classification_report(yv, vali_prediction)

    return WAP_vali, MAP_vali, Recall_vali, CM_vali, CR_vali


def test(model, xt, yt):

    test_prediction = model.predict(xt)
    WAP_test = f1_score(yt, test_prediction, average='weighted')
    MAP_test = f1_score(yt, test_prediction, average='macro')
    Recall_test = precision_recall_fscore_support(yt, test_prediction, average='weighted')
    CM_test = confusion_matrix(yt, test_prediction)
    CR_test = classification_report(yt, test_prediction)

    return WAP_test, MAP_test, Recall_test, CM_test, CR_test


if __name__ == '__main__':
    ldaf, rfc, plsda, selftr, ldatrain, ldavali, ldatest = train()
    WAP_vali, MAP_vali, Recall_vali, CM_vali, CR_vali = vali(model=ldaf, xv=x_vali, yv=y_vali)
    print(CR_vali)
    WAP_test, MAP_test, Recall_test, CM_test, CR_test = test(model=ldaf, yt=y_test, xt=x_test)
    print(CR_test)

