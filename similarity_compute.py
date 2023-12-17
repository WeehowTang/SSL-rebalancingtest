import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from scipy.signal import savgol_filter

def simalirity(df, path1, names, idx, beta, smooth:list):
    def get_cos_similar_multi(v1, v2):
        num = np.dot(v1, v2.T)  # 向量点乘
        denom = np.linalg.norm(v1) * np.linalg.norm(v2, axis=1)  # 求模长的乘积
        res = num / denom
        res[np.isneginf(res)] = 0
        return 0.5 + 0.5 * res
    pathlist = os.listdir(path1)
    pixelslist = [os.path.join(path1, i) for i in pathlist]
    want_names = [key + '.npy' for key in names[idx]]
    Arrdata = np.ones((1, 224))
    for shortname, path_name in zip(pathlist, pixelslist):
        if shortname in want_names:
            mask = df.loc[df['names'] == shortname[:-4]]
            meandata = np.array(mask[np.arange(0, 224)])
            pixeldata = np.load(path_name)
            similarity = get_cos_similar_multi(meandata, pixeldata)
            betaarg = np.argsort(similarity.flatten())[-beta:]
            choose_pixdata = pixeldata[betaarg]
            choose_pixdata = savgol_filter(choose_pixdata, smooth[0], smooth[1], mode='nearest')
            Arrdata = np.vstack((Arrdata, choose_pixdata))
        else:
            pass
    Traindata = Arrdata[1:, :]
    return Traindata