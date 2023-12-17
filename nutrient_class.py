import copy


def nutrient_classification(data, elements=str, value=None):
    if elements == 'N':
        labeln = data[:, 0]
        if value:
            return labeln
        else:
            yclass = copy.deepcopy(labeln)
            for value in range(len(yclass)):
                v = yclass[value,]
                if v <= 2.9:
                    yclass[value,] = 0
                elif 2.9 < v <= 3.2:
                    yclass[value,] = 1
                elif 3.2 < v <= 3.6:
                    yclass[value,] = 2
                elif 3.6 < v <= 3.8:
                    yclass[value,] = 3
                elif v > 3.8:
                    yclass[value,] = 4
            return yclass

    elif elements == 'K':
        labelk = data[:, 1]
        if value:
            return labelk
        else:
            yclass = copy.deepcopy(labelk)
            for value in range(len(yclass)):
                v = yclass[value,]
                if v <= 0.7:
                    yclass[value,] = 0
                elif 0.7 < v <= 0.9:
                    yclass[value,] = 1
                elif 0.9 < v <= 1.1:
                    yclass[value,] = 2
                elif 1.1 < v <= 1.5:
                    yclass[value,] = 3
                elif v > 1.5:
                    yclass[value,] = 4
            return yclass

    else:
        labelmg = data[:, 2]
        yclass = copy.deepcopy(labelmg)
        for value in range(len(yclass)):
            v = yclass[value,]
            if v <= 0.27:
                yclass[value,] = 0
            elif 0.27 < v <= 0.35:
                yclass[value,] = 1
            elif 0.35 < v <= 0.45:
                yclass[value,] = 2
            elif 0.45 < v <= 0.58:
                yclass[value,] = 3
            elif v > 0.58:
                yclass[value,] = 4
        return yclass
