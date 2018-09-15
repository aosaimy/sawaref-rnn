import matplotlib.pyplot as plt
import mplcursors
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd


class SawarefVis(object):

    def __init__(self, y_actual, y_pred, labels):
        # labels_indices = dict((c, i) for i, c in enumerate(labels))
        # indices_labels = dict((i, c) for i, c in enumerate(labels))

        # for i in range(len(y_val)):
        #     y_pred[i] = labels_indices[y_pred[i]]
        #     y_actual[i] = labels_indices[y_actual[i]]

        # print(y_actual, y_pred)
        cm = confusion_matrix(y_actual, y_pred,
                              labels=labels)

        df_cm = pd.DataFrame(cm, labels, labels)
        print(df_cm)

        # for label size
        sn.set(font_scale=0.8)

        # font size
        sn.heatmap(df_cm, fmt="d", annot=True, robust=True,
                   annot_kws={"size": 11},
                   yticklabels=True, xticklabels=True, mask=df_cm == 0)
        mplcursors.cursor(hover=True)
        plt.show()
        plt.savefig('confusion_matrix.png', format='png')
