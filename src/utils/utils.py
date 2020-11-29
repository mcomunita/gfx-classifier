import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import itertools

def get_num_correct_labels(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()


def get_num_correct_settings(preds, settings):
    error = preds - settings
    counter = 0
    for i, row in enumerate(error):
        if all(j < 0.1 for j in abs(row)):
            counter += 1
    return counter


def get_num_correct(preds_fx, preds_set, labels, settings):
    correct_fx = preds_fx.argmax(dim=1).eq(labels)
    error = preds_set - settings
    counter = 0
    for i, row in enumerate(error):
        if correct_fx[i] and  all(j < 0.1 for j in abs(row)):
            counter += 1
    return counter
    


# from https://deeplizard.com/learn/video/0LhiS6yu2qQ
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', title_fontsize=20, text_fontsize=16, cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # print("Normalized confusion matrix")
    # else:
    #     print('Confusion matrix')

    # print(cm)
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=title_fontsize)
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=text_fontsize)
    plt.yticks(tick_marks, classes, fontsize=text_fontsize)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), size=text_fontsize, horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label', fontsize=text_fontsize)
    plt.xlabel('Predicted label', fontsize=text_fontsize)
    plt.tight_layout()

def box_plot(error_dataframe, title, title_fontsize=20, text_fontsize=16):
    ax = sns.boxplot(x="variable", y="value", data=pd.melt(error_dataframe), width=0.5, showfliers = False, fliersize=2)

    plt.title(title, fontsize=title_fontsize)

    plt.xlabel('Control', fontsize=text_fontsize)
    plt.ylabel('Error', fontsize=text_fontsize)

    plt.xticks(fontsize=text_fontsize)
    plt.yticks(fontsize=text_fontsize)
    # plt.rc('font', size=text_fontsize)
    plt.tight_layout()
    
    return ax

def scatter_plot(x, y, dataframe, title, xlabel, ylabel, title_fontsize=20, text_fontsize=16, bins=20):
    ax = sns.jointplot(
        x=x, 
        y=y, 
        data=dataframe, 
        marker='.', x_jitter=0.04, kind="reg", space=0.1,
        scatter_kws={"s": 1}, 
        line_kws={"color":"r","alpha":0.7,"lw":2},
        marginal_kws=dict(kde=False, bins=bins)
    )

    ax.fig.suptitle(title, fontsize=title_fontsize)

    plt.xlabel(xlabel, fontsize=text_fontsize)
    plt.ylabel(ylabel, fontsize=text_fontsize)

    plt.xticks(fontsize=text_fontsize)
    plt.yticks(fontsize=text_fontsize)

    ax.fig.tight_layout()

    return ax

def line_plot(x_gain, x_tone, y_gain, y_tone, xlabel, ylabel, title, title_fontsize=20, text_fontsize=16):
    plt.rc('grid', linestyle="--", linewidth=0.5)

    plt.plot(x_gain, y_gain, linestyle='-', linewidth=1, marker='o')
    plt.plot(x_tone, y_tone, linestyle='-', linewidth=1, marker='x')

    plt.title(title, fontsize=title_fontsize)

    plt.xlabel(xlabel, fontsize=text_fontsize)
    plt.ylabel(ylabel, fontsize=text_fontsize)

    plt.xticks(fontsize=text_fontsize)
    plt.yticks(fontsize=text_fontsize)
    plt.xticks(np.linspace(start=0, stop=1, num=6))

    plt.rc('font', size=text_fontsize) 
    
    plt.grid(True)
    
    plt.legend(['Gain', 'Tone'])
    plt.tight_layout()

    