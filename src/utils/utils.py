import numpy as np
import matplotlib.pyplot as plt
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
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')

#     print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.title(title)
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')