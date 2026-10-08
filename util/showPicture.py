import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import MultipleLocator
import matplotlib.colors as mcolors
from .collective import FRAMES_SIZE, ACTIONS, ACTIVITIES


def multiple_equal(x, y):
    x_len = len(x)
    y_len = len(y)
    times = x_len / y_len
    y_times = [i * times for i in y]
    return y_times

def showLoss(train_loss, test_loss):
    y_train_loss = train_loss
    x_train_loss = range(1, len(y_train_loss) + 1)
    y_test_loss = test_loss
    x_test_loss = range(1, len(y_test_loss) + 1)
    print()
    plt.figure()
    plt.grid(linestyle='--')

    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.xlabel('epochs', size=18)
    plt.ylabel('loss', size=18)

    plt.xlim(0, len(y_train_loss))
    plt.ylim(0, 5)

    x_major_locator = MultipleLocator(1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)

    plt.plot(x_train_loss, y_train_loss, color='red', linewidth=1, linestyle="solid", label="train loss")
    plt.scatter(x_train_loss, y_train_loss, color='red')

    plt.plot(x_test_loss, y_test_loss, color='blue', linewidth=1, linestyle="solid", label="test loss")
    plt.scatter(x_test_loss, y_test_loss, color='blue')
    plt.rcParams.update({'font.size': 14})
    plt.legend()
    plt.title('Loss curve', size=20)
    plt.savefig('/kaggle/working/train_and_test_loss.jpg')
    plt.show()

def showAccuracy(train_accuracy, test_accuracy):
    print(train_accuracy, test_accuracy)
    y_train_acc = train_accuracy
    x_train_acc = range(1, len(y_train_acc) + 1)
    y_test_acc = test_accuracy
    x_test_acc = range(1, len(y_test_acc) + 1)
    plt.figure()

    plt.grid(linestyle='--')

    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.xlim(0, len(y_train_acc))
    plt.ylim(30, 100)

    x_major_locator = MultipleLocator(1)
    y_major_locator = MultipleLocator(5)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)

    plt.xlabel('epochs', size=18)
    plt.ylabel('accuracy', size=18)

    plt.plot(x_train_acc, y_train_acc, color='red', linewidth=1, linestyle="solid", label="train accuracy")
    plt.scatter(x_train_acc, y_train_acc, color='red')

    plt.plot(x_test_acc, y_test_acc, color='blue', linewidth=1, linestyle="solid", label="test accuracy")
    plt.scatter(x_test_acc, y_test_acc, color='blue')
    for i in range(len(x_test_acc)):
        plt.text(x_test_acc[i],y_test_acc[i]+10,"%.2f"%y_test_acc[i]+"%")
    plt.rcParams.update({'font.size': 14})
    plt.legend()
    plt.title('Accuracy curve', size=20)
    plt.savefig('/kaggle/working/train_and_test_accuracy.jpg')
    plt.show()

def showTrainPerAccuracy(train_per_accuracy_total):
    plt.figure()
    plt.grid(linestyle='--')
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlim(0, 20)
    plt.ylim(-1, 101)

    x_major_locator = MultipleLocator(1)
    y_major_locator = MultipleLocator(10)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)

    plt.xlabel('epochs', size=18)
    plt.ylabel('accuracy', size=18)

    colors = {i: a for i, a in enumerate(mcolors.TABLEAU_COLORS.keys())}

    for i, train_per_accuracy in enumerate(train_per_accuracy_total):
        y_train_acc = train_per_accuracy
        x_train_acc = range(1, len(y_train_acc) + 1)

        plt.plot(x_train_acc, y_train_acc, color=colors[i], linewidth=1, linestyle="solid", label=ACTIVITIES[i])
        plt.scatter(x_train_acc, y_train_acc, color=colors[i])
    plt.rcParams.update({'font.size': 14})
    plt.legend()
    plt.title('Accuracy curve for each category', size=20)
    plt.savefig('/kaggle/working/Train_per_accuracy.jpg')
    plt.show()

def showTestPerAccuracy(test_per_accuracy_total):
    plt.figure()
    plt.grid(linestyle='--')
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlim(0, 20)
    plt.ylim(-1, 101)

    x_major_locator = MultipleLocator(1)
    y_major_locator = MultipleLocator(10)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)

    plt.xlabel('epochs', size=18)
    plt.ylabel('accuracy', size=18)

    colors = {i: a for i, a in enumerate(
        mcolors.TABLEAU_COLORS.keys())}

    for i, test_per_accuracy in enumerate(test_per_accuracy_total):
        y_test_acc = test_per_accuracy
        x_test_acc = range(1, len(y_test_acc) + 1)

        plt.plot(x_test_acc, y_test_acc, color=colors[i], linewidth=1, linestyle="solid", label=ACTIVITIES[i])
        plt.scatter(x_test_acc, y_test_acc, color=colors[i])

    plt.rcParams.update({'font.size': 14})
    plt.legend()
    plt.title('Accuracy curve for each category', size=20)
    plt.savefig('Test_per_accuracy.jpg')
    plt.show()
