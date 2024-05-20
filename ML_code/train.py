import numpy as np
import multiprocessing
from warnings import warn
from pickle import dump
from pandas import DataFrame, concat, read_parquet
from matplotlib import pyplot as plt
from sklearn.model_selection import RepeatedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import NuSVC
from sklearn.neural_network import MLPClassifier
from time import time
from os.path import dirname, basename
from process_chunk import process_chunk


np.random.seed(123)
n_fingerprints = 1024


def write_to_file(text):
    with open('Data\\Test\\output.txt', 'a+') as file:
        file.write(text)


def print_metrics(metrics, confidence_intervals, confidence_level, name, cv_time):
    border = '*' * 80
    text = '\n'.join([
        'Evaluation metrics - {}:'.format(name),
        ' - Accuracy: {} with CI({}) = ({}, {})'.
        format(metrics.ACC, confidence_level, confidence_intervals.ACC[0], confidence_intervals.ACC[1]),
        ' - True positive rate/Sensitivity: {} with CI({}) = ({}, {})'.
        format(metrics.TPR, confidence_level, confidence_intervals.TPR[0], confidence_intervals.TPR[1]),
        ' - True negative rate/Specificity: {} with CI({}) = ({}, {})'.
        format(metrics.TNR, confidence_level, confidence_intervals.TNR[0], confidence_intervals.TNR[1]),
        ' - Positive predictive value/Precision: {} with CI({}) = ({}, {})'.
        format(metrics.PPV, confidence_level, confidence_intervals.PPV[0], confidence_intervals.PPV[1]),
        ' - F1 score: {} with CI({}) = ({}, {})'.
        format(metrics.F1, confidence_level, confidence_intervals.F1[0], confidence_intervals.F1[1]),
        ' - Matthews correlation coefficient: {} with CI({}) = ({}, {})'.
        format(metrics.MCC, confidence_level, confidence_intervals.MCC[0], confidence_intervals.MCC[1]),
        ' - Execution time: {} second'.format(cv_time)
    ])
    print(f'\n{border}\n {text} \n{border}\n')
    write_to_file(f'\n{border}\n {text} \n{border}\n')


def box_plot(cv_metrics, name):
    (cv_metrics * 100).boxplot(column=['ACC', 'TPR', 'TNR', 'PPV', 'F1', 'MCC'], figsize=(10, 7))
    plt.title('Boxplot - {}'.format(name))
    plt.tight_layout()
    save_data = input('Do you want to save the boxplot of the metrics? (Y/n): ').lower()
    if save_data == 'y':
        name_fig = input(' + Enter the name of the boxplot: ')
        plt.savefig('Plot\\{}_boxplot.png'.format(name_fig))
        path = dirname('Plot\\{}_boxplot.png'.format(name_fig))
        print('\n   * {}_boxplot.png is saved in folder {} \n'.format(name_fig, path))
    plt.show()


def metrics_plot(cv_metrics):
    fig, axs = plt.subplots(2, 3, figsize=(10, 7))
    x = range(len(cv_metrics))
    for i, metric in enumerate(cv_metrics.columns):
        row, col = i // 3, i % 3
        axs[row, col].plot(x, getattr(cv_metrics, metric))
        axs[row, col].set_title(metric)
    plt.tight_layout()
    save_data = input('Do you want to save the plot of the metrics? (Y/n): ').lower()
    if save_data == 'y':
        name_fig = input(' + Enter name of plot: ')
        plt.savefig('Plot\\{}_metrics.png'.format(name_fig))
        path = dirname('Plot\\{}_metrics.png'.format(name_fig))
        print('\n   * {}_metrics.png is saved in folder {} \n'.format(name_fig, path))
    plt.show()


def bootstrapping(boot_iterations, confidence_level, cv_metrics):
    lower_bound, upper_bound = [], []
    for col in cv_metrics.columns:
        metric = cv_metrics[col]
        bootstrap_means = list(map(lambda _: np.mean(np.random.choice(metric, size=len(metric), replace=True)),
                                   range(boot_iterations)))
        lower = np.percentile(bootstrap_means, 100 * (1 - confidence_level) / 2)
        lower_bound.append(np.round(100 * lower, 2))
        upper = np.percentile(bootstrap_means, 100 * (1 - (1 - confidence_level) / 2))
        upper_bound.append(np.round(100 * upper, 2))
    return DataFrame([lower_bound, upper_bound], columns=['ACC', 'TPR', 'TNR', 'PPV', 'F1', 'MCC'])


def cross_validation(xdata, ydata, model, cross_validation_, probability):
    chunks = [(train_index, test_index) for train_index, test_index in cross_validation_.split(xdata)]
    processes = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=processes) as pool:
        arg = [(chunk, xdata, ydata, model, probability) for chunk in chunks]
        cv_metrics = pool.starmap(process_chunk, arg)
    pool.close()
    pool.join()
    return DataFrame(cv_metrics, columns=['ACC', 'TPR', 'TNR', 'PPV', 'F1', 'MCC'])


def repeated_kfold_cv(xdata, ydata, models, cross_validation_, confidence_level, boot_iterations, probability):
    model, name = models[0][0], models[0][1]
    calibration = input('Do you want to calibrate the model? (Y/n): ').lower() == 'y'
    model = (CalibratedClassifierCV(model) if calibration else model)
    xdata, ydata = xdata.to_numpy(), ydata.to_numpy()
    start_time = time()
    cv_metrics = cross_validation(xdata, ydata, model, cross_validation_, probability)
    metrics = np.round(cv_metrics.mean() * 100, 2)
    print('\n   * Confidence interval measured by boostrap')
    confidence_intervals = bootstrapping(boot_iterations, confidence_level, cv_metrics)
    finish_time = time()
    print_metrics(metrics, confidence_intervals, confidence_level, name, np.round(finish_time - start_time, 2))
    box_plot(cv_metrics, name)
    metrics_plot(cv_metrics)
    return concat([metrics, confidence_intervals.T], axis=1, ignore_index=False).T


def save_trained_model(models, *argv):
    save_model = input('Do you want to save the trained model (Y/n): ').lower() == 'y'
    if save_model:
        for model, name in models:
            filename = ('Data\\Test\\{}_{}-{}.pkl'.format(name, argv[0], argv[1]) if len(argv) == 2 else
                        'Data\\Test\\{}_{}.pkl'.format(name, argv[0]) if len(argv) == 1 else
                        'Data\\Test\\{}.pkl'.format(name))
            with open(filename, 'wb') as file:
                dump(model, file)
            file = basename(filename)
            path = dirname(filename)
            print('\n   * {} is saved in folder {} \n'.format(file, path))


def save_correlation_matrix():
    save_data = input('\nDo you want to save the correlation matrix? (Y/n): ').lower()
    if save_data == 'y':
        name_fig = input(' + Enter name of plot: ')
        plt.savefig('Plot\\correlation_matrix_{}.png'.format(name_fig))
        path = dirname('Plot\\correlation_matrix_{}.png'.format(name_fig))
        print('\n   * correlation_matrix_{}.png is saved in folder {} \n'.format(name_fig, path))


def plot_correlation_matrix(xcorr, column_labels, n_frame):
    f = plt.figure(figsize=(19, 15))
    plt.matshow(xcorr, fignum=f.number)
    plt.xticks(range(0, len(column_labels), n_frame), column_labels[::n_frame], fontsize=14)
    plt.yticks(range(0, len(column_labels), n_frame), column_labels[::n_frame], fontsize=14)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title('Correlation Matrix', fontsize=16)
    save_correlation_matrix()
    plt.show()


def correlation_matrix(xdata, n_frame):
    global n_fingerprints
    xdata = xdata.iloc[:, :-n_fingerprints]
    column_labels = xdata.select_dtypes(['number']).columns
    xcorr = xdata.corr()
    plot_correlation_matrix(xcorr, column_labels, n_frame)
    return xcorr


def plot_loss_mlp(mlp):
    plot = input('Do you want to plot the loss? (Y/n)') == 'y'
    if plot:
        plt.figure(figsize=(19, 15))
        plt.plot(mlp.loss_curve_)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.ylim((0, 1))
        plt.title('Loss function', fontsize=16)
        plt.show()


def train_knn(x, y, param, test):
    n_neighbors = [int(k) for k in input(
        ' + Enter the number of neighbors [separated by commas if more than one]: ').split(',')]
    weights = [w for w in input(
        ' + Enter the weight function (uniform/distance) [separated by commas if more than one] : ').split(',')]
    text = '\n   * Setting: model KNN with k={} and w={} \n'.format(
        [n_neighbors if len(n_neighbors) > 1 else n_neighbors[0]][0],
        [weights if len(weights) > 1 else weights[0]][0])
    print(f'{text}')
    write_to_file(text)
    for k in n_neighbors:
        for w in weights:
            knn = KNeighborsClassifier(n_neighbors=k, weights=w)
            param['models'] = [[knn, 'KNN']]
            metrics = (DataFrame() if test else repeated_kfold_cv(**param))
            knn.fit(x, y)
            save_trained_model([[knn, 'KNN']], k, w)
            return metrics


def train_rf(x, y, param, test):
    n_estimators = [int(tr) for tr in input(
        ' + Enter the number of trees in the forest [separated by commas if more than one]: ').split(',')]
    max_depth = [int(d) for d in input(
        ' + Enter the maximum depth of the trees [separated by commas if more than one]: ').split(',')]
    text = '\n   * Setting: model RF with B={} and d={} \n'.format(
        [n_estimators if len(n_estimators) > 1 else n_estimators[0]][0],
        [max_depth if len(max_depth) > 1 else max_depth[0]][0])
    print(f'{text}')
    write_to_file(text)
    for tr in n_estimators:
        for d in max_depth:
            rf = RandomForestClassifier(n_estimators=tr, max_depth=d, random_state=123)
            param['models'] = [[rf, 'RF']]
            metrics = (DataFrame() if test else repeated_kfold_cv(**param))
            rf.fit(x, y)
            save_trained_model([[rf, 'RF']], tr, d)
            return metrics


def train_svm(x, y, param, test):
    nu = [float(n) for n in input(
        ' + Enter the parameter nu in (0,1] [separated by commas if more than one]: ').split(',')]
    text = '\n   * Setting: model SVM with nu={} \n'.format([nu if len(nu) > 1 else nu[0]][0])
    print(f'{text}')
    write_to_file(text)
    for n in nu:
        svm = NuSVC(nu=n, probability=True, random_state=123)
        param['models'] = [[svm, 'SVM']]
        metrics = (DataFrame() if test else repeated_kfold_cv(**param))
        svm.fit(x, y)
        save_trained_model([[svm, 'SVM']], n)
        return metrics


def train_mlp(x, y, param, test):
    mlp = MLPClassifier(random_state=123, verbose=True, tol=1e-8, alpha=0.0001, learning_rate='adaptive',
                        learning_rate_init=1e-5, max_iter=500, hidden_layer_sizes=(100, 50))
    param['models'] = [[mlp, 'MLP']]
    metrics = (DataFrame() if test else repeated_kfold_cv(**param))
    mlp.fit(x, y)
    save_trained_model([[mlp, 'MLP']])
    plot_loss_mlp(mlp)
    return metrics


def train(n_splits, n_repeats, confidence_level, probability, bootstrap):
    extension = input('Which dataset do you want to use for training? (base/complete/extended)')
    test = input('Do you want to train for testing any allosteric structure? (Y/n)').lower() == 'y'
    if test:
        structure = input('Which allosteric structure do you want to test? (1--19/1213) \n')
        dataset = 'dataset{}'.format(structure, structure)
    else:
        dataset = 'dataset'
    data = read_parquet('Data\\Test\\Datasets\\{}\\{}.parquet'.format(extension, dataset))
    x = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    print('\n   * {}.parquet has {} data and {} features \n'.format(dataset, x.shape[0], x.shape[1]))
    write_to_file('\n* {}.parquet has {} data and {} features'.format(dataset, x.shape[0], x.shape[1]))
    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    param = {'xdata': x, 'ydata': y, 'cross_validation_': cv, 'confidence_level': confidence_level,
             'probability': probability, 'boot_iterations': bootstrap}
    models = {'knn': train_knn, 'rf': train_rf, 'svm': train_svm, 'mlp': train_mlp}
    while True:
        model = input(' + Choose the model to train (KNN/RF/SVM/MLP): ').lower()
        if model in models:
            metrics = models[model](x, y, param, test)
            return x, y, metrics
        elif model == 'exit':
            return x, y, DataFrame()
        else:
            warn('You may have typed incorrectly, please enter the model to train again.')


if __name__ == "__main__":
    parameters = {
        'n_splits': 4,              # n. of folds for Repeated k-fold cross validation
        'n_repeats': 100,           # n. of times cross-validator needs to be repeated
        'confidence_level': 0.99,   # confidence level for the confidence intervals
        'probability': 0.95,        # minimum probability for a class to be chosen
        'bootstrap': 1000           # n. of iterations for bootstrap
    }
    X, Y, evaluation_metrics = train(**parameters)
    corr = correlation_matrix(X, 80)
