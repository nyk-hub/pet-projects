"""Модуль со всеми необходимыми функциями: \
glob(path), convert_to_csr(X), site_freqs(path_to_csv_files), \
prepare_train_set_with_fe(path_to_csv_files, site_freq_path, \
feature_names, session_length=10, window_size=10), \
plot_validation_curves(param_values, grid_cv_results_), \
save_data(num_users, window_size, session_length), \
plot_validation_curves(param_values, grid_cv_results_), \
plot_learning_curve(val_train, val_test, train_sizes, \
xlabel='Training Set Size', ylabel='score'), \
write_to_submission_file(predicted_labels, out_file, target='target', \
index_label="session_id"), add_time_features(df, X_sparse), \
add_day_feature(df, X_sparse)"""


import os
import numpy as np
import pandas as pd
import collections
import itertools
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.proportion import proportion_confint
from scipy.sparse import csr_matrix, hstack
from sklearn.model_selection import train_test_split, cross_val_score, \
    StratifiedKFold, GridSearchCV, learning_curve
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, \
SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder


import pickle
from scipy.sparse import csr_matrix


def arrays_to_vw(X, y=None, train=True, out_file='tmp.vw'):
    _s = ''
    for idx, line in enumerate(X):
        if train:
            label = str(y[idx]) + ' | '
        else:
            label = '1 | '
        _s += label + ' '.join(map(str, line)) + '\n'

    with open(out_file, "w") as f_out:
        f_out.write(_s)


def add_day_feature(df, X_sparse):
    days = pd.get_dummies(df['time1'].apply(lambda ts: ts.dayofweek))
    print(X_sparse.shape, days.shape)
    X = hstack([X_sparse, days.values])
    return X


def add_time_features(df, X_sparse):
    hour = df['time1'].apply(lambda ts: ts.hour)
    morning = ((hour >= 7) & (hour <= 11)).astype('int')
    day = ((hour >= 12) & (hour <= 18)).astype('int')
    evening = ((hour >= 19) & (hour <= 23)).astype('int')
    night = ((hour >= 0) & (hour <= 6)).astype('int')
    X = hstack([X_sparse, morning.values.reshape(-1, 1),
                day.values.reshape(-1, 1), evening.values.reshape(-1, 1),
                night.values.reshape(-1, 1)])
    return X


def glob(path):
    """Возвращает названия всех .csv файлов в папке"""
    return [name for name in os.listdir(path) if name.endswith(".csv")]


def convert_to_csr(X):
    row_ind = X.flatten()
    data = [1] * X.shape[0] * X.shape[1]
    col_ind = [X.shape[1] * x for x in range(X.shape[0] + 1)]
    X_csr = csr_matrix((data, row_ind, col_ind), dtype=int)[:,1:]
    return X_csr


def site_freqs(path_to_csv_files):
    # словарь для частот посещений сайтов
    site_freq = collections.defaultdict(int)
    # получим имена файлов в папке
    files = glob(path_to_csv_files)
    # сначала создадим полный словарь частот
    # в цикле читаем все файлы в папке
    for file_name in files:
        user_data = pd.read_csv(
            os.path.join(path_to_csv_files, file_name))
        # считаем частоты посещений пользователя
        # и обновляем общий словарь
        freq = user_data.site.value_counts()
        for site in freq.index:
            site_freq[site] += freq[site]
    # преобразуем словарь частот, добавив в него site_id
    site_id = 1
    _s = list(site_freq.items())
    _s.sort()  # отсортируемя сначала по алфавиту,
    # чтобы когда сортировали по частоте алфавитный порядок сайтов
    # с одинаковым количеством посещений сохранился
    for s in sorted(_s, key=lambda i: i[-1], reverse=True):
        site_freq[s[0]] = (site_id, s[1])
        site_id += 1
    return site_freq


def prepare_train_set_with_fe(path_to_csv_files, site_freq_path,
                              feature_names,
                              session_length=10, window_size=10):
    # считаем словарь для частот посещений сайтов из файла,
    # сохраненного в первом задании
    with open(site_freq_path, "rb") as f_in:
        site_freq = pickle.load(f_in)
        # получим имена файлов в папке
    files = glob(path_to_csv_files)
    train_data = []  # основной датафрейм для обучающих данных

    # пройдем в цикле по всем файлам и заполним ДФ для обучения
    for file_name in files:
        users_data = pd.read_csv(
            os.path.join(path_to_csv_files, file_name))
        user_id = int(
            file_name[4:-4])  # id пользователя из имени файла
        # преобразуем столбец времени в pd.datetime
        users_data.timestamp = users_data.timestamp.apply(
            pd.to_datetime)
        # sc == id сессии пользователя
        for sc in range((users_data.shape[0] - 1) // window_size + 1):
            user_d = []  # массив для собирания данных о пользователе
            # i-й сайт в сессии
            session_start_time = users_data.timestamp[
                sc * window_size]  # время начала сессии
            # переменная для хранения времени посещения
            # последнего сайта в сессии
            session_end_time = 0
            for i in range(session_length):
                curr_site_id = sc * window_size + i
                if curr_site_id < users_data.shape[0]:
                    user_d.append(
                        site_freq[users_data.site[curr_site_id]][0])
                    session_end_time = users_data.timestamp[
                        curr_site_id]
                else:
                    user_d.append(0)
            # посчитаем количество уникальных элементов
            uniq = collections.Counter(user_d)
            if 0 in uniq:
                del uniq[0]

            user_d.append(
                (session_end_time - session_start_time).seconds)
            user_d.append(len(uniq))
            user_d.append(session_start_time.hour)
            user_d.append(session_start_time.dayofweek)
            user_d.append(user_id)
            train_data.append(user_d)
    train_data = pd.DataFrame(train_data, columns=feature_names)
    return train_data


def save_data(num_users, window_size, session_length):
    X_sparse, y = prepare_sparse_train_set_window(
        os.path.join(PATH_TO_DATA, f'{num_users}users'),
        os.path.join(PATH_TO_DATA, f'site_freq_{num_users}users.pkl'),
        session_length, window_size)
    x_name = os.path.join(PATH_TO_DATA,
    f'X_sparse_{num_users}users_s{session_length}_w{window_size}.pkl')
    y_name = os.path.join(PATH_TO_DATA,
            f'y_{num_users}users_s{session_length}_w{window_size}.pkl')
    with open(x_name, 'wb') as X_pkl:
        pickle.dump(X_sparse, X_pkl)
    with open(y_name, 'wb') as y_pkl:
        pickle.dump(y, y_pkl)

    return X_sparse.shape[0]


def plot_validation_curves(param_values, grid_cv_results_):
    train_mu, train_std = grid_cv_results_['mean_score_time'], \
                          grid_cv_results_['std_score_time']
    valid_mu, valid_std = grid_cv_results_['mean_test_score'], \
                          grid_cv_results_['std_test_score']
    train_line = plt.plot(
        param_values, train_mu, '-', label='train', color='green')
    valid_line = plt.plot(
        param_values, valid_mu, '-', label='test', color='red')
    plt.fill_between(param_values, train_mu - train_std, train_mu + \
                     train_std, edgecolor='none',
                     facecolor=train_line[0].get_color(), alpha=0.2)
    plt.fill_between(param_values, valid_mu - valid_std, valid_mu + \
                     valid_std, edgecolor='none',
                     facecolor=valid_line[0].get_color(), alpha=0.2)
    plt.legend()


def model_assessment(estimator, path_to_X_pickle, path_to_y_pickle, cv,
                     random_state=17, test_size=0.3):
    '''
    Estimates CV-accuracy for (1 - test_size) share of (X_sparse, y) loaded from path_to_X_pickle
    and path_to_y_pickle and holdout accuracy for (test_size) share of (X_sparse, y).
    The split is made with stratified train_test_split with params random_state and test_size.

    :param estimator – Scikit-learn estimator (classifier or regressor)
    :param path_to_X_pickle – path to pickled sparse X (instances and their features)
    :param path_to_y_pickle – path to pickled y (responses)
    :param cv – cross-validation as in cross_val_score (use StratifiedKFold here)
    :param random_state –  for train_test_split
    :param test_size –  for train_test_split

    :returns mean CV-accuracy for (X_train, y_train) and accuracy for (X_valid, y_valid) where (X_train, y_train)
    and (X_valid, y_valid) are (1 - test_size) and (testsize) shares of (X_sparse, y).
    '''

    with open(path_to_X_pickle, 'rb') as X_sparse_users_pkl:
        X_sparse_users = pickle.load(X_sparse_users_pkl)
    with open(path_to_y_pickle, 'rb') as y_users_pkl:
        y_users = pickle.load(y_users_pkl)

    X_train, X_test, y_train, y_test = train_test_split(
        X_sparse_users, y_users, random_state=random_state,
        test_size=test_size, stratify=y_users)
    model = estimator.fit(X_train, y_train)
    cv_score = np.mean(
        cross_val_score(estimator, X_train, y_train, cv=cv))
    acc = accuracy_score(y_test, model.predict(X_test))
    return cv_score, acc


def plot_learning_curve(val_train, val_test, train_sizes,
                        xlabel='Training Set Size', ylabel='score'):
    def plot_with_err(x, data, **kwargs):
        mu, std = data.mean(1), data.std(1)
        lines = plt.plot(x, mu, '-', **kwargs)
        plt.fill_between(x, mu - std, mu + std, edgecolor='none',
                         facecolor=lines[0].get_color(), alpha=0.2)
    plot_with_err(train_sizes, val_train, label='train')
    plot_with_err(train_sizes, val_test, label='valid')
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.legend(loc='lower right');


def write_to_submission_file(predicted_labels, out_file,
                             target='target', index_label="session_id"):
    # turn predictions into data frame and save as csv file
    predicted_df = pd.DataFrame(predicted_labels,
                                index = np.arange(1, predicted_labels.shape[0] + 1),
                                columns=[target])
    predicted_df.to_csv(out_file, index_label=index_label)


def main():
    pass


if __name__ == '__main__':
    main()
