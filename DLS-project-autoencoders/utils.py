import numpy as np
import os
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
import torch
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn import preprocessing
from time import time
import pickle
from copy import deepcopy
from IPython.display import clear_output
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns


def history_plot(data, name='history'):
    """
    Вспомогательная ф-я для отрисовки истории обучения
    :param data: залогированная история обучения
    :param name: подпись графика

    """

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=data['train_losses'], label='train_losses').set(
        xlabel='epochs', ylabel='loss', title=name)
    sns.lineplot(data=data['val_losses'], label='val_losses').set(
        xlabel='epochs', ylabel='loss', title=name)
    plt.legend()


def image_show(original, encoded=False, num=10, t1='original',
               t2='decoded', cmap=None):
    """Вспомогательная ф-я для отрисовки оригинальных и восстановленных
    изображений.

    :param original: батч оригинальных изображений
    :param encoded: батч восстановленных изображений
    :param num: количество изображений в ряду
    :param t1: заголовок изображений первого ряда
    :param t2: заголовок изображений второго ряда
    :param cmap: цветовая карта
    """

    row = 2 if encoded is not False else 1
    plt.figure(figsize=(20, 5))
    for k in range(num):
        plt.subplot(row, num, k + 1)
        plt.imshow(np.moveaxis(original[k].numpy(), 0, 2), cmap=cmap)
        plt.title(t1)
        plt.axis('off')
        if row == 2:
            plt.subplot(row, num, k + 1 + num)
            plt.imshow(np.moveaxis(encoded[k].numpy(), 0, 2), cmap=cmap)
            plt.title(t2)
            plt.axis('off')


def train(model, opt, loss_fn, epochs, data_tr, data_val,
          sheduler=None, pictures=False, cmap=None, im_num=10,
          model_type='AE', data='faces', noise_factor=0):
    """
    :param model: модель
    :param opt: оптимизатор
    :param loss_fn: ф-я потерь
    :param epochs: количество эпох
    :param data_tr: данные для обучения
    :param data_val: данные для проверки
    :param sheduler: шедулер
    :param pictures: нужен ли вывод изображений
    :param cmap: цветовая карта вывода изображений
    :param im_num: кол-во изображений для вывода
    :param model_type: 'AE', 'VAE', 'CVAE' - тип модели
    :param data: 'faces' or 'mnist' - используемый датасет
    :param noise_factor: [0 : 1] - фактор шума
    :return: словарь с данными о обучении
    """
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    train_losses = []
    val_losses = []
    times = []
    train_codes = None
    val_codes = None

    for epoch in range(epochs):
        tic = time()
        print('* Epoch %d/%d' % (epoch + 1, epochs))
        # train mode
        avg_loss = 0
        model.train()
        for X_batch in data_tr:
            # если датасет MNIST, разделяем данные и лейблы
            if data == 'mnist':
                y_batch = X_batch[1]
                X_batch = X_batch[0]

            X_batch = X_batch.float().to(device)
            opt.zero_grad()

            # если модель conditional VAE, в модель будем подавать
            # помимо изображений еще и лейблы
            if model_type == 'CVAE':
                y_batch = y_batch.to(device)
                X_batch = (X_batch, y_batch)

            # forward

            # если шум не нулевой, генерируем тензор с шумом такого же
            # размера, как X_batch и прогоняем их сумму через модель
            if noise_factor:
                X_noisy = noise_factor * torch.randn(
                    (X_batch.size())).to(device)
                X_pred, latent_code = model(X_batch + X_noisy)
            # иначе просто X_batch
            else:
                X_pred, latent_code = model(X_batch)

            # если модель обычный АЕ, вызываем ф-ю потерь только с
            # X_pred и X_batch. Для остальных моделей будем отправлять
            # еще и mu и logsigma, которые лежат в latent_code
            if model_type == 'AE':
                loss = loss_fn(X_pred, X_batch)
            else:
                # для cond. VAE возьмем только данные,
                # тк в X_batch еще и лейблы
                if model_type == 'CVAE':
                    X_batch = X_batch[0]

                loss = loss_fn(X_pred, X_batch, *latent_code)

            loss.backward()
            opt.step()

            avg_loss += loss / len(data_tr)
            # если последняя эпоха и обычный АЕ, сохраним лат. вектора
            if epoch + 1 == epochs and model_type == 'AE':
                if train_codes is not None:
                    train_codes = torch.cat((train_codes, latent_code),
                                            dim=0)
                else:
                    train_codes = latent_code.clone()

        toc = time()
        if not pictures:
            clear_output(wait=True)

        print('loss: %f' % avg_loss)

        # testing mode
        # действия с переменными в зависимости от модели и типа данных
        # аналогичны действиям при обучении
        avg_loss_val = 0
        model.eval()

        for X_val in data_val:
            if data == 'mnist':
                y_val = X_val[1]
                X_val = X_val[0]

            X_val = X_val.float().to(device)
            if model_type == 'CVAE':
                y_val = y_val.to(device)
                X_val = (X_val, y_val)

            if noise_factor:
                X_val_noisy = noise_factor * torch.randn(
                    (X_val.size())).to(device)
                X_pred, latent_code = model(X_val + X_val_noisy)
            else:
                X_pred, latent_code = model(X_val)

            if model_type == 'CVAE':
                X_val = X_val[0]

            X_pred = X_pred.detach().cpu()
            X_val = X_val.detach().cpu()

            if model_type == 'AE':
                loss_val = loss_fn(X_pred, X_val)
            else:
                loss_val = loss_fn(X_pred, X_val, *latent_code)

            avg_loss_val += loss_val / len(data_val)
            # если последняя эпоха, сохраним латентные вектора
            if epoch + 1 == epochs and model_type == 'AE':
                if val_codes is not None:
                    val_codes = torch.cat((val_codes, latent_code),
                                          dim=0)
                else:
                    val_codes = latent_code.clone()

        # логирование истории обучения
        train_losses.append(float(avg_loss.detach().cpu().numpy()))
        val_losses.append(float(avg_loss_val.detach().cpu().numpy()))
        times.append(toc - tic)

        # визуализация
        if pictures:
            clear_output(wait=True)
            # если есть шум, то дополнительно еще выводим изображение
            # input с шумом
            if noise_factor:
                image = X_val + X_val_noisy.detach().cpu()
                image = np.clip(image, 0, 1)
                image_show(image, num=10, t1='Noisy input')

            image_show(X_val, X_pred, im_num, cmap=cmap)
            plt.suptitle(
                '%d / %d - loss: %f' % (epoch + 1, epochs, avg_loss))
            plt.show()

        if sheduler:
            sheduler.step()

    hist = {'train_losses': train_losses,
            'val_losses': val_losses,
            'times': times,
            'train_codes': train_codes,
            'val_codes': val_codes
            }

    return hist


def _main():
    pass


if __name__ == "__main":
    _main()
