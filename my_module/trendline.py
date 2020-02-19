
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import optimize
from sklearn import linear_model
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import minimize

######################
#  最小二乗法（単純な最小2乗法ではない）
def add_linear_regression(df_X, df_Y, color='r'):
    """
    X:input of x ax(example:x_cols)
    Y:input of y ax(example:x_cols)
    """
    # fit_intercept	False に設定すると切片を求める計算を含めない。
    # 目的変数が原点を必ず通る性質のデータを扱うときに利用。 (デフォルト値: True)
    clf = linear_model.LinearRegression(fit_intercept=False)

    X = df_X.values.reshape(-1, 1)
    Y = df_Y.values.reshape(-1, 1)
    clf.fit(X, Y)
    # y_hat = clf.predict(X)

    # # 近似直線との誤差を計算する
    # diff = Y - y_hat

    # # 近似直線との誤差が、比較的が大きい点を無視するような、sample_weightを作成する
    # sample_weight = (np.max(np.abs(diff)) - np.abs(diff)
    #                  ).astype('float32').T[0] ** 2

    # # scikit-learnのsolverに、sample_weightを与えて、近似直線を得る
    # # 詳しくは"https://medium.com/micin-developers/decipher-github-lr-sw-40e519a13c0a"
    # clf.fit(X, Y, sample_weight=sample_weight)
    
    label_slope = "slope:" + str(round(clf.coef_.flatten()[0], 2)) + \
        "\n" + "R$^{2}$:" + str(round(clf.score(X, Y), 2))

    # 回帰直線
    plt.plot(X, clf.predict(X), label=label_slope, c=color)
    plt.legend()

    return clf.coef_.flatten()[0]


def add_weight_linear_regression(df_X, df_Y, color='r'):
    """
    X:input of x ax(example:x_cols)
    Y:input of y ax(example:x_cols)
    """
    # fit_intercept	False に設定すると切片を求める計算を含めない。
    # 目的変数が原点を必ず通る性質のデータを扱うときに利用。 (デフォルト値: True)
    clf = linear_model.LinearRegression(fit_intercept=False)

    X = df_X.values.reshape(-1, 1)
    Y = df_Y.values.reshape(-1, 1)
    clf.fit(X, Y)
    y_hat = clf.predict(X)

    # 近似直線との誤差を計算する
    diff = Y - y_hat

    # 近似直線との誤差が、比較的が大きい点を無視するような、sample_weightを作成する
    sample_weight = (np.max(np.abs(diff)) - np.abs(diff)
                     ).astype('float32').T[0] ** 2

    # scikit-learnのsolverに、sample_weightを与えて、近似直線を得る
    # 詳しくは"https://medium.com/micin-developers/decipher-github-lr-sw-40e519a13c0a"
    clf.fit(X, Y, sample_weight=sample_weight)
    
    label_slope = "slope:" + str(round(clf.coef_.flatten()[0], 2)) + \
        "\n" + "R$^{2}$:" + str(round(clf.score(X, Y), 2))

    # 回帰直線
    plt.plot(X, clf.predict(X), label=label_slope, c=color)
    plt.legend()

    return clf.coef_.flatten()[0]

#################
# 2次曲線近似
#Least squares method with scipy.optimize
def fit_func(parameter, x, y):
    a = parameter[0]
    b = parameter[1]
    c = parameter[2]
    residual = y-(a*x**2+b*x+c)
    return residual

#最小2乗法で変数Cを算出
def least_squares_method(X_list, Y_list):
    parameter0 = [0.,0.,0.]
    result = optimize.leastsq(fit_func,parameter0,args=(X_list,Y_list))
    return result

def plot_least_squares_result(result, X_list, ax):
    a_fit=result[0][0]
    b_fit=result[0][1]
    c_fit=result[0][2]
    print(a_fit, b_fit, c_fit)


    xlim = ax.get_xlim()
    # ylim = ax.get_ylim()
    x = np.arange(xlim[0], xlim[1], 0.1)

    #PLot#############たぶんここがおかしい
    ax.plot(x,a_fit*x**2+b_fit*x+c_fit, label='fitted parabora', color='r')
    ax.legend(loc='best',fancybox=True, shadow=True)

#######################