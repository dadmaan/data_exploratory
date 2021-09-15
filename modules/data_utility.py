from subprocess import call

import tensorflow as tf
import os
import pandas as pd
from pandas.plotting import scatter_matrix, table
import numpy as np
import pydot
import datetime
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
import seaborn as sns
import plotly.express as px
# plt.style.use('fivethirtyeight')

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
sns.set_style("darkgrid")
sns.set(font_scale=1.0)

def plot_features(df, features, figsize=(15, 5)):
    fig = px.line(df, x=df.index, y=features, width=1500, height=800)
    fig.update_xaxes(rangeslider_visible=True)
    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20)
        )
    return fig

def plot_boxplot(df, features, orient='v'):
    fig, ax = plt.subplots()
    sns.boxplot(data=df[features], orient=orient, linewidth=2)
    plt.tight_layout()
    return fig
    
    
def standard_scaler(data, path=None, title=None):

    # create a scaler object
    std_scaler = StandardScaler()
    # fit and transform the data
    df_std = pd.DataFrame(std_scaler.fit_transform(data), columns=data.columns, index=data.index)

    df = df_std.melt(var_name='Column', value_name='Normalized')
    fig = plt.figure(figsize=(12, 6))
    ax = sns.violinplot(x='Column', y='Normalized', data=df)
    _ = ax.set_xticklabels(data.keys(), rotation=90)
    plt.title("Violin plot of normalized data{}".format(' - ' + title if title else ''))
    plt.tight_layout()
    if path:
        plt.savefig(path)
        
    return df_std, fig

def minmax_scaler(data, path=None, title=None):

    # create a scaler object
    scaler = MinMaxScaler(feature_range=(0, 1))
    # fit and transform the data
    df_minmax = pd.DataFrame(scaler.fit_transform(data), columns=data.columns, index=data.index)

    df = df_minmax.melt(var_name='Column', value_name='Normalized')
    fig = plt.figure(figsize=(12, 6))
    ax = sns.violinplot(x='Column', y='Normalized', data=df)
    _ = ax.set_xticklabels(data.keys(), rotation=90)
    plt.title("Violin plot of normalized data{}".format(' - ' + title if title else ''))
    plt.tight_layout()
    if path:
        plt.savefig(path)
    
        
    return df_minmax, fig

def robust_scaler(data, path=None, title=None):
    # create a scaler object
    scaler = RobustScaler()
    # fit and transform the data
    df_robust = pd.DataFrame(scaler.fit_transform(data), columns=data.columns, index=data.index)
    
    df = df_robust.melt(var_name='Column', value_name='Normalized')
    fig = plt.figure(figsize=(12, 6))
    ax = sns.violinplot(x='Column', y='Normalized', data=df)
    _ = ax.set_xticklabels(data.keys(), rotation=90)
    plt.title("Violin plot of robust normalized data{}".format(' - ' + title if title else ''))
    plt.tight_layout()
    if path:
        plt.savefig(path)
        
    return df_robust, fig

def get_pca(df):
    df = df_clean(df)
    d,_ = standard_scaler(df)
    # create the PCA instance
    pca = PCA(0.95).fit(d)
    # transform data
    data_pca = pca.transform(d)
    return pca, data_pca
    
def plot_pca_loading(score, coeff, labels=None, scale=1, xlim=1, ylim=1, figsize=(20,20), title=None, path=None, fontsize=18):
    
    fig = plt.figure(figsize=figsize)
    
    xs = score[0]
    ys = score[1]
    n = coeff.shape[0]
    
    # scale the data for better visibility
    scalex = scale/(xs.max() - xs.min())
    scaley = scale/(ys.max() - ys.min())
    
    # define the color map for labels
    colors = cm.rainbow(np.linspace(0, 1, len(labels)))
    
    # plot the components
    plt.scatter(xs * scalex,ys * scaley, color='g', alpha=0.5)
    
    for i, c in enumerate(colors):
        # plot the arrows
        plt.arrow(0, 0, coeff[i,0], coeff[i,1], color = 'black', alpha = 0.5, 
                  linestyle = '-', linewidth = 1.5, overhang=0.2, head_width=0.02)
        # plot the texts
        if labels is None:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = c, ha = 'center', va = 'center',
                     fontsize=fontsize, bbox=dict(facecolor='red', alpha=0.5))
        else:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'black', ha = 'center', va = 'center',
                     fontsize=fontsize, bbox=dict(facecolor='red', alpha=0.5))
            
    # set the axis limit
    plt.xlim(-xlim,xlim)
    plt.ylim(-ylim,ylim)
    
    # set the axis labels
    plt.xlabel("Principal Component{}".format(1), fontsize=fontsize)
    plt.ylabel("Principal Component{}".format(2), fontsize=fontsize)
    
    if title:
        plt.title(title)
        
    # enable grid
    plt.grid()
    plt.tight_layout()
    
    if path:
        plt.savefig(path)
    
    return fig
def df_read(path):
    df = pd.read_csv(path)
    date_time = pd.to_datetime(df.pop(df.columns[0]))
    df = df.set_index(date_time, drop=True)
    
    return df


def df_clean(data, process=False):
    
    # offset correction

    if any(data.index.duplicated()) is True:
        data = data[~data.index.duplicated()]
        print('Duplicated values removed.')
    
    # remove Nan values
    if any(data.isnull()) is True:
        data = data.dropna()
    #     df = df.fillna(0)
        print('Null values removed.')
        
    if process:
        data[['Grid [A]', 'Grid [VAC]']] = data[['Grid [A]', 'Grid [VAC]']] + 0.8
        query1 = (data['Battery Pack [A]'] > -360) & (data['Battery Pack [A]'] < 60)
        query2 = (data['Battery Pack [VDC]'] > 230) & (data['Battery Pack [VDC]'] < 360)

        query3 = (data['Grid [A]'] > 0) & (data['Grid [A]'] < 33)
        query4 = (data['Grid [VAC]'] > 190) & (data['Grid [VAC]'] < 250)

        query5 = (data['Battery Pack SOC'] != 0) & (data['Battery Pack [VDC]'] != 0)
        query6 = (data['Grid [A]'] != -3276.8)

        for query in [query1, query2, query3, query4, query5, query6]:
            data = data[query]
    return data

def get_boxplot(data, features, title=None, path=None):

    fig, axes = plt.subplots(len(features),1,figsize=(8,30))
    for i, column in enumerate(features):
    #     index = i
        ax = sns.boxplot(data=data[column], orient='h', linewidth=2, ax=axes[i])
    # ax = sns.swarmplot(data=df, color=".25")
        axes[i].set_title("{t} - {c} box plot".format(c=column, t=title if title else 'eSled'))
        plt.tight_layout()
    if path:
        plt.savefig(path)
    # plt.show()
    return fig

def highlight_min(data):
    color= 'red'
    attr = 'background-color: {}'.format(color)

    if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
        is_min = data == data.min()
        return [attr if v else '' for v in is_min]
    else: 
        is_min = data.groupby(level=0).transform('min') == data
        return pd.DataFrame(np.where(is_min, attr, ''),
                            index=data.index, columns=data.columns)
    
def highlight_max(data):
    color= 'green'
    attr = 'background-color: {}'.format(color)

    if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
        is_max = data == data.max()
        return [attr if v else '' for v in is_max]
    else: 
        is_max = data.groupby(level=0).transform('max') == data
        return pd.DataFrame(np.where(is_max, attr, ''),
                            index=data.index, columns=data.columns)

def magnify():
    return [dict(selector="th",
             props=[("font-size", "10pt")]),
        dict(selector="td",
             props=[('padding', "0em 0em")]),
        dict(selector="th:hover",
             props=[("font-size", "16pt")]),
        dict(selector="tr:hover td:hover",
             props=[('max-width', '200px'),
                    ('font-size', '12pt')])
           ]

def get_correlation_matrix(data, features, font_size=14, font_color='black', path=None, title=None):
    
    fig = plt.figure()

    plt.rcParams.update({'font.size': font_size})
    plt.rcParams['text.color'] = font_color
    
    axes = scatter_matrix(data[features], figsize=(20,20))
    # calculate correlation matrix
    corr = data[features].corr().to_numpy()
    for i, j in zip(*plt.np.triu_indices_from(axes, k=1)):
        axes[i, j].annotate("corr: %.3f" %corr[i,j], (0.5, 0.9), xycoords='axes fraction', ha='center', va='center')
    
    if title:
        plt.suptitle(title)
    plt.tight_layout()
    if path:
        plt.savefig(path)
    return axes
    

def get_pairplot(df, columns):
    fig = sns.pairplot(df, diag_kind='kde')
    return fig

def get_correlation_heatmap(data, features, path=None, title=None):
    
    width = len(features)*3
    height = len(features)*2.5
    # calculate correlation matrix
    corr = data[features].corr().to_numpy()
    # plot
    fig = plt.figure(figsize=(width,height))
    ax = sns.heatmap(corr, annot=True, annot_kws={"fontsize":24},
                xticklabels=features, yticklabels=features)

    ax.set_xticklabels(ax.get_xmajorticklabels(), 
                                    fontsize = 24, rotation=90)
    ax.set_yticklabels(ax.get_ymajorticklabels(), 
                                    fontsize = 24, rotation=0)
    if title:
        plt.title(title)
    plt.tight_layout()
    if path:
        plt.savefig(path)
    # plt.show()
    return fig
    
    
def plot_auto_correlation(data, features, lags, title=None, path=None, figsize=None, layout=None):
    
    nrows = len(features)
    ncols = 2
    if layout is not None:
        nrows = layout[0]
        ncols = layout[1]
        
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, 
                             figsize=(20,len(features)*3) if figsize is None else figsize)
    for i, feature in enumerate(features):
        # Calculate ACF and PACF upto 50 lags
        acf_50 = acf(data[feature],fft=True, nlags=lags)
        pacf_50 = pacf(data[feature], nlags=lags)

        # Draw Plot
        plot_acf(acf_50, lags=lags, ax=axes[i, 0] if len(features)>1 else axes[0], title=feature+" Auto-correlation - {}".format(title if title else 'eSled'))
        # the pcaf plot works for less than 50% of the provided number of samples
        plot_pacf(pacf_50, lags=lags/2-1, ax=axes[i, 1] if len(features)>1 else axes[1], title=feature+" Partial Auto-correlation - {}".format(title if title else 'eSled'))
    plt.tight_layout()
    
    if path:
        plt.savefig(path)
    return fig


def get_fft_plot(data, features, title=None, path=None, dataset_yearly_resolution=24*365.2524):
    rows = np.ceil(len(features)/2)
    
    fig = plt.figure(figsize=(10, 4*rows))
    for i, feature in enumerate(features):
        # here we determine which frequency is important 
        fft = tf.signal.rfft(data[feature].dropna())
        f_per_dataset = np.arange(0, len(fft))

        n_samples_h = len(data[feature])
        hours_per_year = dataset_yearly_resolution
        years_per_dataset = n_samples_h/(hours_per_year)

        f_per_year = f_per_dataset/years_per_dataset
        
        plt.subplot(rows, 2, i+1)
        plt.step(f_per_year, np.abs(fft))

        axes = plt.gca()
        axes.set_xscale('log')
        axes.set_ylim(0, max(axes.get_ylim()))
        axes.set_xlim([0.1, max(axes.get_xlim())])
        axes.set_xticks([1, 7, 30, 365.2524])
        axes.set_xticklabels(['1/Year','1/Month', '1/Week', '1/Day'], 
                            fontsize=8, rotation=90)
        axes.set_xlabel('Freaquency (log scale)', fontsize=10)
        axes.set_title("Fourier Analysis of "+feature, fontsize=10)
        plt.tight_layout()
    if path:
        plt.savefig(path)
        
    return fig


def get_barplot(df, columns, period='M'):
    data = df.copy()
    data[period] = data.index.to_period(period)

    if period =='Y':
        rows = len(columns)
        cols = 2
        width = 10
        height = rows*5

    elif period == 'M':
        rows = len(columns)
        cols = 2
        width = len(data[period].unique())*1.5
        height = rows*5

    else:
        rows = len(columns)
        cols = 1
        width = 30
        height = rows*8

    fig = plt.figure(figsize=(width, height))
    for i, feature in enumerate(columns):
        plt.subplot(rows, cols, i+1)
        ax = sns.barplot(x=period, y=feature, data=data, palette='cool')
        ax.set_title(feature)
        ax.set_xticklabels(ax.get_xmajorticklabels(), 
                                        fontsize = 12, rotation=90)
    
    plt.tight_layout()
    return fig

def plot_variable_importances(model, features_list, title='Feature Importances', fontsize=18):
    """ plot the importance of each variable for Random Forest Regression trees.
    Return: list of variable importances.
    """
    
    importances = list(model.feature_importances_)

    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(features_list, importances)]

    feature_importances = sorted(feature_importances, key= lambda x: x[1], reverse=True)

    # [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]
    width = 8
    # 5 if len(features_list) < 3 else len(features_list) *1.2
    fig = plt.figure(figsize=(width,6))
    # list of locations to plot
    x_values = list(range(len(importances)))

    plt.bar(x_values, importances, orientation='vertical', color='black')

    plt.xticks(x_values, features_list, rotation='vertical')

    plt.ylabel('Importance', fontsize=fontsize)
    plt.xlabel('Variable', fontsize=fontsize)
    plt.title(title)
    plt.tight_layout()

    return feature_importances, fig

def get_training_examples(data, target_feature, test_size=.2, random_state=42):
    """ Create the training and test examples by one-hot-encoding the non-numerical values."""
    
    ds_encoded = pd.get_dummies(data)
    
    if ds_encoded.isnull().values.any():
        raise Exception('There exist Nan values.')
    
    # set the labels to target prediction; Oslo Prices
    labels = np.array(ds_encoded[target_feature])
    # remove the target from dataset
    ds_encoded = ds_encoded.drop([target_feature], axis=1)
    # save a list of columns for future use
    features_list = list(ds_encoded.columns)
    # convert dataset to numpy array to feed into model
    ds_encoded = np.array(ds_encoded)

    # split dataset into training and test
    train_dataset, test_dataset, train_labels, test_labels = train_test_split(ds_encoded, labels,
                                                                         test_size=test_size, random_state=random_state)
    
    return train_dataset, test_dataset, train_labels, test_labels, features_list