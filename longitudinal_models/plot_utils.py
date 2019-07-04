from matplotlib import pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
# matplotlib.use('Qt5Agg')


def plot_multiple_ts(df,
                     group='brcid',
                     x_axis='age_at_score_upper_bound',
                     y_axis='score_combined'):
    to_plot = df.pivot_table(index=x_axis, columns=group, values=y_axis, aggfunc='mean')
    to_plot.plot()


def plot_missing_values(df, cols_to_plot):
    # heatmap of missing values
    df.replace(['null', 'unknown', 'other', 'not disclosed', 'not known'], np.nan, inplace=True)
    to_plot = [x for x in cols_to_plot if x in df.columns]
    print('excluded values:', [x for x in cols_to_plot if x not in df.columns])
    sns.heatmap(df[to_plot].isnull(), cbar=False)
    return 0


def plot_correl(df, cols_to_plot):
    df = df[cols_to_plot].replace(['null', 'unknown', np.nan, 'other', 'not disclosed'], None)

    # http://stronginference.com/missing-data-imputation.html

    # correlations
    colormap = plt.cm.RdBu
    plt.figure(figsize=(32, 10))
    plt.title('Pearson Correlation of Features', y=1.05, size=15)
    sns.heatmap(df.corr(), linewidths=0.1, vmax=1.0, square=True, cmap=colormap, linecolor='white',
                annot=True)

