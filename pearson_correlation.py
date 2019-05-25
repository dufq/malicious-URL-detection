import seaborn as sns
import matplotlib.pyplot as plt


def draw(data_test, data_train):
    colormap = plt.cm.RdBu
    plt.figure(figsize=(86, 84))
    plt.title('Pearson Correlation of Features', y=1.05, size=87)
    x_f = data_test.drop(['hostname_is_ip', 'URL', 'label'], axis=1)
    sns.heatmap(x_f.astype(float).corr(), linewidths=0.1, vmax=1.0,
                square=True, cmap=colormap, linecolor='white', annot=True)
    plt.savefig("test.png")

    colormap = plt.cm.RdBu
    plt.figure(figsize=(86, 84))
    plt.title('Pearson Correlation of Features', y=1.05, size=87)
    x_f = data_train.drop(['hostname_is_ip', 'URL', 'label'], axis=1)
    sns.heatmap(x_f.astype(float).corr(), linewidths=0.1, vmax=1.0,
                square=True, cmap=colormap, linecolor='white', annot=True)
    plt.savefig("train.png")
    print('特征相关性绘制结束！')
