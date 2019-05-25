from sklearn.cross_validation import train_test_split
import pandas as pd


def divide_data():
    origin_data_filename = "data/splited_data/X_train.csv"
    data = pd.read_csv(origin_data_filename)
    labels = data['label'].apply(lambda x: 0 if x == 'good' else 1)
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=62454, stratify=labels)
    print('划分好啦！')
    print(y_test.sum() / len(y_test))
    print(y_train.sum() / len(y_train))
    X_train.to_csv("data/splited_data/Training set.csv", index=False, encoding='utf-8')
    X_test.to_csv("data/splited_data/Cross Validation set.csv", index=False, encoding='utf-8')
    print('DONE!')


def read_data():
    filename_train = "data/splited_data/Training set.csv"
    filename_cv = "data/splited_data/Cross Validation set.csv"
    filename_test = "data/splited_data/Test set.csv"
    data_train = pd.read_csv(filename_train)
    data_cv = pd.read_csv(filename_cv)
    data_test = pd.read_csv(filename_test)
    return (data_train, data_cv, data_test)
