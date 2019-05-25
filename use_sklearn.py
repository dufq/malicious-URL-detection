from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import BaseDiscreteNB
from sklearn.naive_bayes import BaseNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
import lightgbm as lgb
from lightgbm.sklearn import LGBMClassifier
from sklearn.externals import joblib
import os


colnames=['URL', 'label', 'URL_a', 'URL_b', 'URL_c', 'URL_d', 'URL_e', 'URL_f',
       'URL_g', 'URL_h', 'URL_i', 'URL_j', 'URL_k', 'URL_l', 'URL_m', 'URL_n',
       'URL_o', 'URL_p', 'URL_q', 'URL_r', 'URL_s', 'URL_t', 'URL_u', 'URL_v',
       'URL_w', 'URL_x', 'URL_y', 'URL_z', 'URL_depth', 'URL_len', 'exe_flag',
       'badword_n', 'popular_n', 'URL_point', 'http_flag', 'letter_ratio',
       'at_flag', 'dig_ratio', 'special_ch', 'special_ch_kind', 'TLD_id',
       'hash_token_n', 'hostname_a', 'hostname_b', 'hostname_c',
       'hostname_ch_n', 'hostname_d', 'hostname_dig_ratio', 'hostname_e',
       'hostname_entropy', 'hostname_f', 'hostname_g', 'hostname_h',
       'hostname_i', 'hostname_is_ip', 'hostname_j', 'hostname_k',
       'hostname_l', 'hostname_len', 'hostname_letter_ratio', 'hostname_m',
       'hostname_n', 'hostname_o', 'hostname_p', 'hostname_point_n',
       'hostname_q', 'hostname_r', 'hostname_s', 'hostname_std', 'hostname_t',
       'hostname_token_n', 'hostname_u', 'hostname_v', 'hostname_w',
       'hostname_x', 'hostname_y', 'hostname_z', 'pathname_ch_kind',
       'pathname_depth', 'pathname_len', 'pathname_longest_token',
       'pathname_std', 'pathname_token_n', 'search_and_n', 'search_len',
       'search_std', 'search_token_n']

def report(results, n_top=5488):
    f = open('\grid_search_rf.txt', 'w')
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            f.write("Model with rank: {0}".format(i) + '\n')
            f.write("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                results['mean_test_score'][candidate],
                results['std_test_score'][candidate]) + '\n')
            f.write("Parameters: {0}".format(results['params'][candidate]) + '\n')
            f.write("\n")
    f.close()


def selectRFParam(url_features, labels):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV
    clf_RF = RandomForestClassifier()
    param_grid = {"max_depth": [3, 15],
                  "min_samples_split": [3, 5, 10],
                  "min_samples_leaf": [3, 5, 10],
                  "bootstrap": [True, False],
                  "criterion": ["gini", "entropy"],
                  "n_estimators": range(10, 50, 10),
                  "oob_score": [True, False]}
    X_train, X_test, y_train, y_test = train_test_split(url_features, labels, test_size=0.1)
    # "class_weight": [{0:1,1:13.24503311,2:1.315789474,3:12.42236025,4:8.163265306,5:31.25,6:4.77326969,7:19.41747573}],
    # "max_features": range(3,10),
    # "warm_start": [True, False],
    # "verbose": [True, False]}
    grid_search = GridSearchCV(clf_RF, param_grid=param_grid, n_jobs=4)
    grid_search.fit(X_train, y_train)  # 传入训练集矩阵和训练样本类标
    print(grid_search.cv_results_)
    report(grid_search.cv_results_)


def multi_machine_learing_models(data_train, data_cv):
    print('正在训练模型！')
    data_train=pd.concat([data_train,data_cv],axis=0)
    y_train = data_train['label'].apply(lambda x: 0 if x == 'good' else 1)
    y_test = data_cv['label'].apply(lambda x: 0 if x == 'good' else 1)

    X_train = data_train.drop(['URL', 'label'], axis=1)
    X_test = data_cv.drop(['URL', 'label'], axis=1)

    filename_bayes = 'classifier_model\c_bayes.model'
    filename_LGB = 'classifier_model\c_LGB.model'
    filename_ada = 'classifier_model\c_ada.model'
    filename_rf = 'classifier_model\c_rf.model'
    filename_decision_tree = 'classifier_model\c_decision_tree.model'
    filename_lgs = 'classifier_model\c_lgs.model'

    vote = []
    for i in range(len(y_test)):
        vote.append(0)

    bayes = BernoulliNB()
    bayes.fit(X_train, y_train)
    print('\nbayes模型的准确度:', bayes.score(X_test, y_test))
    predict = bayes.predict(X_test)
    vote = list(map(lambda x: x[0] + x[1], zip(predict, vote)))
    precision = metrics.precision_score(y_test, predict)
    recall = metrics.recall_score(y_test, predict)
    print("precison:", precision)
    print("recall:", recall)
    joblib.dump(bayes, filename_bayes)

    gbc = LGBMClassifier(n_estimators=200, objective='binary')
    gbc.fit(X_train, y_train)
    print('LGBMClassifier模型的准确度:', gbc.score(X_test, y_test))
    predict = gbc.predict(X_test)
    vote = list(map(lambda x: 3 * x[0] + x[1], zip(predict, vote)))
    precision = metrics.precision_score(y_test, predict)
    recall = metrics.recall_score(y_test, predict)
    print("precison:", precision)
    print("recall:", recall)
    joblib.dump(gbc, filename_LGB)

    ada = AdaBoostClassifier(n_estimators=100)  # 迭代100次
    ada.fit(X_train, y_train)
    print('ada模型的准确度:', ada.score(X_test, y_test))
    predict = ada.predict(X_test)
    vote = list(map(lambda x: 2 * x[0] + x[1], zip(predict, vote)))
    precision = metrics.precision_score(y_test, predict)
    recall = metrics.recall_score(y_test, predict)
    print("precison:", precision)
    print("recall:", recall)
    joblib.dump(ada, filename_ada)

    rf = RandomForestClassifier(n_estimators=100, oob_score=True)
    rf.fit(X_train, y_train)
    print('\nrf模型的准确度:', rf.score(X_test, y_test))
    predict = rf.predict(X_test)
    vote = list(map(lambda x: x[0] * 3 + x[1], zip(predict, vote)))
    precision = metrics.precision_score(y_test, predict)
    recall = metrics.recall_score(y_test, predict)
    print("precison:", precision)
    print("recall:", recall)
    joblib.dump(rf, filename_rf)

    decision_tree = tree.DecisionTreeClassifier()
    decision_tree.fit(X_train, y_train)
    print('\ndecision_tree模型的准确度:', decision_tree.score(X_test, y_test))
    predict = decision_tree.predict(X_test)
    vote = list(map(lambda x: x[0] * 2 + x[1], zip(predict, vote)))
    precision = metrics.precision_score(y_test, predict)
    recall = metrics.recall_score(y_test, predict)
    print("precison:", precision)
    print("recall:", recall)
    joblib.dump(decision_tree, filename_decision_tree)

    lgs = LogisticRegression()
    lgs.fit(X_train, y_train)
    print('\nLogisticRegression模型的准确度:', lgs.score(X_test, y_test))
    predict = lgs.predict(X_test)
    vote = list(map(lambda x: x[0] * 2 + x[1], zip(predict, vote)))
    precision = metrics.precision_score(y_test, predict)
    recall = metrics.recall_score(y_test, predict)
    print("precison:", precision)
    print("recall:", recall)
    joblib.dump(lgs, filename_lgs)

    print('\n投票结果：')
    vote_r = []
    for i in range(len(vote)):
        if vote[i] >= 3:
            vote_r.append(1)
        else:
            vote_r.append(0)
    precision = metrics.precision_score(y_test, vote_r)
    recall = metrics.recall_score(y_test, vote_r)
    acc = metrics.accuracy_score(y_test, vote_r)
    print('准确度：', acc)
    print("precison:", precision)
    print("recall:", recall)


def loadModel():
    filename_bayes = 'classifier_model\c_bayes.model'
    filename_LGB = 'classifier_model\c_LGB.model'
    filename_ada = 'classifier_model\c_ada.model'
    filename_rf = 'classifier_model\c_rf.model'
    filename_decision_tree = 'classifier_model\c_decision_tree.model'
    filename_lgs = 'classifier_model\c_lgs.model'
    if os.path.exists(filename_bayes):
        bayes = joblib.load(filename_bayes)
        print('成功读取贝叶斯模型！')
    if os.path.exists(filename_LGB):
        gbc = joblib.load(filename_LGB)
        print('成功读取梯度提升树模型！')
    if os.path.exists(filename_ada):
        ada = joblib.load(filename_ada)
        print('成功读取AdaBoost模型！')
    if os.path.exists(filename_rf):
        rf = joblib.load(filename_rf)
        print('成功读取随机森林模型！')
    if os.path.exists(filename_decision_tree):
        dt = joblib.load(filename_decision_tree)
        print('成功读取决策树模型！')
    if os.path.exists(filename_lgs):
        lgs = joblib.load(filename_lgs)
        print('成功读取逻辑回归模型！')
    return (bayes, gbc, ada, rf, dt, lgs)


def vote_to_predict(X_test, y_test):
    bayes, gbc, ada, rf, dt, lgs = loadModel()
    vote = []
    for i in range(len(y_test)):
        vote.append(0)

    print('\nbayes模型的准确度:', bayes.score(X_test, y_test))
    predict = bayes.predict(X_test)
    vote = list(map(lambda x: x[0] + x[1], zip(predict, vote)))
    precision = metrics.precision_score(y_test, predict)
    recall = metrics.recall_score(y_test, predict)
    print("precison:", precision)
    print("recall:", recall)


    print('\nada模型的准确度:', ada.score(X_test, y_test))
    predict = ada.predict(X_test)
    vote = list(map(lambda x: 2 * x[0] + x[1], zip(predict, vote)))
    precision = metrics.precision_score(y_test, predict)
    recall = metrics.recall_score(y_test, predict)
    print("precison:", precision)
    print("recall:", recall)

    print('\nrf模型的准确度:', rf.score(X_test, y_test))
    predict = rf.predict(X_test)
    vote = list(map(lambda x: x[0] * 20 + x[1], zip(predict, vote)))
    precision = metrics.precision_score(y_test, predict)
    recall = metrics.recall_score(y_test, predict)
    print("precison:", precision)
    print("recall:", recall)

    print('\ndecision_tree模型的准确度:', dt.score(X_test, y_test))
    predict = dt.predict(X_test)
    vote = list(map(lambda x: x[0] * 2 + x[1], zip(predict, vote)))
    precision = metrics.precision_score(y_test, predict)
    recall = metrics.recall_score(y_test, predict)
    print("precison:", precision)
    print("recall:", recall)

    print('\nLogisticRegression模型的准确度:', lgs.score(X_test, y_test))
    predict = lgs.predict(X_test)
    vote = list(map(lambda x: x[0] * 2 + x[1], zip(predict, vote)))
    precision = metrics.precision_score(y_test, predict)
    recall = metrics.recall_score(y_test, predict)
    print("precison:", precision)
    print("recall:", recall)

    print('\nLGBMClassifier模型的准确度:', gbc.score(X_test, y_test))
    predict = gbc.predict(X_test)
    vote = list(map(lambda x: 10 * x[0] + x[1], zip(predict, vote)))
    precision = metrics.precision_score(y_test, predict)
    recall = metrics.recall_score(y_test, predict)
    print("precison:", precision)
    print("recall:", recall)
    print('\n投票结果：')
    print(vote)
    vote_r = []
    sum_rf = 0
    sum_gbt = 0
    sum_vote_pos = 0
    sum_all_voted = 0
    sum_pos = 0
    for i in range(len(vote)):
        if vote[i] >= 7:
            vote_r.append(1)
            sum_pos += 1
            if vote[i] >= 37:
                sum_all_voted += 1
            elif vote[i] > 20:
                sum_rf += 1
            elif vote[i] > 10:
                sum_gbt += 1
            else:
                sum_vote_pos += 1
        else:
            vote_r.append(0)
    precision = metrics.precision_score(y_test, vote_r)
    recall = metrics.recall_score(y_test, vote_r)
    acc = metrics.accuracy_score(y_test, vote_r)
    print('准确度：', acc)
    print("precison:", precision)
    print("recall:", recall)
    print("总共有 ", sum_pos, " 条URL被预测为恶意。")
    print("被全体投票的：", sum_all_voted)
    print("被随机森林投票的：", sum_rf)
    print("被梯度提升树投票的：", sum_gbt)
    print("被四个弱分类器选中的：", sum_vote_pos)

def vote_to_predict_single(url,label,colnames):
    import feature_extraction
    df= pd.DataFrame(columns=['URL'], data=[url])
    extend=df['URL'].apply(feature_extraction.extract_url_features)
    df=pd.concat([df,extend],axis=1)
    extend2=df['URL'].apply(feature_extraction.url_trans_token)
    df=pd.concat([df,extend2],axis=1)
    df['label']=label
    df=df[colnames]
    df.to_csv("test_single.csv", index=False, encoding='utf-8')

    X=pd.read_csv("test_single.csv")
    X = X.drop(['URL','label'], axis=1)

    bayes, gbc, ada, rf, dt, lgs = loadModel()
    vote = 0

    predict = bayes.predict(X)
    print('bayes:',predict)
    vote +=predict

    predict = ada.predict(X)
    print('ada:',predict)
    vote += predict*2

    predict = rf.predict(X)
    print('rf:',predict)
    vote += predict*20

    predict = dt.predict(X)
    print('dt:',predict)
    vote += predict*2

    predict = lgs.predict(X)
    print('lgs:',predict)
    vote +=predict

    predict = gbc.predict(X)
    print('gbc:',predict)
    vote += predict*10

    if vote>=7:
        print('bad!')
        label=1
    else:
        print('good!')
        label=0


def vote_to_predict_single_muti(colnames):
    import feature_extraction
    bayes, gbc, ada, rf, dt, lgs = loadModel()

    while True:
        print('请输入URL：')
        url=input()
        url=feature_extraction.wash_URL(url)
        label=0
        df = pd.DataFrame(columns=['URL'], data=[url])
        extend = df['URL'].apply(feature_extraction.extract_url_features)
        df = pd.concat([df, extend], axis=1)
        extend2 = df['URL'].apply(feature_extraction.url_trans_token)
        df = pd.concat([df, extend2], axis=1)
        df['label'] = label
        df = df[colnames]
        df.to_csv("test_single.csv", index=False, encoding='utf-8')
        X = pd.read_csv("test_single.csv")
        X = X.drop(['URL', 'label'], axis=1)

        vote = 0

        predict = bayes.predict(X)
        print('bayes:', predict)
        vote += predict

        predict = ada.predict(X)
        print('ada:', predict)
        vote += predict * 2

        predict = rf.predict(X)
        print('rf:', predict)
        vote += predict * 20

        predict = dt.predict(X)
        print('dt:', predict)
        vote += predict * 2

        predict = lgs.predict(X)
        print('lgs:', predict)
        vote += predict

        predict = gbc.predict(X)
        print('gbc:', predict)
        vote += predict * 10

        if vote >= 7:
            print('bad!')
            label = 1
        else:
            print('good!')
            label = 0


def vote_to_predict_single_X(bayes, gbc, ada, rf, dt, lgs ,X):
    vote = 0

    predict = bayes.predict(X)
    print('bayes:', predict)
    vote += predict

    predict = ada.predict(X)
    print('ada:', predict)
    vote += predict * 2

    predict = rf.predict(X)
    print('rf:', predict)
    vote += predict * 20

    predict = dt.predict(X)
    print('dt:', predict)
    vote += predict * 2

    predict = lgs.predict(X)
    print('lgs:', predict)
    vote += predict

    predict = gbc.predict(X)
    print('gbc:', predict)
    vote += predict * 10

    if vote >= 7:
        print('bad!')
        label = 1
    else:
        print('good!')
        label = 0


def getFishX():
    import feature_extraction
    features=feature_extraction.csv_to_features("data/test_data/fishtank.csv")
    features['label']=1
    labels=features['label']
    features = features.drop(['URL'], axis=1)
    features = features.drop(['label'], axis=1)
    features=features.fillna(0)
    features.to_csv("fishtank_features.csv", index=False, encoding='utf-8')


def testFish():
    fishData=pd.read_csv("fishtank_features.csv")
    fishData['label']=1
    fishData['URL']='fish.com'
    fishData=fishData[colnames]
    labels=fishData['label']
    features=fishData.drop(['label','URL'], axis=1)
    vote_to_predict(features, labels)


def test_single(test_X):
    bayes, gbc, ada, rf, dt, lgs = loadModel()
    colnames=test_X.columns
    print(test_X.head(10))
    for i in range(len(test_X)):
        cur_X=pd.DataFrame(test_X.iloc[i].reshape(1,85),columns=colnames)
        print(cur_X)
        vote_to_predict_single_X(bayes, gbc, ada, rf, dt, lgs,cur_X)
        input()





def machine_learning(url_features, labels):
    urls = pd.DataFrame(data=url_features['URL'], columns=['URL'])
    urls['R'] = 'unknown'

    url_features = url_features.drop(['URL'], axis=1)
    url_features = url_features.drop(['label'], axis=1)

    #  X_train, X_test, y_train, y_test = train_test_split(url_features, labels, test_size=0.3)
    '''
    from sklearn.ensemble import GradientBoostingClassifier
    print('训练中')
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
                                    max_depth = 100, random_state = 0).fit(X_train, y_train)
    clf.score(X_test, y_test)
    print('\n梯度提升树模型的准确度:', clf.score(X_test, y_test))
    predict = clf.predict(X_test)
    predict_positive = 0
    for i in predict:
        predict_positive += i
    print(predict_positive)
    precision = metrics.precision_score(y_test, predict)
    recall = metrics.recall_score(y_test, predict)
    print("precison:", precision)
    print("recall:", recall)

    test_index = X_test.index

    vote = []
    for i in range(len(labels)):
        vote.append(0)
    '''
    # random forest
    import time
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.externals import joblib
    import os
    for k in range(5):
        filename = 'classifier_model\classifier[' + time.strftime('%Y-%m-%d', time.localtime(time.time())) + ']' + str(
            k) + '.model'
        #    filename='classifier_model\classifier[2018-04-03]'+ str(k) + '.model'
        if os.path.exists(filename):
            print('正在读取分类器模型......', filename)
            rf = joblib.load(filename)
        else:
            '''
            fish_filename = "test_data\download_from_fishtank\log" + time.strftime('%Y-%m-%d',
                                                                                   time.localtime(time.time())) + ".txt"
            #  fish_filename = "test_data\download_from_fishtank\log2018-04-03.txt"
            csv_filename = fish_filename + "_features.csv"
            if os.path.exists(fish_filename):
                pass
            else:
                import get_fishtank
                get_fishtank.get_today()
            if os.path.exists(csv_filename):
                pass
            else:
                print('新增fish urls......')
                fish_urls = readTextUrls(fish_filename)
                fish_features = pd.read_csv(csv_filename)
                X_train = X_train.append(fish_features)
                for i in range(len(fish_urls)):
                    fish_features.loc[i, 'label'] = 1
                    fish_features.loc[i, 'URL'] = fish_urls[i]
                ori_url_features = ori_url_features.append(fish_features)
                ori_url_features.to_csv("url_features.csv", index=False, encoding='utf-8')
                print('已更新 url_features.csv')
                new_label = fish_features['label']
                y_train = y_train.append(new_label)
            '''
            print('正在训练RandomForestClassifier模型！')
            rf = RandomForestClassifier(n_estimators=100,
                                        criterion='entropy',
                                        oob_score=True
                                        )
            rf.fit(X_train, y_train)
            #  joblib.dump(rf, filename)
        print(k, '\nrf模型的准确度:', rf.score(X_test, y_test))
        predict = rf.predict(X_test)
        predict_positive = 0
        for i in predict:
            predict_positive += i
        print(predict_positive)
        '''
        vote = list(map(lambda x: x[0] + x[1], zip(predict, vote)))
        precision = metrics.precision_score(y_test, predict)
        recall = metrics.recall_score(y_test, predict)

        print("precison:", precision)
        print("recall:", recall)

    print('\n投票结果：')
    vote_r = []
    for i in range(len(vote)):
        if vote[i] >= 1:
            vote_r.append(1)
        else:
            vote_r.append(0)
    predict_positive = 0
    for i in vote_r:
        predict_positive += i
    print(predict_positive)
    precision = metrics.precision_score(y_test, vote_r)
    recall = metrics.recall_score(y_test, vote_r)
    acc = metrics.accuracy_score(y_test, vote_r)
    print('准确度：', acc)
    print("precison:", precision)
    print("recall:", recall)
 
    for i in range(len(vote_r)):
        ii = test_index[i]
        if labels[ii] == 1:
            if vote_r[i] == 1:
                urls.loc[ii, 'R'] = 'TP'
            else:
                urls.loc[ii, 'R'] = 'FN'
        else:
            if vote_r[i] == 1:
                urls.loc[ii, 'R'] = 'FP'
            else:
                urls.loc[ii, 'R'] = 'TN'
    import time
    filename = "analyse_errors2018-4-03.csv"
    '''
        #   filename = "analyse_errors" + time.strftime('%Y-%m-%d', time.localtime(time.time())) + ".csv"
        # urls.to_csv(filename, index=False)


def use_model_to_predict(url_features, urls):
    vote = []
    for i in range(len(url_features)):
        vote.append(0)

    from sklearn.externals import joblib
    import os
    for k in range(5):
        import time
        # filename = 'classifier_model\classifier['+time.strftime('%Y-%m-%d', time.localtime(time.time()))+']'+ str(k) + '.model'
        filename = 'classifier_model\classifier[2018-04-06]' + str(k) + '.model'
        #  filename = 'classifier_model\classifier' + str(k) + '.model'

        if os.path.exists(filename):
            rf = joblib.load(filename)
        else:
            print(filename, '不存在！')
            return
        predict = rf.predict(url_features)
        predict_positive = 0
        for i in predict:
            predict_positive += i
        vote = list(map(lambda x: x[0] + x[1], zip(predict, vote)))
    print('\n投票结果：')
    vote_r = []
    for i in range(len(vote)):
        if vote[i] >= 1:
            vote_r.append(1)
        else:
            vote_r.append(0)
    predict_positive = 0
    for i in vote_r:
        predict_positive += i
    print("在总共", len(url_features), "条URL中，认为下列url是恶意的：", predict_positive)
    for i in range(len(url_features)):
        if vote_r[i] > 0:
            print(i, urls[i])
    print("在总共", len(url_features), "条URL中，认为下列url不是恶意的：", len(url_features) - predict_positive)
    for i in range(len(url_features)):
        if vote_r[i] == 0:
            print(i, urls[i])


def justtest():
    urls = [
        'artinfo.com/news/story/38796/want-a-shot-at-john-waynes-warhol-see-highlights-from-the-la-sale-of-the-actors-memorabilia-and-art',
        'utoronto.ca/dcb-dbc/dcba/listOfSubjects/m.htm',
        'singingthefaithplus.org.uk/',
        'fergusonknowlesfh.com/obituaries/65692',
        'allhighschools.com/school/lima-central-catholic-high-school/818481',
        'myspace.com/petrzelenka',
        'tastekid.com/like/Les+Colocs',
        'gunhistoryvermont.com/listing.html',
        'trekmovie.com/2011/06/15/exclusive-interview-damon-lindelof-roberto-orci-alex-kurtzman-on-star-trek-sequel',
        'nudestarmale.com/nude/d/david-julian-hirsh-nude.html',
        'streema.com/radios/Mix_93.3_KMXV',
        'sfcm.edu/departments/composition.aspx',
        'spoke.com/info/p6SPIvj/RobertBaldwin',
        'topix.com/cfl/montreal-alouettes',
        'junkss.asia/wired/server/cp.php?m=login']
    for i in range(len(urls)):
        vote_to_predict_single(urls[i], 0,colnames)
        input()

