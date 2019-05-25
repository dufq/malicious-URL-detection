import data_split
import use_sklearn
import pandas as pd
import single_test

colnames = ['URL', 'label', 'URL_a', 'URL_b', 'URL_c', 'URL_d', 'URL_e', 'URL_f',
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

if __name__ == '__main__':
    data_train, data_cv, data_test = data_split.read_data()

  #  use_sklearn.multi_machine_learing_models(data_train,data_cv)

    print('---------------cv')
    y_cv = data_cv['label'].apply(lambda x: 0 if x == 'good' else 1)
    x_cv = data_cv.drop(['URL', 'label'], axis=1)
    use_sklearn.vote_to_predict(x_cv, y_cv)

    print('---------------test')
    y_test = data_test['label'].apply(lambda x: 0 if x == 'good' else 1)
    x_test = data_test.drop(['URL', 'label'], axis=1)
    use_sklearn.vote_to_predict(x_test, y_test)

    print('---------------combine')
    x_fish = pd.read_csv("fishtank_features.csv")
    x_fish['URL'] = 'fish.com'
    x_fish['label'] = 1
    y_fish = x_fish['label']
    x_fish = x_fish[colnames]
    x_fish = x_fish.drop(['label', 'URL'], axis=1)
    x_combine = pd.concat([x_test, x_fish], ignore_index=True)
    print(len(x_combine))
    x_combine.to_csv("combine.csv", index=False, encoding='utf-8')
    labels = pd.concat([y_test, y_fish], ignore_index=True)
    print(labels.sum() / len(labels))
    print(labels)
    labels.to_csv("labels.csv", index=False, encoding='utf-8')
    use_sklearn.vote_to_predict(x_combine, labels)
