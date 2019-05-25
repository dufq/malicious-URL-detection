import re
import urllib
import pandas as pd
import csv
import math
import bad_urls
import numpy as np

all_features = []
url_features = [
    'URL长度',
    '字母比例',
    '数字比例',
    '特殊符号的种类个数',
    '特殊字符个数',
    'URL深度(/)',
    '出现点的次数(.)',
    '存在@符号',
    '顶级域名TLD',
    '出现恶意词的次数',
    '出现流行网站名的次数',
    '出现.php或者.exe的次数',
    '在除了开头位置出现http,www的次数'
]
hostname_features = []
all_features.extend(url_features)


# 判断字符x是否为数字
def is_digit(x):
    if 48 <= ord(x) <= 57:
        return True
    return False


# 判断字符x是否为字母
def is_letter(x):
    if (97 <= ord(x) <= 122) or (65 <= ord(x) <= 90):
        return True
    return False


# 判断是否既不是数字也不是字母的特殊字符
def is_special_ch(x):
    if is_letter(x):
        return False
    elif is_digit(x):
        return False
    else:
        return True


# 判断是否含有数字
def contain_dig(str):
    for x in str:
        if 48 <= ord(x) <= 57:
            return True
    return False


# 判断是否含有英文字母
def contain_letter(str):
    for x in str:
        if (97 <= ord(x) <= 122) or (65 <= ord(x) <= 90):
            return True
    return False


# 字母夹杂数字符号的情况出现了几次
def numAndLetter(token):
    if len(token) < 2:
        return 0
    num = 0
    pre = 1
    cur = 0
    # 0 表示字母 1表示非字母
    if is_letter(token[0]):
        pre = 0
    else:
        pre = 1
    for i in range(1, len(token)):
        if is_letter(token[i]):
            cur = 0
        else:
            cur = 1
        if cur == pre:
            continue
        else:
            num += 1
            pre = cur
    return num


def calculate_entropy(string):
    str_list = list(string)
    n = len(str_list)
    str_list_single = list(set(str_list))
    num_list = []
    for i in str_list_single:
        num_list.append(str_list.count(i))
    entropy = 0
    for j in range(len(num_list)):
        entropy += -1 * (float(num_list[j] / n)) * math.log(float(num_list[j] / n), 2)
    if len(str(entropy).split('.')[-1]) >= 7:
        return ('%.7f' % entropy)
    else:
        return (entropy)


def getLength_std(str_list):
    if len(str_list) == 0:
        return (0, 0)
    str_list_len = 0
    str_list_len_count = []
    str_list_std = 0
    for str in str_list:
        length = len(str)
        str_list_len += length
        str_list_len_count.append(length)
    str_arr = np.array(str_list_len_count)
    str_list_std = str_arr.std()
    return (str_list_len, str_list_std)


def url_trans_token(url):
    decode_url = urllib.parse.unquote(url)
    print(decode_url)
    url = decode_url
    http_p = url.find('http://')
    if http_p != -1 and http_p < url.find('/'):
        url = url[http_p + 7:]

    https_p = url.find('https://')
    if https_p != -1 and https_p < url.find('/'):
        url = url[https_p + 8:]

    www_p = url.find('www')
    if www_p != -1 and www_p < url.find('.'):
        url = url[www_p + 4:]

    other_p = url.find('://')
    if other_p != -1 and other_p < url.find('/'):
        url = url[other_p + 4:]

    url_before = url
    url = url.lower()
    #   print("后来的:",url)


    host_tail = url.find('/')
    if host_tail != -1:
        url_part = url[:host_tail]
    else:
        url_part = url

    pathname_depth = 0
    pathname_longest_token = 0
    pathname_ch_kind = 0
    search_and_n = 0

    hostname_ch_n = 0
    hostname_letter_num = 0
    hostname_dig_num = 0
    hostname_point_n = 0
    hostname_is_ip = 1

    if contain_letter(url_part):
        hostname_is_ip = 0

    # hostname部分是一定存在的
    for i in range(len(url_part)):
        if is_letter(url_part[i]):
            hostname_letter_num += 1
        elif is_digit(url_part[i]):
            hostname_dig_num += 1
        elif url_part[i] == '.':
            hostname_point_n += 1
        else:
            hostname_ch_n += 1
    hostname_dig_ratio = hostname_dig_num / len(url_part)
    hostname_letter_ratio = hostname_letter_num / len(url_part)

    hostname = re.split('[-_/&.()<>^@!#$*=+~:; ]', url_part)
    hostname_entropy = calculate_entropy(url_part)
    while '' in hostname:
        hostname.remove('')

    # pathname、search 、hash部分可能存在
    pathname = []
    search = []
    hash = []

    # path部分存在
    if host_tail != -1 and host_tail != len(url) - 1:
        remains = url[host_tail + 1:]
        pathname_tail = remains.find('?')

        # search部分存在
        if pathname_tail != -1:

            ch_list = []
            pathname_part = remains[:pathname_tail]
            for ch_in_path in pathname_part:
                if ch_in_path == '/':
                    pathname_depth += 1
                elif is_special_ch(ch_in_path):
                    ch_list.append(ch_in_path)
            pathname_ch_kind = len(list(set(ch_list)))

            pathname = re.split('[-_/&.()<>^@!#$*=+~:; ]', pathname_part)
            while '' in pathname:
                pathname.remove('')

            for path_token in pathname:
                if len(path_token) > pathname_longest_token:
                    pathname_longest_token = len(path_token)

            search_and_hash = remains[pathname_tail + 1:]
            search_tail = search_and_hash.find('#')
            if search_tail != -1:
                search = search_and_hash[:search_tail]
            else:
                search = search_and_hash
            search_and_n = search.count('&')
            search = re.split('[-_/&.()<>^@!#$*=+~:; ]', search)
            while '' in search:
                search.remove('')

            # hash部分存在
            if search_tail != -1:
                hash.append(search_and_hash[search_tail + 1:])

            # hash部分不存在
            else:
                hash = []

        # search部分不存在
        else:
            ch_list = []
            pathname_part = remains
            for ch_in_path in pathname_part:
                if ch_in_path == '/':
                    pathname_depth += 1
                elif is_special_ch(ch_in_path):
                    ch_list.append(ch_in_path)
            pathname_ch_kind = len(list(set(ch_list)))

            pathname = re.split('[-_/&.()<>^@!#$*=+~:; ]', remains)

            while '' in pathname:
                pathname.remove('')

            search = []
            hash = []

    # path部分不存在
    else:
        pathname = []
        search = []
        hash = []
    print(hostname,pathname,search,hash)
    hostname_len, hostname_std = getLength_std(hostname)

    pathname_len, pathname_std = getLength_std(pathname)

    search_len, search_std = getLength_std(search)

    return pd.Series(
        {
            'hostname_a':url_part.count('a'),
            'hostname_b': url_part.count('b'),
            'hostname_c': url_part.count('c'),
            'hostname_d': url_part.count('d'),
            'hostname_e': url_part.count('e'),
            'hostname_f': url_part.count('f'),
            'hostname_g': url_part.count('g'),
            'hostname_h': url_part.count('h'),
            'hostname_i': url_part.count('i'),
            'hostname_j': url_part.count('j'),
            'hostname_k': url_part.count('k'),
            'hostname_l': url_part.count('l'),
            'hostname_m': url_part.count('m'),
            'hostname_n': url_part.count('n'),
            'hostname_o': url_part.count('o'),
            'hostname_p': url_part.count('p'),
            'hostname_q': url_part.count('q'),
            'hostname_r': url_part.count('r'),
            'hostname_s': url_part.count('s'),
            'hostname_t': url_part.count('t'),
            'hostname_u': url_part.count('u'),
            'hostname_v': url_part.count('v'),
            'hostname_w': url_part.count('w'),
            'hostname_x': url_part.count('x'),
            'hostname_y': url_part.count('y'),
            'hostname_z': url_part.count('z'),
            'hostname_token_n': len(hostname),
            'hostname_len': hostname_len,
            'hostname_ch_n': hostname_ch_n,
            'hostname_letter_ratio': hostname_letter_ratio,
            'hostname_dig_ratio': hostname_dig_ratio,
            'hostname_entropy': hostname_entropy,
            'hostname_point_n': hostname_point_n,
            'hostname_is_ip': hostname_is_ip,
            'hostname_std': hostname_std,
            'pathname_token_n': len(pathname),
            'pathname_len': pathname_len,
            'pathname_depth': pathname_depth,
            'pathname_longest_token': pathname_longest_token,
            'pathname_ch_kind': pathname_ch_kind,
            'pathname_std': pathname_std,
            'search_token_n': len(search),
            'search_len': search_len,
            'search_std': search_std,
            'search_and_n': search_and_n,
            'hash_token_n': len(hash)
        }
    )

def wash_URL(url):
    url = url.lower()
    url_len = len(url)

    http_p = url.find('http://')
    if http_p != -1 and http_p < url.find('/'):
        url = url[http_p + 7:]

    https_p = url.find('https://')
    if https_p != -1 and https_p < url.find('/'):
        url = url[https_p + 8:]

    www_p = url.find('www')
    if www_p != -1 and www_p < url.find('.'):
        url = url[www_p + 4:]

    other_p = url.find('://')
    if other_p != -1 and other_p < url.find('/'):
        url = url[other_p + 4:]
    if url.find('/')==-1:
        return url+'/'
    else:
        return url


def extract_url_features(url):
    url = url.lower()
    url_len = len(url)

    http_p = url.find('http://')
    if http_p != -1 and http_p < url.find('/'):
        url = url[http_p + 7:]

    https_p = url.find('https://')
    if https_p != -1 and https_p < url.find('/'):
        url = url[https_p + 8:]

    www_p = url.find('www')
    if www_p != -1 and www_p < url.find('.'):
        url = url[www_p + 4:]

    other_p = url.find('://')
    if other_p != -1 and other_p < url.find('/'):
        url = url[other_p + 4:]

        #  print(url)

    url_letter_ratio = 0
    url_dig_ratio = 0
    url_ch_kind_n = 0
    url_ch_n = 0
    url_depth = 0
    url_point_n = 0
    url_contain_at = 0

    ch_list = []
    letter_num = 0
    dig_num = 0
    for i in range(len(url)):
        if is_letter(url[i]):
            letter_num += 1
        elif is_digit(url[i]):
            dig_num += 1
        elif url[i] == '/':
            url_depth += 1
        elif url[i] == '.':
            url_point_n += 1
        elif url[i] == '@':
            url_contain_at += 1
        else:
            ch_list.append(url[i])

    url_letter_ratio = letter_num / url_len
    url_dig_ratio = dig_num / url_len
    url_ch_n = len(ch_list)
    url_ch_kind_n = len(list(set(ch_list)))

    url_port = 80  # 默认端口号是80
    parts = url.split('/')
    first_part = parts[0]
    hostname = first_part.split('.')

    if contain_dig(hostname[-1]) == False:
        tld = hostname[-1].split(':')
        url_TLD_temp = tld[0]
        if len(tld) == 2 and type(tld[1]) == type(url_port):
            url_port = tld[1]
        common_tlds = bad_urls.common_TLDs
        url_TLD = 500
        for i in range(len(common_tlds)):
            if common_tlds[i] == url_TLD_temp:
                url_TLD = i
                break
    else:
        url_TLD = 1000

    url_badword_n = 0

    badwords = bad_urls.get_txt('data/badwords.txt')
    for i in range(len(badwords)):
        if url.find(badwords[i]) != -1:
            url_badword_n += 1

    url_popular_n = 0
    popular_web_words = bad_urls.get_txt('data/popular_web.txt')
    for i in range(len(popular_web_words)):
        if url.find(popular_web_words[i]) != -1:
            url_popular_n += 1

    url_exe_n = 0
    if url.find('.exe') != -1:
        url_exe_n += 1
    if url.find('.php') != -1:
        url_exe_n += 1

    url_http_n = 0
    if url.find('http') != -1:
        url_http_n += 1
    if url.find('www') != -1:
        url_http_n += 1

    return pd.Series(
        {
            'URL_len': url_len,
            'letter_ratio': url_letter_ratio,
            'dig_ratio': url_dig_ratio,
            'special_ch_kind': url_ch_kind_n,
            'special_ch': url_ch_n,
            'URL_depth': url_depth,
            'URL_point': url_point_n,
            'at_flag': url_contain_at,
            'TLD_id': url_TLD,
            'badword_n': url_badword_n,
            'popular_n': url_popular_n,
            'exe_flag': url_exe_n,
            'http_flag': url_http_n,
            'URL_a': url.count('a'),
            'URL_b': url.count('b'),
            'URL_c': url.count('c'),
            'URL_d': url.count('d'),
            'URL_e': url.count('e'),
            'URL_f': url.count('f'),
            'URL_g': url.count('g'),
            'URL_h': url.count('h'),
            'URL_i': url.count('i'),
            'URL_j': url.count('j'),
            'URL_k': url.count('k'),
            'URL_l': url.count('l'),
            'URL_m': url.count('m'),
            'URL_n': url.count('n'),
            'URL_o': url.count('o'),
            'URL_p': url.count('p'),
            'URL_q': url.count('q'),
            'URL_r': url.count('r'),
            'URL_s': url.count('s'),
            'URL_t': url.count('t'),
            'URL_u': url.count('u'),
            'URL_v': url.count('v'),
            'URL_w': url.count('w'),
            'URL_x': url.count('x'),
            'URL_y': url.count('y'),
            'URL_z': url.count('z'),
        })

def pandas_get_features(df):
    extend=df['URL'].apply(extract_url_features)
    df=pd.concat([df,extend],axis=1)
    extend2=df['URL'].apply(url_trans_token)
    df=pd.concat([df,extend2],axis=1)
    return df


def csv_to_features(filename):
    data=pd.read_csv(filename)
    features=pandas_get_features(data)
    return features


