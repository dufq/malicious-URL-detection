import pandas as pd
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt

common_TLDs = ['co', 'info', 'ru', 'pl', 'ca', 'us', 'wang', 'fr',
               'org', 'it', 'au', 'jp', 'im', 'com', 'gov', 'io',
               'eu', 'me', 'nl', 'net', 'se', 'tw', 'mil', 'ch',
               'ly', 'edu', 'gl', 'cn', 'dzcom', 'to', 'cz', 'cc',
               'br', 'uk', 'be', 'es', 'no', 'in', 'tv', 'hk',
               'de', 'int', 'la', 'pro']


def draw_cloud():
    bad_domains = pd.read_csv("data/bad_urls.csv")
    bad_domains = bad_domains.drop_duplicates()
    bad_word_bags = []
    for url in bad_domains['URL']:
        str_part = re.split('[-_/&.()<>^@!#$*=+~:;? ]', url)
        while '' in str_part:
            str_part.remove('')
        for p in str_part:
            tld_flag = False
            for tld in common_TLDs:
                if p == tld:
                    tld_flag = True
                    break
            if not tld_flag:
                if 2 < len(p) < 12:
                    bad_word_bags.append(p)
    text = ' '.join(bad_word_bags)
    wordcloud = WordCloud(background_color='white',
                          max_words=300,
                          random_state=30,
                          max_font_size=20,
                          scale=15).generate(text)
    # 显示词云图片
    plt.imshow(wordcloud)
    plt.savefig("badword_cloud.png")
    plt.axis('off')
    plt.show()


# 统计恶意url里经常出现的词
def count_bad_words():
    bad_domains = pd.read_csv("bad_urls.csv")
    bad_domains = bad_domains.drop_duplicates()
    bad_word_bags = []
    for url in bad_domains['URL']:
        str_part = re.split('[-_/&.()<>^@!#$*=+~:;? ]', url)
        while '' in str_part:
            str_part.remove('')
        for p in str_part:
            tld_flag = False
            for tld in common_TLDs:
                if p == tld:
                    tld_flag = True
                    break
            if not tld_flag:
                bad_word_bags.append(p)

    bad_words = pd.DataFrame(data=bad_word_bags, columns=['word'])
    words_bag = bad_words['word'].value_counts()

    words_bag_index = words_bag.index
    bad_words_in_url = []
    for i in range(len(words_bag)):
        if words_bag[i] > 10:
            temp_word = words_bag_index[i]
            if 2 < len(temp_word) < 20:
                bad_words_in_url.append(temp_word)
    bad_words_in_url = list(set(bad_words_in_url))

    f_bad_words = open("data/bad_words.txt", "w")
    for i in range(len(bad_words_in_url)):
        f_bad_words.write(bad_words_in_url[i])
        f_bad_words.write("\n")
    f_bad_words.close()


def get_bad_words():
    f_bad_words = open("data/badwords.txt")
    bad_words = []
    while True:
        lines = f_bad_words.readline().strip("\n")
        if not lines:
            break
            pass
        bad_words.append(lines)
    f_bad_words.close()
    return bad_words


def test_draw():
    badwords = open('data/badwords.txt', 'r').read()
    print(badwords)
    wordcloud = WordCloud(background_color='white', scale=1.5).generate(badwords)
    # 显示词云图片

    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()


def get_txt(filename):
    f_bad_words = open(filename)
    bad_words = []
    while True:
        lines = f_bad_words.readline().strip("\n")
        if not lines:
            break
            pass
        bad_words.append(lines)
    f_bad_words.close()
    return bad_words


def updateBadWords(new_bad_urls):
    old_bad_words = get_bad_words()
    print('更新前的bad words：')
    print(old_bad_words)
    print(len(old_bad_words))
    bad_word_bags = []

    for url in new_bad_urls:
        url = url.lower()
        str_part = re.split('[-_/&.()<>^@!#$*=+~:;? ]', url)
        while '' in str_part:
            str_part.remove('')
        for p in str_part:
            tld_flag = False
            for tld in common_TLDs:
                if p == tld:
                    tld_flag = True
                    break
            if not tld_flag:
                bad_word_bags.append(p)

    bad_words = pd.DataFrame(data=bad_word_bags, columns=['word'])
    words_bag = bad_words['word'].value_counts()

    words_bag_index = words_bag.index
    bad_words_in_url = []
    for i in range(len(words_bag)):
        if words_bag[i] > 3:
            temp_word = words_bag_index[i]
            if 2 < len(temp_word) < 20:
                bad_words_in_url.append(temp_word)
    bad_words_in_url = list(set(bad_words_in_url))
    print('新增的：')
    print(bad_words_in_url)
    print(len(bad_words_in_url))

    bad_words_in_url.extend(old_bad_words)
    bad_words_in_url = list(set(bad_words_in_url))
    bad_words_in_url.remove('www')
    bad_words_in_url.remove('https')
    bad_words_in_url.remove('http')
    print('更新后的bad words：')
    print(bad_words_in_url)
    print(len(bad_words_in_url))
    input()
    f_bad_words = open("data/badwords.txt", "w")
    for i in range(len(bad_words_in_url)):
        f_bad_words.write(bad_words_in_url[i])
        f_bad_words.write("\n")
    f_bad_words.close()
