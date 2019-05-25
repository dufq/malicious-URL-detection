#!/usr/bin/python
# -*- coding: UTF-8 -*-
import pandas as pd  # 数据分析
import whois
from wordcloud import WordCloud
import matplotlib.pyplot as plt


# 返回whois查询结果的结构体
def whois_domain_name(url):
    try:
        w = whois.whois(url)
    except (whois.parser.PywhoisError):  # NOT FOUND
        return "ERROR"
    return w


# 获取顶级域名TLD
def getTLD(str):
    str_part = str.split('.')
    tld0 = str_part[-1].split(':')
    tld = tld0[0]
    return tld


# 判断是否含有数字
def contain_dig(str):
    for x in str:
        if ord(x) >= 48 and ord(x) <= 57:
            return True
    return False


# 判断是否含有英文字母
def contain_letter(str):
    for x in str:
        if (ord(x) >= 97 and ord(x) <= 122) or (ord(x) >= 65 and ord(x) <= 90):
            return True
    return False


# top网站的whois查询结果
# top_domain_whois=pd.read_csv("whois_about_top_sites.csv")

# 中国Top1000+世界Top500网站的url，共1464个
top_sites = []

# top_sites中出现过的TLD
TLDs = []
common_TLD_in_Top = []

# top_sites中出现过的word_token
words = []
top_domain_use_words = []

# ----------------------------------------------------------------------------------------
# 获取中国top网站
china_top_sites = []
'''

china_top_sites.append("buaa.edu.cn")
china_top=pd.read_csv("China_Top_1000.csv")
for url in china_top['URL']:
    china_top_sites.append(url.lower())
'''
# 获取世界top500
top_domains = pd.read_csv("data/World_Top_500.csv")
for url in top_domains['URL']:
    china_top_sites.append(url.lower())

top_sites = list(set(china_top_sites))
top_sites.sort(key=china_top_sites.index)

# ----------------------------------------------------------------------------------------
# 统计top_sites里面的常见TLD
for url in top_sites:
    tld_of_top_site = getTLD(url)
    TLDs.append(tld_of_top_site)
common_TLDs = list(set(TLDs))

num_of_Top_sites = len(top_sites)

for i in range(0, len(common_TLDs)):
    count = TLDs.count(common_TLDs[i])
    if count > 2:
        common_TLD_in_Top.append(common_TLDs[i])

top_words_source = []
top_domain_use_words = []
# 统计top_sites里面的词
index = 0
for url in top_sites:
    url_block = url.split('.')
    url_block.pop(-1)  # 去掉最后一个位置的.com之类的域名后缀
    for block in url_block:
        if (len(block) > 2):  # 长度太短的就不考虑了
            flag = 0
            for fix in common_TLD_in_Top:
                if block == fix:
                    flag += 1
                    break
                else:
                    pass
            if flag == 0:  # 不是后缀那样的词
                top_domain_use_words.append(block)
                top_words_source.append(index)
    index += 1

text = ' '.join(top_domain_use_words)
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
top_domain_use_words = list(set(top_domain_use_words))


def save():
    f_pupular_words = open("pupular.txt", "w")
    for i in range(len(top_domain_use_words)):
        f_pupular_words.write(top_domain_use_words[i])
        f_pupular_words.write("\n")
    f_pupular_words.close()
