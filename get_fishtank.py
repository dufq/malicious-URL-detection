import requests
from bs4 import BeautifulSoup
import time


def getFishUrlIds():
    print('正在获取fish url的id......')
    fish_url_ids = []
    for i in range(10):
        import time
        time.sleep(5)
        try:
            session = requests.Session()
            if i % 2 == 0:
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:56.0) "
                                  "Gecko/20100101 Firefox/56.0",
                    "Accept": "*/*"}
            else:
                headers = {
                    'User-Agent': 'User-Agent:Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_3) '
                                  'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36'}
            page_url = 'https://www.phishtank.com/phish_search.php?page=' + str(i) + '&valid=y&Search=Search'
            print(page_url)
            req = session.get(page_url, headers=headers)
            bsObj = BeautifulSoup(req.text, 'html.parser')
            IDs = bsObj.findAll("td", {"class": "value"})
            for id in IDs:
                id_str = str(id)
                s_pos = id_str.find('phish_id=')
                if s_pos != -1:
                    id_str = id_str[s_pos + 9:]
                    e_pos = id_str.find('"')
                    id_str = id_str[:e_pos]
                    if len(id_str) > 1:
                        fish_url_ids.append(id_str)
                        print(id_str)
            session.keep_alive = False
        except:
            session.keep_alive = False
            print('获取page_url时发生了异常！')
    return fish_url_ids


def getFishUrls(fish_url_ids):
    print('正在获取fish url......')
    fish_urls = []
    id = 0
    for i in fish_url_ids:
        id += 1
        try:
            session = requests.Session()
            if id % 2 == 0:
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:56.0) Gecko/20100101 Firefox/56.0",
                    "Accept": "*/*"}
            else:
                headers = {
                    'User-Agent': 'User-Agent:Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_3) '
                                  'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36'}
            url = 'https://www.phishtank.com/phish_detail.php?phish_id=' + str(i)
            req = session.get(url, headers=headers)
            bsObj = BeautifulSoup(req.text, 'html.parser')
            fish_part = bsObj.find("span", {"style": "word-wrap:break-word;"})
            fish_url = str(fish_part)
            start_pos = fish_url.find('<b>')
            fish_url = fish_url[start_pos + 3:]
            end_pos = fish_url.find('</b>')
            fish_url = fish_url[:end_pos]
            print(fish_url)
            fish_urls.append(fish_url)
            session.keep_alive = False
        except:
            session.keep_alive = False
            print('获取url时发生了异常！')
    return fish_urls


def get_today():
    start = time.clock()
    ids = getFishUrlIds()
    urls = getFishUrls(ids)
    filename = "test_data\download_from_fishtank\log" + time.strftime('%Y-%m-%d', time.localtime(time.time())) + ".txt"
    f = open(filename, "w")
    for i in range(len(urls)):
        print(i, urls[i])
        f.write(urls[i])
        f.write('\n')

    elapsed = (time.clock() - start)
    print("[fishtank] Time used:", elapsed)
