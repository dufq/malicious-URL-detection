import requests
import time


# 调用virustotal提供的API对url进行检测
# #返回值是在几个数据库中发现是异常，如返回0则代表没有异常,返回-1代表没有在数据库中找到匹配
def url_check(my_url):
    url = "https://www.virustotal.com/vtapi/v2/url/report"
    params = {"apikey": "a2c4c89637e57dc27bdb3048989da16c530c2dfffc4783c62fa95ea936e19d80", "resource": my_url}
    response = requests.get(url, params=params)
    time.sleep(15)
    if response.status_code == 200:
        json = response.json()
        #   print(json)  显示完整的json信息
        if json['response_code'] == 1:
            return json['positives']
        else:
            return -1
    else:
        return -1
