import whois


# 返回whois查询结果的结构体
def get_whois(url):
    try:
        w = whois.whois(url)
    except whois.parser.PywhoisError:  # NOT FOUND
        return "ERROR"
    return w
