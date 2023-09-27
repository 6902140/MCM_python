import urllib.request

from fake_useragent import UserAgent
import urllib.parse

def get_response(url):
    """
    获取网站返回的Response对象.

    根据输入的url，自动封装headers，返回获取响应对象.

    :param url: 需要访问的url网址
    :type url: str
    :return: 返回urlopen取得的响应对象
    :rtype: urllib.request.Request
    """
    headers={
        "User-Agent":"",
    }
    headers['User-Agent']=UserAgent().edge
    req=urllib.request.Request(url=url,headers=headers)
    return urllib.request.urlopen(req)


def google_by_keyword(word):
    """
    根据传入的关键词进行google搜索
    :param word: 关键词，属性为字符串
    :return: urllib.request.Request 返回的是访问网站返回的消息体
    """
    encoded_word=urllib.parse.quote(word)
    url = "https://www.google.com/search?q={}".format(encoded_word)
    return get_response(url)

# https://tieba.baidu.com/f?kw=%E5%AD%99%E7%AC%91%E5%B7%9D&ie=utf-8&pn=50
def tieba_by_keyword(word,i):
    """
    根据传入的关键词进行 贴吧 搜索
    :param word: 关键词，属性为字符串
    :param i: 页码，0为起始页码
    :return: urllib.request.Request 返回的是访问网站返回的消息体
    """
    search_dictionary={
        "kw":"",
        "pn":"",
    }

    search_dictionary['kw']=word
    search_dictionary['pn']=str(i*50)
    encoded_word = urllib.parse.urlencode(search_dictionary)

    base_url="https://tieba.baidu.com/f?{}".format(encoded_word)

    return get_response(base_url)



