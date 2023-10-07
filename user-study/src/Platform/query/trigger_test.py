from ctypes import *
import time

# pdll = windll.inpoutx64

# base_code = 0x3010

# def send_trigger(code):
#     time.sleep(0.01)
#     pdll.Out32(base_code, code)
#     time.sleep(0.01)
#     pdll.Out32(base_code, 0)

def send_trigger(code):
    print("send trigger", code)

trigger_dic = {
    'query_id_show' : 'qid',
    'doc_id_show': 'did',
    'click' : 5,
    'abandon': 6,
    'first_screen': 12,
    'end': 7,
    'start': 9,
    'back': 8,
    'resume': 10,
    'history': 11,
    'lucky':13
}
chinese_trigger_dic = {
    '结束搜索':7,
    '下一个结果':6,
    '进入结果页':5,
    '返回': 8,
    '查看历史':11,
    '首屏展示':12,
    '手气不错':13
}

def test():
    t = range(1,200)
    for i in range(1000):
        send_trigger(t[i%200])
        time.sleep(0.5)

if __name__ == '__main__':
    test()