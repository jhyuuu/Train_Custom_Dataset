#!/usr/bin/env python
# coding: utf-8

# # 图像采集
# 
# 本代码仅供编程科普教学、科研学术等非盈利用途。
# 
# 请遵守国家相关法律法规和互联网数据使用规定。
# 
# 请勿用于商业用途，请勿高频长时间访问服务器，请勿用于网络攻击，请勿恶意窃取信息，请勿用于作恶。
# 
# 任何后果与作者无关。

# ## 导入工具包

# In[5]:


import os

import time

import requests

import urllib3
urllib3.disable_warnings()

# 进度条库
from tqdm import tqdm

import os


# ## HTTP请求参数

# In[6]:


cookies = {
'BDqhfp': '%E7%8B%97%E7%8B%97%26%26NaN-1undefined%26%2618880%26%2621',
'BIDUPSID': '06338E0BE23C6ADB52165ACEB972355B',
'PSTM': '1646905430',
'BAIDUID': '104BD58A7C408DABABCAC9E0A1B184B4:FG=1',
'BDORZ': 'B490B5EBF6F3CD402E515D22BCDA1598',
'H_PS_PSSID': '35836_35105_31254_36024_36005_34584_36142_36120_36032_35993_35984_35319_26350_35723_22160_36061',
'BDSFRCVID': '8--OJexroG0xMovDbuOS5T78igKKHJQTDYLtOwXPsp3LGJLVgaSTEG0PtjcEHMA-2ZlgogKK02OTH6KF_2uxOjjg8UtVJeC6EG0Ptf8g0M5',
'H_BDCLCKID_SF': 'tJPqoKtbtDI3fP36qR3KhPt8Kpby2D62aKDs2nopBhcqEIL4QTQM5p5yQ2c7LUvtynT2KJnz3Po8MUbSj4QoDjFjXJ7RJRJbK6vwKJ5s5h5nhMJSb67JDMP0-4F8exry523ioIovQpn0MhQ3DRoWXPIqbN7P-p5Z5mAqKl0MLPbtbb0xXj_0D6bBjHujtT_s2TTKLPK8fCnBDP59MDTjhPrMypomWMT-0bFH_-5L-l5js56SbU5hW5LSQxQ3QhLDQNn7_JjOX-0bVIj6Wl_-etP3yarQhxQxtNRdXInjtpvhHR38MpbobUPUDa59LUvEJgcdot5yBbc8eIna5hjkbfJBQttjQn3hfIkj0DKLtD8bMC-RDjt35n-Wqxobbtof-KOhLTrJaDkWsx7Oy4oTj6DD5lrG0P6RHmb8ht59JROPSU7mhqb_3MvB-fnEbf7r-2TP_R6GBPQtqMbIQft20-DIeMtjBMJaJRCqWR7jWhk2hl72ybCMQlRX5q79atTMfNTJ-qcH0KQpsIJM5-DWbT8EjHCet5DJJn4j_Dv5b-0aKRcY-tT5M-Lf5eT22-usy6Qd2hcH0KLKDh6gb4PhQKuZ5qutLTb4QTbqWKJcKfb1MRjvMPnF-tKZDb-JXtr92nuDal5TtUthSDnTDMRhXfIL04nyKMnitnr9-pnLJpQrh459XP68bTkA5bjZKxtq3mkjbPbDfn02eCKuj6tWj6j0DNRabK6aKC5bL6rJabC3b5CzXU6q2bDeQN3OW4Rq3Irt2M8aQI0WjJ3oyU7k0q0vWtvJWbbvLT7johRTWqR4enjb3MonDh83Mxb4BUrCHRrzWn3O5hvvhKoO3MA-yUKmDloOW-TB5bbPLUQF5l8-sq0x0bOte-bQXH_E5bj2qRCqVIKa3f',
'BDSFRCVID_BFESS': '8--OJexroG0xMovDbuOS5T78igKKHJQTDYLtOwXPsp3LGJLVgaSTEG0PtjcEHMA-2ZlgogKK02OTH6KF_2uxOjjg8UtVJeC6EG0Ptf8g0M5',
'H_BDCLCKID_SF_BFESS': 'tJPqoKtbtDI3fP36qR3KhPt8Kpby2D62aKDs2nopBhcqEIL4QTQM5p5yQ2c7LUvtynT2KJnz3Po8MUbSj4QoDjFjXJ7RJRJbK6vwKJ5s5h5nhMJSb67JDMP0-4F8exry523ioIovQpn0MhQ3DRoWXPIqbN7P-p5Z5mAqKl0MLPbtbb0xXj_0D6bBjHujtT_s2TTKLPK8fCnBDP59MDTjhPrMypomWMT-0bFH_-5L-l5js56SbU5hW5LSQxQ3QhLDQNn7_JjOX-0bVIj6Wl_-etP3yarQhxQxtNRdXInjtpvhHR38MpbobUPUDa59LUvEJgcdot5yBbc8eIna5hjkbfJBQttjQn3hfIkj0DKLtD8bMC-RDjt35n-Wqxobbtof-KOhLTrJaDkWsx7Oy4oTj6DD5lrG0P6RHmb8ht59JROPSU7mhqb_3MvB-fnEbf7r-2TP_R6GBPQtqMbIQft20-DIeMtjBMJaJRCqWR7jWhk2hl72ybCMQlRX5q79atTMfNTJ-qcH0KQpsIJM5-DWbT8EjHCet5DJJn4j_Dv5b-0aKRcY-tT5M-Lf5eT22-usy6Qd2hcH0KLKDh6gb4PhQKuZ5qutLTb4QTbqWKJcKfb1MRjvMPnF-tKZDb-JXtr92nuDal5TtUthSDnTDMRhXfIL04nyKMnitnr9-pnLJpQrh459XP68bTkA5bjZKxtq3mkjbPbDfn02eCKuj6tWj6j0DNRabK6aKC5bL6rJabC3b5CzXU6q2bDeQN3OW4Rq3Irt2M8aQI0WjJ3oyU7k0q0vWtvJWbbvLT7johRTWqR4enjb3MonDh83Mxb4BUrCHRrzWn3O5hvvhKoO3MA-yUKmDloOW-TB5bbPLUQF5l8-sq0x0bOte-bQXH_E5bj2qRCqVIKa3f',
'indexPageSugList': '%5B%22%E7%8B%97%E7%8B%97%22%5D',
'cleanHistoryStatus': '0',
'BAIDUID_BFESS': '104BD58A7C408DABABCAC9E0A1B184B4:FG=1',
'BDRCVFR[dG2JNJb_ajR]': 'mk3SLVN4HKm',
'BDRCVFR[-pGxjrCMryR]': 'mk3SLVN4HKm',
'ab_sr': '1.0.1_Y2YxZDkwMWZkMmY2MzA4MGU0OTNhMzVlNTcwMmM2MWE4YWU4OTc1ZjZmZDM2N2RjYmVkMzFiY2NjNWM4Nzk4NzBlZTliYWU0ZTAyODkzNDA3YzNiMTVjMTllMzQ0MGJlZjAwYzk5MDdjNWM0MzJmMDdhOWNhYTZhMjIwODc5MDMxN2QyMmE1YTFmN2QyY2M1M2VmZDkzMjMyOThiYmNhZA==',
'delPer': '0',
'PSINO': '2',
'BA_HECTOR': '8h24a024042g05alup1h3g0aq0q',
}

headers = {
'Connection': 'keep-alive',
'sec-ch-ua': '" Not;A Brand";v="99", "Google Chrome";v="97", "Chromium";v="97"',
'Accept': 'text/plain, */*; q=0.01',
'X-Requested-With': 'XMLHttpRequest',
'sec-ch-ua-mobile': '?0',
'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/97.0.4692.99 Safari/537.36',
'sec-ch-ua-platform': '"macOS"',
'Sec-Fetch-Site': 'same-origin',
'Sec-Fetch-Mode': 'cors',
'Sec-Fetch-Dest': 'empty',
'Referer': 'https://image.baidu.com/search/index?tn=baiduimage&ipn=r&ct=201326592&cl=2&lm=-1&st=-1&fm=result&fr=&sf=1&fmq=1647837998851_R&pv=&ic=&nc=1&z=&hd=&latest=&copyright=&se=1&showtab=0&fb=0&width=&height=&face=0&istype=2&dyTabStr=MCwzLDIsNiwxLDUsNCw4LDcsOQ%3D%3D&ie=utf-8&sid=&word=%E7%8B%97%E7%8B%97',
'Accept-Language': 'zh-CN,zh;q=0.9',
}


# ## 爬取一类

# ### 指定关键词

# In[12]:


# 关键词
keyword = '西瓜'

# 拟爬取图像个数
DOWNLOAD_NUM = 5


# ### 创建文件夹

# In[13]:


if not os.path.exists('dataset'):
    os.makedirs('dataset')
    print('新建 dataset 文件夹')

if os.path.exists('dataset/'+keyword):
    print('文件夹 dataset/{} 已存在，之后直接将爬取到的图片保存至该文件夹中'.format(keyword))
else:
    os.makedirs('dataset/{}'.format(keyword))
    print('新建文件夹：dataset/{}'.format(keyword))


# ### 爬取并保存图像文件至本地

# In[14]:


count = 1

# 爬取第几张
num = 1

# 是否继续爬取
FLAG = True

while FLAG:
    
    page = 30 * count
    
    params = (
        ('tn', 'resultjson_com'),
        ('logid', '12508239107856075440'),
        ('ipn', 'rj'),
        ('ct', '201326592'),
        ('is', ''),
        ('fp', 'result'),
        ('fr', ''),
        ('word', f'{keyword}'),
        ('queryWord', f'{keyword}'),
        ('cl', '2'),
        ('lm', '-1'),
        ('ie', 'utf-8'),
        ('oe', 'utf-8'),
        ('adpicid', ''),
        ('st', '-1'),
        ('z', ''),
        ('ic', ''),
        ('hd', ''),
        ('latest', ''),
        ('copyright', ''),
        ('s', ''),
        ('se', ''),
        ('tab', ''),
        ('width', ''),
        ('height', ''),
        ('face', '0'),
        ('istype', '2'),
        ('qc', ''),
        ('nc', '1'),
        ('expermode', ''),
        ('nojc', ''),
        ('isAsync', ''),
        ('pn', f'{page}'),
        ('rn', '30'),
        ('gsm', '1e'),
        ('1647838001666', ''),
    )

    response = requests.get('https://image.baidu.com/search/acjson', headers=headers, params=params, cookies=cookies)
    if response.status_code == 200:
        try:
            json_data = response.json().get("data")
            
            if json_data:
                for x in json_data:
                    type = x.get("type")
                    if type not in ["gif"]:
                        img = x.get("thumbURL")
                        fromPageTitleEnc = x.get("fromPageTitleEnc")
                        try:
                            resp = requests.get(url=img, verify=False)
                            time.sleep(1)
                            print(f"链接 {img}")
                            # 保存文件名
                            # file_save_path = f'dataset/{keyword}/{num}-{fromPageTitleEnc}.{type}'
                            file_save_path = f'dataset/{keyword}/{num}.{type}'
                            with open(file_save_path, 'wb') as f:
                                f.write(resp.content) # 保存图像文件到本地
                                f.flush()
                                print('第 {} 张图像 {} 爬取完成'.format(num, fromPageTitleEnc))
                                num += 1


                            # 爬取数量达到要求
                            if num > DOWNLOAD_NUM:
                                FLAG = False
                                print('{} 张图像爬取完毕'.format(num))
                                break

                        except Exception:
                            pass
        except:
            pass
    else:
        break

    count += 1


# ## 封装函数

# In[10]:


def craw_single_class(keyword, DOWNLOAD_NUM = 200):
    if os.path.exists('dataset/'+keyword):
        print('文件夹 dataset/{} 已存在，之后直接将爬取到的图片保存至该文件夹中'.format(keyword))
    else:
        os.makedirs('dataset/{}'.format(keyword))
        print('新建文件夹：dataset/{}'.format(keyword))
    count = 1
    
    with tqdm(total=DOWNLOAD_NUM, position=0, leave=True) as pbar:
        
        # 爬取第几张
        num = 0

        # 是否继续爬取
        FLAG = True

        while FLAG:

            page = 30 * count

            params = (
                ('tn', 'resultjson_com'),
                ('logid', '12508239107856075440'),
                ('ipn', 'rj'),
                ('ct', '201326592'),
                ('is', ''),
                ('fp', 'result'),
                ('fr', ''),
                ('word', f'{keyword}'),
                ('queryWord', f'{keyword}'),
                ('cl', '2'),
                ('lm', '-1'),
                ('ie', 'utf-8'),
                ('oe', 'utf-8'),
                ('adpicid', ''),
                ('st', '-1'),
                ('z', ''),
                ('ic', ''),
                ('hd', ''),
                ('latest', ''),
                ('copyright', ''),
                ('s', ''),
                ('se', ''),
                ('tab', ''),
                ('width', ''),
                ('height', ''),
                ('face', '0'),
                ('istype', '2'),
                ('qc', ''),
                ('nc', '1'),
                ('expermode', ''),
                ('nojc', ''),
                ('isAsync', ''),
                ('pn', f'{page}'),
                ('rn', '30'),
                ('gsm', '1e'),
                ('1647838001666', ''),
            )

            response = requests.get('https://image.baidu.com/search/acjson', headers=headers, params=params, cookies=cookies)
            if response.status_code == 200:
                try:
                    json_data = response.json().get("data")

                    if json_data:
                        for x in json_data:
                            type = x.get("type")
                            if type not in ["gif"]:
                                img = x.get("thumbURL")
                                fromPageTitleEnc = x.get("fromPageTitleEnc")
                                try:
                                    resp = requests.get(url=img, verify=False)
                                    time.sleep(1)
                                    # print(f"链接 {img}")

                                    # 保存文件名
                                    # file_save_path = f'dataset/{keyword}/{num}-{fromPageTitleEnc}.{type}'
                                    file_save_path = f'dataset/{keyword}/{num}.{type}'
                                    with open(file_save_path, 'wb') as f:                                    
                                        f.write(resp.content)
                                        f.flush()
                                        # print('第 {} 张图像 {} 爬取完成'.format(num, fromPageTitleEnc))
                                        num += 1
                                        pbar.update(1) # 进度条更新

                                    # 爬取数量达到要求
                                    if num > DOWNLOAD_NUM:
                                        FLAG = False
                                        print('{} 张图像爬取完毕'.format(num))
                                        break

                                except Exception:
                                    pass
                except:
                    pass
            else:
                break

            count += 1


# ### 爬取一类

# In[21]:


craw_single_class('柚子', DOWNLOAD_NUM = 200)


# ### 爬取多类

# In[9]:


class_list = ['黄瓜','南瓜','冬瓜','木瓜','苦瓜','丝瓜','窝瓜','甜瓜','香瓜','白兰瓜','黄金瓜','西葫芦','人参果','羊角蜜','佛手瓜','伊丽莎白瓜']


# In[10]:


for each in class_list:
    craw_single_class(each, DOWNLOAD_NUM = 200)


# ## 一些参考类别关键词

# In[ ]:


# 苹果
'苹果 水果','青苹果'


# In[ ]:


# 常见水果
'菠萝','榴莲','椰子','香蕉','梨','芒果'


# In[ ]:


# 西红柿类
'圣女果','西红柿'


# In[ ]:


# 桔橙类
'砂糖橘','脐橙','金桔','柠檬','西柚','血橙','芦柑','青柠','沃柑','粑粑柑','橘子','柚子'


# In[ ]:


# 桃类
'猕猴桃','油桃','水蜜桃','蟠桃','杨桃','黄桃'


# In[ ]:


# 樱桃类
'樱桃','智利 车厘子'


# In[ ]:


# 火龙果类
'白心火龙果','红心火龙果'


# In[ ]:


# 葡萄类
'马奶提子','红提'


# In[ ]:


# 萝卜类
'胡萝卜','白萝卜'


# In[ ]:


# 莓类
'桑葚','蔓越莓','蓝莓','草莓','树莓','菠萝莓''黑莓 水果'


# In[ ]:


# 其它类
'山楂','桂圆','杨梅','西梅','沙果','枣','荔枝','腰果','无花果','沙棘','羊奶果','百香果','黄金百香果','甘蔗','菠萝蜜','酸角','蛇皮果','人参果','红芭乐','白芭乐','牛油果','莲雾','山竹','杏','李子','柿子','枇杷','香橼','毛丹','石榴'


# In[ ]:




