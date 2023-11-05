'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2023-11-02 08:58:55
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2023-11-03 10:58:49
FilePath: \newrgzn\dataset_download.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import requests
import time
import os
import json.decoder  # 导入 JSONDecodeError 异常
from tqdm import tqdm
def download(str_zh):
    # ... 其他代码 ...
    headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0'
    }
    keyword = str_zh# 关键字
    max_page =30
    folder_path = './download/'+keyword
    os.makedirs(folder_path)
    i=1 # 记录图片数
    for page in tqdm(range(1, max_page + 1), desc="Processing Pages"):
        page = page * 30
        # 网址
        url = 'https://image.baidu.com/search/acjson?tn=resultjson_com&ipn=rj&ct=201326592&is=&fp=result&queryWord=' \
            + keyword + '&cl=2&lm=-&ie=utf-8&oe=utf-8&adpicid=&st=-1&z=&ic=0&hd=&latest=&copyright=&word=' \
            + keyword + '&s=&se=&tab=&width=&height=&face=0&istype=2&qc=&nc=1&fr=&expermode=&force=&cg=wallpaper&pn=' \
            + str(page) + '&rn=30&gsm=1e&1596899786625='
        #print(url)
        # 请求响应
        response = requests.get(url=url, headers=headers)
        try:
            # 得到相应的json数据
            json_data = response.json()
            if json_data.get('data'):
                for item in json_data.get('data')[:30]:
                    # 图片地址
                    img_url = item.get('thumbURL')
                    # 获取图片
                    image = requests.get(url=img_url)
                    # 下载图片

                    with open('./download/' + keyword + '/'+str(i)+'.jpg' , 'wb') as f:
                        f.write(image.content)  # 图片二进制数据
                    i += 1
        except json.decoder.JSONDecodeError as e:
            print(f"JSON解码错误: {e}")
            # 这里可以添加你的错误处理逻辑，比如跳过这个页面，继续下一个页面的爬取
            continue

    print('End!')
if __name__=="__main__":
    download("狗")
    download("面包")
  