import os
import re
import urllib

import requests


def get_context(url):
    web_context = requests.get(url)
    return web_context.text

url = 'http://openaccess.thecvf.com//CVPR2016.py'
web_context = get_context(url)

link_list = re.findall(r"(?<=href=\").+?pdf(?=\">pdf)|(?<=href=\').+?pdf(?=\">pdf)",web_context)
name_list = re.findall(r"(?<=2016_paper.html\">).+(?=</a>)",web_context)

local_dir = 'D:\\CVPR16\\'
if not os.path.exists(local_dir):
    os.makedirs(local_dir)

cnt = 0
while cnt < len(link_list):
    file_name = name_list[cnt]
    download_url = link_list[cnt]
    file_name = re.sub('[:\?/]+',"_",file_name).replace(' ','_')
    file_path = local_dir + file_name + '.pdf'
    print( '['+str(cnt)+'/'+str(len(link_list))+'] Downloading' + file_path)
    urllib.request.urlretrieve("http://openaccess.thecvf.com/" + download_url, file_path)
    cnt += 1
print('Finished')