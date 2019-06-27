
from bs4 import BeautifulSoup
import requests
import os
import subprocess

response = requests.get('http://www.viewfinderpanoramas.org/Coverage%20map%20viewfinderpanoramas_org3.htm')

soup = BeautifulSoup(response.content, "html.parser")

try:
    os.mkdir('srtmdata')
except:
    pass


res = []
for item in soup.findAll('area', {"shape" : "rect"}):
    res.append(item.get('href'))




#--directory-prefix=/var/cache/foobar/
done = []
for href in res:
    if href not in done:
        p = subprocess.Popen(['wget', '--directory-prefix=./sourcegeo/srtm/', href])
        p.wait()
        done.append(href)




