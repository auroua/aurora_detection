import os
import urllib, urlparse
import simplejson as json


def download_images():
    URL = 'http://www.panoramio.com/map/get_panoramas.php?' \
          'order=popularity&set=public&size=medium&' \
          'from=20&to={n}&minx={minx}&miny={miny}&maxx={maxx}&maxy={maxy}'
    x = -122.026618
    y = 36.951614
    d = 0.001
    url = URL.format(n=40, minx=x - d, miny=y - d, maxx=x + d, maxy=y + d)
    c = urllib.urlopen(url)
    j = json.loads(c.read())
    imurls = []
    for im in j['photos']:
        imurls.append(im['photo_file_url'])
    for url in imurls:
        image = urllib.URLopener()
        image.retrieve(url, os.pardir.basename(urlparse.urlparse(url).path))
        print 'downloading: ', url


# the image url
# /home/aurora/hdd/workspace/PycharmProjects/data/pcv_img/panoimages/
if __name__=='__main__':
    download_images()