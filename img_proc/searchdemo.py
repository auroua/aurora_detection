#encoding:UTF-8
import cherrypy, os, urllib, pickle
import imagesearch
from vocabulary import *
import vocabulary
import numpy as np

class SearchDemo(object):
    def __init__(self):
        with open('vocabulary-new.pkl', 'rb') as f:
            self.voc = pickle.load(f)
        url = 'static/'
        self.imlist = vocabulary.get_img_list(url)
        self.feature = vocabulary.get_feature_list(url)
        self.nbr_images = len(self.imlist)
        self.ndx = range(self.nbr_images)
        self.maxres = 15
        self.header = """
            <!doctype html>
            <head>
            <title>Image search</title>
            </head>
            <body>
            """
        self.footer = """
            </body>
            </html>
            """
    def index(self, query=None):
        self.src = imagesearch.Searcher('test.db', self.voc)
        html = self.header
        html += """
            <br />
            Click an image to search. <a href='?query='> Random selection </a> of images.
            <br /><br />
            """
        if query:
            # query the database and get top images
            #查询数据库，并获取前面的图像
            res = self.src.query(query)[:self.maxres]
            for dist, ndx in res:
                imname = self.src.get_filename(ndx)
                html += "<a href='?query="+imname+"'>"
                html += "<img src='"+imname+"' width='200' />"
                html += "</a>"
            # show random selection if no query
            # 如果没有查询图像则随机显示一些图像
        else:
            np.random.shuffle(self.ndx)
            for i in self.ndx[:self.maxres]:
                imname = self.imlist[i]
                html += "<a href='?query="+imname+"'>"
                html += "<img src='"+imname+"' width='200' />"
                html += "</a>"

        html += self.footer
        return html

    index.exposed = True


def init_web_db():
    imlist = imagesearch.get_img_list('static/')

#conf_path = os.path.dirname(os.path.abspath(__file__))
#conf_path = os.path.join(conf_path, "service.conf")
#cherrypy.config.update(conf_path)
#cherrypy.quickstart(SearchDemo())

cherrypy.quickstart(SearchDemo(), '/', config=os.path.join(os.path.dirname(__file__), 'service.conf'))