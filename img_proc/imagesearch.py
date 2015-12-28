# encoding:UTF-8
from pysqlite2 import dbapi2 as sqlite
from vocabulary import *
import vocabulary
import sift
import pickle
import matplotlib.pyplot as plt

class Indexer(object):
    def __init__(self, db, voc):
        """初始化数据库的名称及词汇对象"""
        self.con = sqlite.connect(db)
        self.voc = voc

    def __del__(self):
        self.con.close()

    def db_commit(self):
        self.con.commit()

    def create_tables(self):
        """创建数据库表单"""
        self.con.execute('create table imlist(filename)')
        self.con.execute('create table imwords(imid, wordid, vocname)')
        self.con.execute('create table imhistograms(imid, histogram, vocname)')
        self.con.execute('create index im_idx on imlist(filename)')
        self.con.execute('create index wordid_idx on imwords(wordid)')
        self.con.execute('create index imid_idx on imwords(imid)')
        self.con.execute('create index imidhist_idex on imhistograms(imid)')
        self.db_commit()

    def is_indexed(self, imname):
        """如果图像名字被索引到，就返回True"""
        im = self.con.execute("select rowid from imlist where filename='%s'" % imname).fetchone()
        return im != None

    def get_id(self, imname):
        """获取图像id， 如果不存在， 就进行添加"""
        cur = self.con.execute("select rowid from imlist where filename='%s'" % imname)
        res = cur.fetchone()
        if res==None:
            cur = self.con.execute("insert into imlist(filename) values ('%s')"%imname)
            return cur.lastrowid
        else:
            return res[0]

    def add_to_index(self, imname, descr):
        """获取一副带有特征描述子的图像， 投影到词汇上并添加进数据库"""
        if self.is_indexed(imname):
            return
        print 'indexing', imname
        imid = self.get_id(imname)
        imwords = self.voc.project(descr)
        # print descr.shape, imwords.shape, self.voc.voc.shape, self.voc.name
        nbr_words = imwords.shape[0]
        for i in range(nbr_words):
            word = imwords[i]
            self.con.execute("insert into imwords(imid, wordid, vocname) values (?,?,?)", (imid, word, self.voc.name))
            self.con.execute("insert into imhistograms(imid, histogram, vocname) VALUES (?,?,?)", (imid, pickle.dumps(imwords), self.voc.name))


class Searcher(object):
    def __init__(self, db, voc):
        self.con = sqlite.connect(db)
        self.voc = voc

    def __del__(self):
        self.con.close()

    def candidates_from_word(self, imword):
        """获取包含imword的图像列表"""
        im_ids = self.con.execute("select distinct imid from imwords where wordid=%d" % imword).fetchall()
        print im_ids
        return [i[0] for i in im_ids]

    def candidates_from_histogram(self, imwords):
        """获取具有相似单词的图像列表"""
        # 返回非零单词的索引
        words = imwords.nonzero()[0]
        print words
        # 寻找候选图像
        candidates = []
        for word in words:
            c = self.candidates_from_word(word)
            candidates += c
        # 获取所有唯一的单词，并按照出现次数
        tmp = [(w, candidates.count(w)) for w in set(candidates)]
        tmp.sort(cmp=lambda x, y: cmp(x[1], y[1]))
        tmp.reverse()
        # 返回排序后的列表， 最匹配的排在最前面
        return [w[0] for w in tmp]


if __name__ == '__main__':
    url = '/home/aurora/hdd/workspace/PycharmProjects/data/pcv_img/first1000/'
    with open('vocabulary.pkl', 'rb') as f:
        voc = pickle.load(f)
    indx = Indexer('test.db', voc)
    # # init database
    # indx.create_tables()
    # # insert data to database
    # imlists = vocabulary.get_img_list(url)
    # feature = vocabulary.get_feature_list(url)
    # count = 0
    # nbr_images = len(imlists)
    # for i in range(nbr_images):
    #     locs, descr = sift.read_feature_from_file(feature[i])
    #     if locs.shape[1] == 0:
    #         continue
    #     indx.add_to_index(imlists[i], descr)
    # indx.db_commit()

    src = Searcher('test.db', voc)
    locs, descr = sift.read_feature_from_file(url+'ukbench00432.sift')
    iw = src.voc.project(descr)
    index = range(0, iw.shape[0], 1)
    figure, ax = plt.subplots()
    ax.scatter(index, iw)
    plt.show()

    print 'ask using a histogram...'
    print src.candidates_from_histogram(iw)