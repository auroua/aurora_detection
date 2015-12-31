# encoding:UTF-8
from pysqlite2 import dbapi2 as sqlite
from vocabulary import *
import vocabulary
import sift
import pickle
import matplotlib.pyplot as plt
import math
from PIL import Image



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

    def create_tables2(self):
        """创建数据库表单"""
        self.con.execute('create table imwords2(imid, wordid, vocname)')
        self.con.execute('create index wordid_idx2 on imwords2(wordid)')
        self.con.execute('create index imid_idx2 on imwords2(imid)')
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
            cur = self.con.execute("insert into imlist(filename) values ('%s')" % imname)
            return cur.lastrowid
        else:
            return res[0]

    def add_to_index(self, imname, descr):
        """获取一副带有特征描述子的图像， 投影到词汇上并添加进数据库"""
        # if self.is_indexed(imname):
        #     return
        print 'indexing', imname
        imid = self.get_id(imname)
        imwords = self.voc.project(descr)
        # print descr.shape, imwords.shape, self.voc.voc.shape, self.voc.name
        nbr_words = imwords.shape[0]
        for i in range(nbr_words):
            word = imwords[i]
            self.con.execute("insert into imwords2(imid, wordid, vocname) values (?,?,?)", (imid, word, self.voc.name))
            if word == 0:
                self.con.execute("insert into imwords(imid, wordid, vocname) values (?,?,?)", (imid, 0, self.voc.name))
            else:
                self.con.execute("insert into imwords(imid, wordid, vocname) values (?,?,?)", (imid, i, self.voc.name))
        self.con.execute("insert into imhistograms(imid, histogram, vocname) VALUES (?,?,?)", (imid, pickle.dumps(imwords), self.voc.name))


class Searcher(object):
    def __init__(self, db, voc):
        self.con = sqlite.connect(db)
        self.voc = voc

    def __del__(self):
        self.con.close()

    def candidates_from_word(self, imword):
        """获取包含imword的图像列表"""
        # im_ids = self.con.execute("select distinct imid from imwords where wordid=%d" % imword).fetchall()
        im_ids = self.con.execute("select distinct imid from imwords2 where wordid=%d" % imword).fetchall()
        return [i[0] for i in im_ids]

    def candidates_from_histogram(self, imwords):
        """获取具有相似单词的图像列表"""
        # 返回非零单词的索引
        words = imwords.nonzero()[0]
        # 寻找候选图像
        candidates = []
        for word in words:
            c = self.candidates_from_word(word)
            candidates += c
        # 获取所有唯一的单词，并按照出现次数
        # print '------', len(candidates)
        if len(candidates) < 6:
            candidates.append(10000)
            candidates.append(10001)
            candidates.append(10002)
            candidates.append(10003)
        tmp = [(w, candidates.count(w)) for w in set(candidates)]
        tmp.sort(cmp=lambda x, y: cmp(x[1], y[1]))
        tmp.reverse()
        # 返回排序后的列表， 最匹配的排在最前面
        # print tmp
        return [w[0] for w in tmp]

    def get_imhistogram(self, imname):
        """返回衣服图像的单词直方图"""
        im_id = self.con.execute("select rowid from imlist where filename2='%s'" %imname).fetchone()
        print imname, im_id
        s = self.con.execute("select histogram from imhistograms where rowid='%d'" % im_id[0]).fetchone()

        return pickle.loads(str(s[0]))

    def query(self, imname):
        """查找所有与imname匹配的图像列表"""
        h = self.get_imhistogram(imname)
        candidates = self.candidates_from_histogram(h)

        matchscores = []
        for imid in candidates:
            # 获取名字
            if imid > 9999:
                matchscores.append((10000000000, imid))
                continue
            cand_name = self.con.execute("select filename2 from imlist where rowid='%d'" % imid).fetchone()
            cand_h = self.get_imhistogram(cand_name)
            cand_dist = math.sqrt(sum(self.voc.idf*(h-cand_h)**2))
            matchscores.append((cand_dist, imid))
        matchscores.sort()
        return matchscores

    def get_filename(self, imid):
        """返回图像id对应的文件名"""
        s = self.con.execute("select filename2 from imlist where rowid='%s'" % imid).fetchone()
        print 'in line 143, imagesearch', imid
        return s[0]

    def get_id(self, imname):
        """获取图像id， 如果不存在， 就进行添加"""
        cur = self.con.execute("select rowid from imlist where filename='%s'" % imname)
        res = cur.fetchone()
        return res[0]


def compute_ukbench_score(src, images):
    """对查询返回的前34个结果计算平均相似图像数，并返回结果"""
    nbr_images = len(images)
    pos = np.zeros((nbr_images, 4))
    ids = []
    for i in range(nbr_images):
        ids.append(src.get_id(images[i]))
        # results = src.query(images[i])
        pos[i] = [w[1]-1 for w in src.query(images[i])[:4]]
    # print pos
    # 计算分数，并返回平均分数
    count = 0
    # print ids
    for i in range(nbr_images):
        id = ids[i]
        id_mod = id % 4
        if id_mod == 1:
            val = [id, id+1, id+2, id+3]
            for j in range(pos.shape[1]):
                if (pos[i][j]+1) in val:
                    count += 1
        else:
            if id_mod==0:
                init = id - 3
            else:
                init = id - id_mod + 1
            val = [init, init+1, init+2, init+3]
            for j in range(pos.shape[1]):
                if (pos[i][j]+1) in val:
                    count += 1
    print count
    return count/(nbr_images*1.0)
    # score = np.array([(pos[i]//4) == (i//4) for i in range(nbr_images)])*1.0
    # return sum(score)/nbr_images


def plot_results(src, res):
    """显示在列表res中的图像"""
    plt.figure()
    nbr_results = len(res)
    for i in range(nbr_results):
        imname = src.get_filename(res[i])
        plt.subplot(1, nbr_results, i+1)
        plt.imshow(np.array(Image.open(imname)))
        plt.axis('off')
    plt.show()


if __name__ == '__main__':
    url = '/home/aurora/hdd/workspace/PycharmProjects/data/pcv_img/first1000/'
    with open('vocabulary-new.pkl', 'rb') as f:
        voc = pickle.load(f)
    indx = Indexer('test.db', voc)
    # # init database
    # indx.create_tables()
    # indx.create_tables2()
    # insert data to database
    # imlists = vocabulary.get_img_list(url)
    # feature = vocabulary.get_feature_list(url)
    # nbr_images = len(imlists)
    # for i in range(nbr_images):
    #     locs, descr = sift.read_feature_from_file(feature[i])
    #     if locs.shape[1] == 0:
    #         continue
    #     indx.add_to_index(imlists[i], descr)
    # indx.db_commit()
    #
    # src = Searcher('/home/aurora/hdd/workspace/PycharmProjects/test.db', voc)
    src = Searcher('test.db', voc)
    # locs, descr = sift.read_feature_from_file(url+'ukbench00432.sift')
    # iw = src.voc.project(descr)
    # # index = range(0, iw.shape[0], 1)
    # # figure, ax = plt.subplots()
    # # ax.scatter(index, iw)
    # # plt.show()
    #
    # print 'ask using a histogram...'
    # print src.candidates_from_histogram(iw)[:10]

    # print 'try a query...'
    # print src.query(url+'ukbench00000.jpg')[:4]

    imlists = vocabulary.get_img_list(url)
    # images = imlists
    # print images
    # results = compute_ukbench_score(src, images)
    # print results

    nbr_results = 6
    res = [w[1] for w in src.query(imlists[0])[:nbr_results]]
    plot_results(src, res)
    # test file match
    # imlists = vocabulary.get_img_list(url)
    # features_lists = vocabulary.get_feature_list(url)
    # files = [(w[-17:], d[-16:]) for w, d in zip(imlists, features_lists)]
    # print files