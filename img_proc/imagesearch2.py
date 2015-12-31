# encoding:UTF-8
import pickle
import sift
import imagesearch
import homography
import vocabulary
from vocabulary import *

if __name__=='__main__':
    url = '/home/aurora/hdd/workspace/PycharmProjects/data/pcv_img/first1000/'
    imlists = vocabulary.get_img_list(url)
    feature = vocabulary.get_feature_list(url)
    nbr_images = len(imlists)
    with open('vocabulary-new.pkl', 'rb') as f:
        voc = pickle.load(f)

    src = imagesearch.Searcher('test.db', voc)

    q_ind = 50
    nbr_results = 20

    res_reg = [w[1] for w in src.query(imlists[q_ind])[:nbr_images]]
    print 'top matches (regular):', res_reg

    q_locs, q_descr = sift.read_feature_from_file(feature[q_ind])
    fp = homography.make_homog(q_locs[:, :2].T)
    model = homography.RansacModel()

    rank = {}
    # load image features for result
    #载入候选图像的特征
    for ndx in res_reg[1:]:
        locs,descr = sift.read_feature_from_file(feature[ndx])  # because 'ndx' is a rowid of the DB that starts at 1
        # get matches
        matches = sift.match(q_descr,descr)
        ind = matches.nonzero()[0]
        ind2 = matches[ind]
        tp = homography.make_homog(locs[:,:2].T)
        # compute homography, count inliers. if not enough matches return empty list
        try:
            H,inliers = homography.H_from_ransac(fp[:,ind],tp[:,ind2],model,match_theshold=4)
        except:
            inliers = []
        # store inlier count
        rank[ndx] = len(inliers)

    # sort dictionary to get the most inliers first
    sorted_rank = sorted(rank.items(), key=lambda t: t[1], reverse=True)
    res_geom = [res_reg[0]]+[s[0] for s in sorted_rank]
    print 'top matches (homography):', res_geom

    # 显示查询结果
    imagesearch.plot_results(src,res_reg[:8]) #常规查询
    imagesearch.plot_results(src,res_geom[:8]) #重排后的结果
