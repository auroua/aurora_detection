#encoding:UTF-8
from itertools import combinations
import numpy as np
from PIL import Image, ImageDraw


class ClusterNode(object):
    def __init__(self, vec, left, right, distance=0.0, count=1):
        self.left = left
        self.right = right
        self.vec = vec
        self.distance = distance
        self.count = count

    def extract_cluster(self, dist):
        """从层次聚类中提取距离小于dist的子树簇群列表"""
        if self.distance < dist:
            return [self]
        return self.left.extract_cluster(dist)+self.right.extract_cluster(dist)

    def get_cluster_elements(self):
        """在聚类子树中返回元素id"""
        return self.left.get_cluster_elements()+self.right.get_cluster_elements()

    def get_height(self):
        """返回节点的高度，高度是各分支的和"""
        return self.left.get_height()+self.right.get_height()

    def get_depth(self):
        """返回节点的深度，深度是每个节点取最大再加上它的自身距离"""
        return max(self.left.get_depth(), self.right.get_depth())+self.distance

    def get_distance(self):
        return self.left.get_distance() + self.right.get_distance() + [self.distance]

    def draw(self,draw,x,y,s,imlist,im):
        """    Draw nodes recursively with image
            thumbnails for leaf nodes. """

        h1 = int(self.left.get_height()*20 / 2)
        h2 = int(self.right.get_height()*20 /2)
        top = y-(h1+h2)
        bottom = y+(h1+h2)

        # vertical line to children
        draw.line((x,top+h1,x,bottom-h2),fill=(0,0,0))

        # horizontal lines
        ll = self.distance*s
        draw.line((x,top+h1,x+ll,top+h1),fill=(0,0,0))
        draw.line((x,bottom-h2,x+ll,bottom-h2),fill=(0,0,0))

        # draw left and right child nodes recursively
        self.left.draw(draw,x+ll,top+h1,s,imlist,im)
        self.right.draw(draw,x+ll,bottom-h2,s,imlist,im)


class ClusterLeafNode(object):
    def __init__(self, vec, id):
        self.vec = vec
        self.id = id

    def extract_clusters(self, dist):
        return [self]

    def get_cluster_elements(self):
        return [self.id]

    def get_height(self):
        return 1

    def get_depth(self):
        return 0

    def get_distance(self):
        return [-1]

    def draw(self,draw,x,y,s,imlist,im):
        nodeim = Image.open(imlist[self.id])
        nodeim.thumbnail([20,20])
        ns = nodeim.size
        im.paste(nodeim,[int(x),int(y-ns[1]//2),int(x+ns[0]),int(y+ns[1]-ns[1]//2)])

    def extract_cluster(self, dist):
        return [self]


def L2dist(v1, v2):
    return np.sqrt(np.sum((v1-v2)**2))


def L1dist(v1, v2):
    return np.sum(np.abs(v1-v2))


def hcluster(features, distfcn=L2dist):
    """用层次聚类对行特征进行聚类"""
    # 用于保存计算出的距离
    distances = {}

    node = [ClusterLeafNode(np.array(f), id=i) for i, f in enumerate(features)]

    while len(node) > 1:
        closest = float('Inf')

        for ni, nj in combinations(node, 2):
            if (ni, nj) not in distances:
                distances[ni, nj] = distfcn(ni.vec, nj.vec)
            d = distances[ni, nj]
            if d < closest:
                closest = d
                lowestpair = (ni, nj)
        ni, nj = lowestpair

        new_vec = (ni.vec+nj.vec)/2.0

        new_node = ClusterNode(new_vec, left=ni, right=nj, distance=closest)
        node.remove(ni)
        node.remove(nj)
        node.append(new_node)
    return node[0]


def draw_dendrogram(node, imlist, filename='clusters.jpg'):
    """绘制聚类树状图，并保存到文件"""
    rows = node.get_height()*20
    cols = 1200

    s = float(cols-150)/node.get_depth()
    im = Image.new('RGB', (cols, rows), (255, 255, 255))
    draw = ImageDraw.Draw(im)

    draw.line((0, rows/2, 20, rows/2), fill=(0, 0, 0))
    node.draw(draw, 20, (rows/2), s, imlist, im)
    im.save(filename)
    im.show()

if __name__ == '__main__':
    class1 = 1.5*np.random.randn(100, 2)
    class2 = np.random.randn(100, 2)+np.array([5, 5])
    features = np.vstack((class1, class2))
    tree = hcluster(features)

    clusters = tree.extract_cluster(5)
    print tree.get_depth(), tree.get_height()
    print 'number of clusters', len(clusters)
    for c in clusters:
        print c.get_cluster_elements()
