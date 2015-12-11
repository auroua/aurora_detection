#encoding:UTF-8
__author__ = 'auroua'
import PIL.Image as Image
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import img_tools as tools

if __name__ == '__main__':
    files = tools.get_imlist('/home/auroua/workspace/PycharmProjects/data/pcv_img/avg/')
    img = Image.open(files[0])
    im = pylab.array(img)
    pylab.imshow(im)

    x = [100, 100, 400, 400]
    y = [200, 500, 200, 500]
    pylab.plot(x, y, 'r*')
    pylab.plot(x[:2], y[:2], 'ks:')
    pylab.title('Plotting: "empire.jpg"')
    pylab.axis('off')
    plt.show()