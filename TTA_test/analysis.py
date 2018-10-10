import mxnet as mx
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import cv2

def show_images(imgs, nrows, ncols, figsize=None):
    """plot a list of images"""
    if not figsize:
        figsize = (ncols, nrows)
    _, figs = plt.subplots(nrows, ncols, figsize=figsize)
    for i in range(nrows):
        for j in range(ncols):
            figs[i][j].imshow(imgs[i*ncols+j])
            figs[i][j].axes.get_xaxis().set_visible(False)
            figs[i][j].axes.get_yaxis().set_visible(False)
    plt.savefig('picture.jpg')
    plt.show()

def ten_crop(img, size):
    W, H = size
    iW, iH = img.shape[0:2]

    if iH < H or iW < W:
        print (iH,iW,H,W)
        raise ValueError('image size is smaller than crop size')

    img_flip = img[:, ::-1, :]
    crops = np.stack(
        [img[(iW - W) // 2:(iW + W) // 2, (iH - H) // 2:(iH + H) // 2, :],
        img[0:W, 0:H, :],
        img[0:W, iH - H:iH, :],
        img[iW - W:iW, 0:H, :],
        img[iW - W:iW, iH - H:iH, :],

        img_flip[(iW - W) // 2:(iW + W) // 2, (iH - H) // 2:(iH + H) // 2, :],
        img_flip[0:W, 0:H, :],
        img_flip[0:W, iH - H:iH, :],
        img_flip[iW - W:iW:, 0:H, :],
        img_flip[iW - W:iW, iH - H:iH, :]], axis=0
    )
    return (crops)

width = 512
img = cv2.resize(cv2.imread("test.jpg"), (width+50, width+50))
img_10 = ten_crop(img[:,:,::-1], (width, width))
show_images(img_10,2,5,(8,8))
