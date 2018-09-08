import sys
import os

import scipy.ndimage as nd
import scipy.io as io
import numpy as np
import matplotlib.pyplot as plt
import skimage.measure as sk

from mpl_toolkits import mplot3d

from keras.preprocessing.image import img_to_array, load_img
from math import floor

try:
    import trimesh
    from stl import mesh
except:
    pass
    print('All dependencies not loaded, some functionality may not work')

# LOCAL_PATH = '/home/meetshah1995/datasets/ModelNet/3DShapeNets/volumetric_data/'
# SERVER_PATH = '/home/gpu_users/meetshah/3dgan/volumetric_data/'

LOCAL_PATH = '/Users/lucagaegauf/Documents/Misc/p_programming/deep_learning/GAN/statue_GAN/'
SERVER_PATH = ''


def getVF(path):
    raw_data = tuple(open(path, 'r'))
    header = raw_data[1].split()
    n_vertices = int(header[0])
    n_faces = int(header[1])
    vertices = np.asarray([map(float, raw_data[i + 2].split()) for i in range(n_vertices)])
    faces = np.asarray([map(int, raw_data[i + 2 + n_vertices].split()) for i in range(n_faces)])
    return vertices, faces


def plotFromVF(vertices, faces):
    input_vec = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            input_vec.vectors[i][j] = vertices[f[j], :]
    figure = plt.figure()
    axes = mplot3d.Axes3D(figure)
    axes.add_collection3d(mplot3d.art3d.Poly3DCollection(input_vec.vectors))
    scale = input_vec.points.flatten(-1)
    axes.auto_scale_xyz(scale, scale, scale)
    plt.show()


def plotFromVoxels(voxels):
    z, x, y = voxels.nonzero()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, -z, zdir='z', c='red')
    plt.show()


def getVFByMarchingCubes(voxels, threshold=0.5):
    v, f = sk.marching_cubes(voxels, level=threshold)
    return v, f


def plotMeshFromVoxels(voxels, threshold=0.5):
    v, f = getVFByMarchingCubes(voxels, threshold)
    plotFromVF(v, f)


def plotVoxelVisdom(voxels, visdom, title):
    v, f = getVFByMarchingCubes(voxels)
    visdom.mesh(X=v, Y=f, opts=dict(opacity=0.5, title=title))


def plotFromVertices(vertices):
    figure = plt.figure()
    axes = mplot3d.Axes3D(figure)
    axes.scatter(vertices.T[0, :], vertices.T[1, :], vertices.T[2, :])
    plt.show()


def getVolumeFromOFF(path, sideLen=32):
    mesh = trimesh.load(path)
    volume = trimesh.voxel.Voxel(mesh, 0.5).raw
    (x, y, z) = map(float, volume.shape)
    volume = nd.zoom(volume.astype(float),
                     (sideLen/x, sideLen/y, sideLen/z),
                     order=1,
                     mode='nearest')
    volume[np.nonzero(volume)] = 1.0
    return volume.astype(np.bool)


def getVoxelFromMat(path, cube_len=64):
    voxels = io.loadmat(path)['instance']
    voxels = np.pad(voxels, (1, 1), 'constant', constant_values=(0, 0))
    if cube_len != 32 and cube_len == 64:
        voxels = nd.zoom(voxels, (2, 2, 2), mode='constant', order=0)
    return voxels


# def getAll(obj='src_png', train=True, is_local=False, cube_len=64, obj_ratio=1.0):
#    if train:
#        objPath = SERVER_PATH + obj
#        if is_local:
#            objPath = LOCAL_PATH + obj
#    else:
#        objPath = TEST_PATH
#    fileList = [f for f in os.listdir(objPath) if f.endswith('.mat')]
#    fileList = fileList[0:int(obj_ratio*len(fileList))]
#    volumeBatch = np.asarray([getVoxelFromMat(objPath + f, cube_len) for f in fileList], dtype=np.bool)
#    return volumeBatch


def get_unique_image_names(path):
    """Get the unique (in terms of membership) png image names."""
    unique_imgs = list(set([f[:-8] for f in os.listdir(path)]))
    return unique_imgs


def scaleVoxelAspectRatio(voxels, img_dim=128):
    """Scale the voxels to img_dim and maintain the aspect ratio."""
    max_dim = np.max(voxels.shape)
    voxels = nd.zoom(voxels, img_dim/max_dim, mode='constant', order=0)

    dim_v = voxels.shape
    pad_d0 = img_dim - dim_v[0]
    pad_d1 = img_dim - dim_v[1]
    pad_d2 = img_dim - dim_v[2]

    pad_d1l, pad_d2l = floor(pad_d1 / 2), floor(pad_d2 / 2)
    pad_d1r, pad_d2r = pad_d1 - pad_d1l, pad_d2 - pad_d2l

    voxels = np.pad(voxels,
                    ((0, pad_d0),
                     (pad_d1l, pad_d1r),
                     (pad_d1l, pad_d2r)),
                    'constant',
                    constant_values=0)
    return voxels


def getVoxelFromPNG(path, imagesList, img_dim=128):
    """Load array of voxels belonging to one scan."""
    for i, image in enumerate(imagesList):
        img = load_img(path + image)
        if i == 0:
            voxels = np.zeros((len(imagesList), img.size[0], img.size[1]))

        voxels[i] = img_to_array(img)[:, :, 1]

    voxels = scaleVoxelAspectRatio(voxels, img_dim)
    voxels = voxels.astype('bool')

    return voxels


def getAll(obj='data/src_png/', train=True, is_local=False, img_dim=128, obj_ratio=1.0):
    """Load all scans."""
    if train:
        path = SERVER_PATH + obj
        if is_local:
            path = LOCAL_PATH + obj
    else:
        path = LOCAL_PATH + 'data/test'

    unique_imgs = get_unique_image_names(path)
    unique_imgs = unique_imgs[0:int(obj_ratio * len(unique_imgs))]
    imagesList = os.listdir(path)

    all_images = np.zeros((len(unique_imgs), img_dim, img_dim, img_dim))

    for i, img in enumerate(unique_imgs):
        imgs = [f for f in imagesList if img in f]
        all_images[i] = getVoxelFromPNG(path, imgs, img_dim)

    return all_images


if __name__ == '__main__':
    path = sys.argv[1]
    volume = getVolumeFromOFF(path)
    plotFromVoxels(volume)
