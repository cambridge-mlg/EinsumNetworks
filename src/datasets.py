import numpy as np
import os
import tempfile
import urllib.request
import utils
import shutil
import gzip
import subprocess
import csv
import scipy.io as sp


def maybe_download(directory, url_base, filename):
    filepath = os.path.join(directory, filename)
    if os.path.isfile(filepath):
        return False

    if not os.path.isdir(directory):
        utils.mkdir_p(directory)

    url = url_base + filename
    _, zipped_filepath = tempfile.mkstemp(suffix='.gz')
    print('Downloading {} to {}'.format(url, zipped_filepath))
    urllib.request.urlretrieve(url, zipped_filepath)
    print('{} Bytes'.format(os.path.getsize(zipped_filepath)))
    print('Move to {}'.format(filepath))
    shutil.move(zipped_filepath, filepath)
    return True


def maybe_download_mnist():
    mnist_files = ['train-images-idx3-ubyte.gz',
                   'train-labels-idx1-ubyte.gz',
                   't10k-images-idx3-ubyte.gz',
                   't10k-labels-idx1-ubyte.gz']

    for file in mnist_files:
        if not maybe_download('../data/mnist', 'http://yann.lecun.com/exdb/mnist/', file):
            continue
        print('unzip ../data/mnist/{}'.format(file))
        filepath = os.path.join('../data/mnist/', file)
        with gzip.open(filepath, 'rb') as f_in:
            with open(filepath[0:-3], 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)


def load_mnist():
    """Load MNIST"""

    maybe_download_mnist()

    data_dir = '../data/mnist'

    fd = open(os.path.join(data_dir, 'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    train_x = loaded[16:].reshape((60000, 784)).astype(np.float32)

    fd = open(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    train_labels = loaded[8:].reshape((60000)).astype(np.float32)

    fd = open(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    test_x = loaded[16:].reshape((10000, 784)).astype(np.float32)

    fd = open(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    test_labels = loaded[8:].reshape((10000)).astype(np.float32)

    train_labels = np.asarray(train_labels)
    test_labels = np.asarray(test_labels)

    return train_x, train_labels, test_x, test_labels


def maybe_download_fashion_mnist():
    mnist_files = ['train-images-idx3-ubyte.gz',
                   'train-labels-idx1-ubyte.gz',
                   't10k-images-idx3-ubyte.gz',
                   't10k-labels-idx1-ubyte.gz']

    for file in mnist_files:
        if not maybe_download('../data/fashion-mnist', 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/', file):
            continue
        print('unzip ../data/fashion-mnist/{}'.format(file))
        filepath = os.path.join('../data/fashion-mnist/', file)
        with gzip.open(filepath, 'rb') as f_in:
            with open(filepath[0:-3], 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)


def load_fashion_mnist():
    """Load fashion-MNIST"""

    maybe_download_fashion_mnist()

    data_dir = '../data/fashion-mnist'

    fd = open(os.path.join(data_dir, 'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    train_x = loaded[16:].reshape((60000, 784)).astype(np.float32)

    fd = open(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    train_labels = loaded[8:].reshape((60000)).astype(np.float32)

    fd = open(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    test_x = loaded[16:].reshape((10000, 784)).astype(np.float32)

    fd = open(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    test_labels = loaded[8:].reshape((10000)).astype(np.float32)

    train_labels = np.asarray(train_labels)
    test_labels = np.asarray(test_labels)

    return train_x, train_labels, test_x, test_labels


def maybe_download_debd():
    if os.path.isdir('../data/debd'):
        return
    subprocess.run(['git', 'clone', 'https://github.com/arranger1044/DEBD', '../data/debd'])
    wd = os.getcwd()
    os.chdir('../data/debd')
    subprocess.run(['git', 'checkout', '80a4906dcf3b3463370f904efa42c21e8295e85c'])
    subprocess.run(['rm', '-rf', '.git'])
    os.chdir(wd)


def load_debd(name, dtype='int32'):
    """Load one of the twenty binary density esimtation benchmark datasets."""

    maybe_download_debd()

    data_dir = '../data/debd'

    train_path = os.path.join(data_dir, 'datasets', name, name + '.train.data')
    test_path = os.path.join(data_dir, 'datasets', name, name + '.test.data')
    valid_path = os.path.join(data_dir, 'datasets', name, name + '.valid.data')

    reader = csv.reader(open(train_path, 'r'), delimiter=',')
    train_x = np.array(list(reader)).astype(dtype)

    reader = csv.reader(open(test_path, 'r'), delimiter=',')
    test_x = np.array(list(reader)).astype(dtype)

    reader = csv.reader(open(valid_path, 'r'), delimiter=',')
    valid_x = np.array(list(reader)).astype(dtype)

    return train_x, test_x, valid_x


DEBD = ['accidents', 'ad', 'baudio', 'bbc', 'bnetflix', 'book', 'c20ng', 'cr52', 'cwebkb', 'dna', 'jester', 'kdd',
        'kosarek', 'moviereview', 'msnbc', 'msweb', 'nltcs', 'plants', 'pumsb_star', 'tmovie', 'tretail', 'voting']

DEBD_shapes = {
    'accidents': dict(train=(12758, 111), valid=(2551, 111), test=(1700, 111)),
    'ad': dict(train=(2461, 1556), valid=(491, 1556), test=(327, 1556)),
    'baudio': dict(train=(15000, 100), valid=(3000, 100), test=(2000, 100)),
    'bbc': dict(train=(1670, 1058), valid=(330, 1058), test=(225, 1058)),
    'bnetflix': dict(train=(15000, 100), valid=(3000, 100), test=(2000, 100)),
    'book': dict(train=(8700, 500), valid=(1739, 500), test=(1159, 500)),
    'c20ng': dict(train=(11293, 910), valid=(3764, 910), test=(3764, 910)),
    'cr52': dict(train=(6532, 889), valid=(1540, 889), test=(1028, 889)),
    'cwebkb': dict(train=(2803, 839), valid=(838, 839), test=(558, 839)),
    'dna': dict(train=(1600, 180), valid=(1186, 180), test=(400, 180)),
    'jester': dict(train=(9000, 100), valid=(4116, 100), test=(1000, 100)),
    'kdd': dict(train=(180092, 64), valid=(34955, 64), test=(19907, 64)),
    'kosarek': dict(train=(33375, 190), valid=(6675, 190), test=(4450, 190)),
    'moviereview': dict(train=(1600, 1001), valid=(250, 1001), test=(150, 1001)),
    'msnbc': dict(train=(291326, 17), valid=(58265, 17), test=(38843, 17)),
    'msweb': dict(train=(29441, 294), valid=(5000, 294), test=(3270, 294)),
    'nltcs': dict(train=(16181, 16), valid=(3236, 16), test=(2157, 16)),
    'plants': dict(train=(17412, 69), valid=(3482, 69), test=(2321, 69)),
    'pumsb_star': dict(train=(12262, 163), valid=(2452, 163), test=(1635, 163)),
    'tmovie': dict(train=(4524, 500), valid=(591, 500), test=(1002, 500)),
    'tretail': dict(train=(22041, 135), valid=(4408, 135), test=(2938, 135)),
    'voting': dict(train=(1214, 1359), valid=(350, 1359), test=(200, 1359)),
}

DEBD_display_name = {
    'accidents': 'accidents',
    'ad': 'ad',
    'baudio': 'audio',
    'bbc': 'bbc',
    'bnetflix': 'netflix',
    'book': 'book',
    'c20ng': '20ng',
    'cr52': 'reuters-52',
    'cwebkb': 'web-kb',
    'dna': 'dna',
    'jester': 'jester',
    'kdd': 'kdd-2k',
    'kosarek': 'kosarek',
    'moviereview': 'moviereview',
    'msnbc': 'msnbc',
    'msweb': 'msweb',
    'nltcs': 'nltcs',
    'plants': 'plants',
    'pumsb_star': 'pumsb-star',
    'tmovie': 'each-movie',
    'tretail': 'retail',
    'voting': 'voting'}


def maybe_download_svhn():
    svhn_files = ['train_32x32.mat', 'test_32x32.mat', "extra_32x32.mat"]
    for file in svhn_files:
        maybe_download('../data/svhn', 'http://ufldl.stanford.edu/housenumbers/', file)


def load_svhn(dtype=np.uint8):
    """
    Load the SVHN dataset.
    """

    maybe_download_svhn()

    data_dir = '../data/svhn'

    data_train = sp.loadmat(os.path.join(data_dir, "train_32x32.mat"))
    data_test = sp.loadmat(os.path.join(data_dir, "test_32x32.mat"))
    data_extra = sp.loadmat(os.path.join(data_dir, "extra_32x32.mat"))

    train_x = data_train["X"].astype(dtype).reshape(32*32, 3, -1).transpose(2, 0, 1)
    train_labels = data_train["y"].reshape(-1)

    test_x = data_test["X"].astype(dtype).reshape(32*32, 3, -1).transpose(2, 0, 1)
    test_labels = data_test["y"].reshape(-1)

    extra_x = data_extra["X"].astype(dtype).reshape(32*32, 3, -1).transpose(2, 0, 1)
    extra_labels = data_extra["y"].reshape(-1)

    return train_x, train_labels, test_x, test_labels, extra_x, extra_labels


if __name__ == '__main__':
    print('Downloading dataset -- this might take a while')

    print()
    print('MNIST')
    maybe_download_mnist()

    print()
    print('fashion MNIST')
    maybe_download_fashion_mnist()

    print()
    print('20 binary datasets')
    maybe_download_debd()

    print()
    print('SVHN')
    maybe_download_svhn()
