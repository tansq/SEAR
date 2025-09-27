import cv2
import numpy as np
from tqdm import tqdm


def random_crop(img, size):
    h, w = img.shape[:2]
    h = np.random.randint(0, h - size)
    w = np.random.randint(0, w - size)
    return img[h:h + size, w:w + size]


def load_img_array(base_path, filelines, img_size=512):
    X = []
    Y = []
    O = []
    size = (img_size, img_size)
    with tqdm(total=len(filelines)) as pbar:
        for line in filelines:
            pbar.update(1)
            pbar.set_description("loading image")
            # origin = cv2.imread(base_path+line.split(' ')[0])
            # origin = cv2.resize(origin,size,interpolation=cv2.INTER_AREA)
            # origin = origin[:,:,::-1]
            if 'IMD2020' in base_path:
                rgb = cv2.imread(base_path + line.split(' ')[0])
                rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
                rgb = rgb[:, :, ::-1]

                mask = cv2.imread(base_path + line.split(' ')[1])  # ,cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
            else:
                rgb = cv2.imread(base_path + line.split(' ')[1])
                rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
                rgb = rgb[:, :, ::-1]

                mask = cv2.imread(base_path + line.split(' ')[2])  # ,cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
            # for tip-dataset the forgery is 0 so change it

            if 'NIST' not in base_path:
                # trans black and white
                mask = 255. - mask
            mask = (255. - np.mean(mask, axis=-1)).reshape(size[0], size[1], 1)
            if 'IMD2020' in base_path:
                X.append(rgb / 255.)
            else:
                X.append(rgb / 255. * 2 - 1)
            Y.append((mask / 255.))
            # O.append((origin/255.)*2.0-1)

    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)  # ,np.array(O,dtype=np.float32)



# def load_locate_train_adv_img(base_path, database='nist'):
#     X = []
#     Y = []
#     adv_filelines = glob.glob(base_path + "adv/*")
#     gt_filelines = glob.glob(base_path + "gt/*")
#     size = (512, 512)
#     with tqdm(total=len(adv_filelines)) as pbar:
#         for i in range(len(adv_filelines)):
#             pbar.update(1)
#             pbar.set_description("loading adv image")
#             # origin = cv2.imread(base_path+line.split(' ')[0])
#             # origin = cv2.resize(origin,size,interpolation=cv2.INTER_AREA)
#             # origin = origin[:,:,::-1]
#             rgb = cv2.imread(adv_filelines[i])
#             rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
#             rgb = rgb[:, :, ::-1]
#
#             mask = cv2.imread(gt_filelines[i])
#             mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
#             # for tip-dataset the forgery is 0 so change it
#
#             if database != 'nist':
#                 # trans black and white
#                 mask = 255. - mask
#             mask = np.round(255. - np.mean(mask, axis=-1)).reshape(size[0], size[1], 1)
#
#             X.append(rgb / 255.)
#             Y.append((mask / 255.))
#             # O.append((origin/255.)*2.0-1)
#
#     return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)  # ,np.array(O,dtype=np.float32)


# def imageLoader(files, batch_size, model):
#     size = 256
#     L = len(files)
#
#     # for .51 gpu groups
#     # for i, name in enumerate(files):
#     #    files[i]=name.replace('data1','data2')
#     # this line is just to make the generator infinite, keras needs that
#     while True:
#         random.shuffle(files)
#         batch_start = 0
#         batch_end = batch_size
#
#         while batch_start < L:
#             limit = min(batch_end, L)
#             '''
#             if limit==L:
#                 break
#             X,Y = load_img_array(files[batch_start:limit])
#             '''
#             file = files[batch_start:limit]
#
#             with cf.ThreadPoolExecutor() as executor:
#                 rgb = executor.map(load_rgb, file)
#                 mask = executor.map(load_mask, file)
#                 rgb = list(rgb)
#                 mask = list(mask)
#
#                 X = np.array(rgb, dtype=np.float32) / 255.
#                 Y = np.array(mask, dtype=np.float32) / 255.
#
#             yield (X, Y)  # a tuple with two numpy arrays with batch_size samples
#
#             batch_start += batch_size
#             batch_end += batch_size


# def compare(present, best, auc=True):
#     if auc:
#         if present < best:
#             return present, True
#         else:
#             return best, False
#     else:
#         if present > best:
#             return present, True
#         else:
#             return best, False


def load_rgb(base_path, line, img_size=512):
    size = (img_size, img_size)
    # rgb = Image.open(line.split(' ')[0])
    # rgb = np.array(rgb)/255.
    if 'IMD2020' in base_path:
        rgb = cv2.imread(base_path + line.split(' ')[0])
    else:
        rgb = cv2.imread(base_path + line.split(' ')[1])
    rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
    # print rgb
    rgb = rgb[:, :, ::-1]

    return rgb


def load_mask(base_path, line, img_size=512):
    size = (img_size, img_size)

    # mask = Image.open(line.split(' ')[1]).convert('L')
    # mask = np.array(mask).reshape(512,512,1)/255.
    if 'IMD2020' in base_path:
        mask = cv2.imread(base_path + line.split(' ')[1])
    else:
        mask = cv2.imread(base_path + line.split(' ')[2])
    mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    mask = np.expand_dims(np.mean(mask, axis=-1), axis=-1)

    # for tip-dataset the forgery is 0 so change it
    # np.round(np.mean(self.y_val,axis=-1))
    if 'NIST' in base_path:
        # trans black and white
        mask = 255 - mask

    return mask / 255.


def get_data(dataset):
    if dataset == 'nist':
        batch_size = 20
        base_path = '/data2/zolo/forgrey_location/NIST/NC2016_Test/'
        nist = open(base_path + 'nist.txt', 'r').read().splitlines()
        train_files = nist[160:]  #
        valid_files = nist[:160]
    if dataset == 'casia1':
        batch_size = 20
        # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
        base_path = '/data2/zolo/forgrey_location/CASIA/'
        casia1 = open(base_path + 'casia1.txt', 'r').read().splitlines()
        casia2 = open(base_path + 'casia2.txt', 'r').read().splitlines()
        train_files = casia2  # [:20]
        valid_files = casia1
    if dataset == 'coverage':
        batch_size = 5
        # os.environ["CUDA_VISIBLE_DEVICES"] = "2"
        base_path = '/data2/zolo/forgrey_location/COVERAGE/'
        # data2/forgrey_location/COVERAGE/coverage.txt
        coverage = open(base_path + 'coverage.txt', 'r').read().splitlines()
        train_files = coverage[25:]
        valid_files = coverage[:25]
    if dataset == 'columbia':
        batch_size = 9
        # os.environ["CUDA_VISIBLE_DEVICES"] = "3"
        base_path = '/data2/zolo/forgrey_location/Columbia/'
        columbia = open(base_path + 'columbia.txt', 'r').read().splitlines()
        no = int(len(columbia) * 0.3)
        train_files = columbia[no:]
        valid_files = columbia[:no]
    if dataset == 'imd':
        base_path = '/data2/luosh/IMD2020/'
        file = open(base_path + 'imd.txt', 'r', encoding='utf-8')
        # file = open(base_path + 'nist.txt', 'r', encoding='utf-8')
        imd = file.read().splitlines()
        train_files = imd[:int(len(imd) * 0.75)]
        valid_files = imd[int(len(imd) * 0.75):]
    return base_path, train_files, valid_files
