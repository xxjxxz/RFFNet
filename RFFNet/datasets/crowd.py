import cv2
import scipy.io
from PIL import Image
import torch.utils.data as data
import os
from glob import glob
import torch
import torchvision.transforms.functional as F
from matplotlib import pyplot as plt
from scipy import ndimage, spatial
from scipy.spatial import cKDTree
from torchvision import transforms
import random
import numpy as np
import scipy.io as sio


def random_crop(im_h, im_w, crop_h, crop_w):
    res_h = im_h - crop_h
    res_w = im_w - crop_w
    i = random.randint(0, res_h)
    j = random.randint(0, res_w)
    return i, j, crop_h, crop_w


def generate_density_map(height, width, points):
    """
    自适应高斯核密度图生成
    """
    dmap = np.zeros((height, width), dtype=np.float32)
    if len(points) == 0:
        return dmap

    # 使用固定sigma
    for pt in points:
        x, y = min(width - 1, max(0, int(pt[0]))), min(height - 1, max(0, int(pt[1])))
        dmap[y, x] = 1
    return ndimage.gaussian_filter(dmap, 2, mode='constant')

    # tree = spatial.KDTree(points.copy(), leafsize=2048)
    # distances, _ = tree.query(points, k=4)
    #
    # for i, pt in enumerate(points):
    #     x, y = min(width - 1, max(0, int(pt[0]))), min(height - 1, max(0, int(pt[1])))
    #
    #     pt_map = np.zeros((height, width), dtype=np.float32)
    #     pt_map[y, x] = 1.0
    #
    #     if len(points) > 3:
    #         sigma_i = (distances[i][1] + distances[i][2] + distances[i][3]) * 0.1
    #     else:
    #         sigma_i = np.average(np.array(dmap.shape)) / 2. / 2.  # fallback for very few points
    #
    #     dmap += ndimage.gaussian_filter(pt_map, sigma=sigma_i, mode='constant')
    #
    # return dmap

def gen_discrete_map(im_height, im_width, points):
    """
        func: generate the discrete map.
        points: [num_gt, 2], for each row: [width, height]
        """
    discrete_map = np.zeros([im_height, im_width], dtype=np.float32)
    h, w = discrete_map.shape[:2]
    num_gt = points.shape[0]
    if num_gt == 0:
        return discrete_map

    # fast create discrete map
    points_np = np.array(points).round().astype(int)
    p_h = np.minimum(points_np[:, 1], np.array([h - 1] * num_gt).astype(int))
    p_w = np.minimum(points_np[:, 0], np.array([w - 1] * num_gt).astype(int))
    p_index = torch.from_numpy(p_h * im_width + p_w).to(torch.int64)
    discrete_map = torch.zeros(im_width * im_height).scatter_add_(0, index=p_index,
                                                                  src=torch.ones(im_width * im_height)).view(im_height,
                                                                                                             im_width).numpy()

    ''' slow method
    for p in points:
        p = np.round(p).astype(int)
        p[0], p[1] = min(h - 1, p[1]), min(w - 1, p[0])
        discrete_map[p[0], p[1]] += 1
    '''
    assert np.sum(discrete_map) == num_gt
    return discrete_map


class Base(data.Dataset):
    def __init__(self, root_path, crop_size, downsample_ratio=8):

        self.root_path = root_path
        self.c_size = crop_size
        self.d_ratio = downsample_ratio
        assert self.c_size % self.d_ratio == 0
        self.dc_size = self.c_size // self.d_ratio
        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.trans_diff = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
        ])

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass

    def train_transform(self, img, keypoints):
        wd, ht = img.size
        st_size = 1.0 * min(wd, ht)
        assert st_size >= self.c_size
        assert len(keypoints) >= 0
        i, j, h, w = random_crop(ht, wd, self.c_size, self.c_size)
        img = F.crop(img, i, j, h, w)
        if len(keypoints) > 0:
            keypoints = keypoints - [j, i]
            idx_mask = (keypoints[:, 0] >= 0) * (keypoints[:, 0] <= w) * \
                       (keypoints[:, 1] >= 0) * (keypoints[:, 1] <= h)
            keypoints = keypoints[idx_mask]
        else:
            keypoints = np.empty([0, 2])

        gt_discrete = gen_discrete_map(h, w, keypoints)
        down_w = w // self.d_ratio
        down_h = h // self.d_ratio
        gt_discrete = gt_discrete.reshape([down_h, self.d_ratio, down_w, self.d_ratio]).sum(axis=(1, 3))
        assert np.sum(gt_discrete) == len(keypoints)

        if len(keypoints) > 0:
            if random.random() > 0.5:
                img = F.hflip(img)
                gt_discrete = np.fliplr(gt_discrete)
                keypoints[:, 0] = w - keypoints[:, 0]
        else:
            if random.random() > 0.5:
                img = F.hflip(img)
                gt_discrete = np.fliplr(gt_discrete)
        gt_discrete = np.expand_dims(gt_discrete, 0)

        return self.trans(img), torch.from_numpy(keypoints.copy()).float(), torch.from_numpy(
            gt_discrete.copy()).float()


class Crowd_mall_norm(Base):
    def __init__(self, root_path, crop_size,
                 downsample_ratio=8,
                 method='train',
                 initial_mask_ratio=0.9):
        super().__init__(root_path, crop_size, downsample_ratio)
        self.method = method
        if method not in ['train', 'val']:
            raise Exception("not implement")
        self.mask_ratio = initial_mask_ratio
        self.im_list = sorted(glob(os.path.join(self.root_path, 'frames', '*.jpg')))
        self.gt_path = os.path.join(self.root_path, 'mall_gt.mat')
        self.gt = sio.loadmat(self.gt_path)['frame'][0]
        self.pMapN = sio.loadmat(os.path.join(self.root_path, 'perspective_roi.mat'))['pMapN']
        self.roi = sio.loadmat(os.path.join(self.root_path, 'perspective_roi.mat'))['roi'][0][0][0]
        self.roi = np.array(self.roi, dtype=np.int32)  # 如果需要整数数组

        self.dmap = sio.loadmat(os.path.join(self.root_path, 'dmap.mat'))['dmap']
        if method == 'train':
            self.im_list = self.im_list[:800]
            self.gt = self.gt[:800]
            self.dmap = self.dmap[:800]
        elif method == 'val':
            self.im_list = self.im_list[800:]
            self.gt = self.gt[800:]
            self.dmap = self.dmap[800:]

        self.crop_point = True
        self.use_pot = False
        print('number of img [{}]: {}'.format(method, len(self.im_list)))

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, item):
        img_path = self.im_list[item]
        name = os.path.basename(img_path).split('.')[0]
        img = Image.open(img_path).convert('RGB')
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img[self.roi == 0] = 0
        mask_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 转换为 PIL.Image 对象
        img = Image.fromarray(mask_img)
        keypoints = self.gt[item][0][0][0]
        dmap = self.dmap[item]

        if self.method == 'train':
            if self.crop_point:
                # if self.use_pot:
                #     if random.random() > self.mask_ratio:
                #         img, keypoints = self.mask_people(img, keypoints)
                return self.train_transform(img, keypoints)
            else:
                return self.trans(img), dmap
        elif self.method == 'val':
            if self.crop_point:
                wd, ht = img.size
                st_size = 1.0 * min(wd, ht)
                if st_size < self.c_size:
                    rr = 1.0 * self.c_size / st_size
                    wd = round(wd * rr)
                    ht = round(ht * rr)
                    st_size = 1.0 * min(wd, ht)
                    img = img.resize((wd, ht), Image.BICUBIC)
                img = self.trans(img)
                return img, len(keypoints), name
            else:
                return self.trans(img), dmap.sum(), name

    def train_transform(self, img, keypoints):
        wd, ht = img.size
        st_size = 1.0 * min(wd, ht)
        # resize the image to fit the crop size
        if st_size < self.c_size:
            rr = 1.0 * self.c_size / st_size
            wd = round(wd * rr)
            ht = round(ht * rr)
            st_size = 1.0 * min(wd, ht)
            img = img.resize((wd, ht), Image.BICUBIC)
            keypoints = keypoints * rr
        assert st_size >= self.c_size, print(wd, ht)
        assert len(keypoints) >= 0
        i, j, h, w = random_crop(ht, wd, self.c_size, self.c_size)
        img = F.crop(img, i, j, h, w)
        if len(keypoints) > 0:
            keypoints = keypoints - [j, i]
            idx_mask = (keypoints[:, 0] >= 0) * (keypoints[:, 0] <= w) * \
                       (keypoints[:, 1] >= 0) * (keypoints[:, 1] <= h)
            keypoints = keypoints[idx_mask]
        else:
            keypoints = np.empty([0, 2])

        gt_discrete = gen_discrete_map(h, w, keypoints)
        down_w = w // self.d_ratio
        down_h = h // self.d_ratio
        gt_discrete = gt_discrete.reshape([down_h, self.d_ratio, down_w, self.d_ratio]).sum(axis=(1, 3))
        assert np.sum(gt_discrete) == len(keypoints)

        if len(keypoints) > 0:
            if random.random() > 0.5:
                img = F.hflip(img)
                gt_discrete = np.fliplr(gt_discrete)
                keypoints[:, 0] = w - keypoints[:, 0] - 1
        else:
            if random.random() > 0.5:
                img = F.hflip(img)
                gt_discrete = np.fliplr(gt_discrete)
        gt_discrete = np.expand_dims(gt_discrete, 0)

        return self.trans(img), torch.from_numpy(keypoints.copy()).float(), torch.from_numpy(
            gt_discrete.copy()).float()

    def update_mask_ratio(self, epoch, max_epochs):
        self.mask_ratio = max(0.1, 0.9 - epoch / max_epochs)

    def start_iter(self, cmd):
        self.use_pot = cmd

    def mask_people(self, img, kpoint):
        img_np = np.array(img)
        kdtree = cKDTree(kpoint.copy(), leafsize=2048)
        distances, _ = kdtree.query(kpoint, k=4)

        # 全局的平均距离作为参考
        global_r = sum((d[1] + d[2] + d[3]) / 3.0 for d in distances) / len(distances)

        # 将 pMapN 归一化到 0-1 范围内
        pMapN_normalized = (self.pMapN - self.pMapN.min()) / (self.pMapN.max() - self.pMapN.min())

        # 遍历每个点并绘制矩形
        for i, p in enumerate(kpoint):
            # 从归一化的 pMapN 中获取缩放因子，越远的区域 pMapN 越大
            x, y = int(p[0]), int(p[1])
            scale_factor = np.log(1 / pMapN_normalized[y, x]) * 0.6  # 取得 pMapN 中对应位置的比例因子
            r = global_r * scale_factor  # 调整 R1 以适应透视图

            r = min(r, global_r)

            # 根据调整后的 R1 设置矩形区域
            top_left = (int(p[0] - r / 2), int(p[1] - 0.5 * r))
            bottom_right = (int(p[0] + r / 2), int(p[1] + 1.5 * r))

            # 绘制矩形遮挡区域
            cv2.rectangle(img_np, top_left, bottom_right, (0, 0, 0), -1)  # -1填充矩形
        img_np[self.roi == 0] = 0

        return Image.fromarray(img_np), []


class Crowd_mall_no_crop(Base):
    def __init__(self, root_path,
                 downsample_ratio=8,
                 method='train'
                 ):
        super().__init__(root_path, downsample_ratio)
        self.method = method
        if method not in ['train', 'val']:
            raise Exception("not implement")
        self.im_list = sorted(glob(os.path.join(self.root_path, 'frames', '*.jpg')))
        self.gt_path = os.path.join(self.root_path, 'mall_gt.mat')
        self.gt = sio.loadmat(self.gt_path)['frame'][0]
        self.pMapN = sio.loadmat(os.path.join(self.root_path, 'perspective_roi.mat'))['pMapN']
        self.roi = sio.loadmat(os.path.join(self.root_path, 'perspective_roi.mat'))['roi'][0][0][0]
        if method == 'train':
            self.im_list = self.im_list[:800]
            self.gt = self.gt[:800]
        elif method == 'val':
            self.im_list = self.im_list[800:]
            self.gt = self.gt[800:]
        print('number of img [{}]: {}'.format(method, len(self.im_list)))

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, item):
        img_path = self.im_list[item]
        name = os.path.basename(img_path).split('.')[0]
        img = Image.open(img_path).convert('RGB')
        keypoints = self.gt[item][0][0][0]

        if self.method == 'train':

            return self.train_transform(img, keypoints)

        elif self.method == 'val':
            wd, ht = img.size
            st_size = 1.0 * min(wd, ht)
            if st_size < self.c_size:
                rr = 1.0 * self.c_size / st_size
                wd = round(wd * rr)
                ht = round(ht * rr)
                st_size = 1.0 * min(wd, ht)
                img = img.resize((wd, ht), Image.BICUBIC)
            img = self.trans(img)
            return img, len(keypoints), name

    def train_transform(self, img, keypoints):
        wd, ht = img.size
        st_size = 1.0 * min(wd, ht)
        # resize the image to fit the crop size
        if st_size < self.c_size:
            rr = 1.0 * self.c_size / st_size
            wd = round(wd * rr)
            ht = round(ht * rr)
            st_size = 1.0 * min(wd, ht)
            img = img.resize((wd, ht), Image.BICUBIC)
            keypoints = keypoints * rr
        assert st_size >= self.c_size, print(wd, ht)
        assert len(keypoints) >= 0

        gt_discrete = gen_discrete_map(ht, wd, keypoints)
        down_w = wd // self.d_ratio
        down_h = ht // self.d_ratio
        gt_discrete = gt_discrete.reshape([down_h, self.d_ratio, down_w, self.d_ratio]).sum(axis=(1, 3))
        assert np.sum(gt_discrete) == len(keypoints)

        if len(keypoints) > 0:
            if random.random() > 0.5:
                img = F.hflip(img)
                gt_discrete = np.fliplr(gt_discrete)
                keypoints[:, 0] = wd - keypoints[:, 0] - 1
        else:
            if random.random() > 0.5:
                img = F.hflip(img)
                gt_discrete = np.fliplr(gt_discrete)
        gt_discrete = np.expand_dims(gt_discrete, 0)

        return self.trans(img), torch.from_numpy(keypoints.copy()).float(), torch.from_numpy(
            gt_discrete.copy()).float()


class CustomDataset(Base):
    '''
    Class that allows training for a custom dataset. The folder are designed in the following way:
    root_dataset_path:
        -> images_1
        ->another_folder_with_image
        ->train.list
        ->valid.list

    The content of the lists file (csv with space as separator) are:
        img_xx__path label_xx_path
        img_xx1__path label_xx1_path

    where label_xx_path contains a list of x,y position of the head.
    '''

    def __init__(self, root_path, crop_size,
                 downsample_ratio=8,
                 method='train'):
        super().__init__(root_path, crop_size, downsample_ratio)
        self.method = method
        if method not in ['train', 'valid', 'test']:
            raise Exception("not implement")

        # read the list file
        self.img_to_label = {}
        list_file = f'{method}.list'  # train.list, valid.list or test.list
        with open(os.path.join(self.root_path, list_file)) as fin:
            for line in fin:
                if len(line) < 2:
                    continue
                line = line.strip().split()
                self.img_to_label[os.path.join(self.root_path, line[0].strip())] = \
                    os.path.join(self.root_path, line[1].strip())
        self.img_list = sorted(list(self.img_to_label.keys()))

        print('number of img [{}]: {}'.format(method, len(self.img_list)))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, item):
        img_path = self.img_list[item]
        gt_path = self.img_to_label[img_path]
        img_name = os.path.basename(img_path).split('.')[0]

        img = Image.open(img_path).convert('RGB')
        keypoints = self.load_head_annotation(gt_path)

        if self.method == 'train':
            return self.train_transform(img, keypoints)
        elif self.method == 'valid' or self.method == 'test':
            wd, ht = img.size
            st_size = 1.0 * min(wd, ht)
            if st_size < self.c_size:
                rr = 1.0 * self.c_size / st_size
                wd = round(wd * rr)
                ht = round(ht * rr)
                st_size = 1.0 * min(wd, ht)
                img = img.resize((wd, ht), Image.BICUBIC)
            img = self.trans(img)
            return img, len(keypoints), img_name

    def load_head_annotation(self, gt_path):
        annotations = []
        with open(gt_path) as annotation:
            for line in annotation:
                x = float(line.strip().split(' ')[0])
                y = float(line.strip().split(' ')[1])
                annotations.append([x, y])
        return np.array(annotations)

    def train_transform(self, img, keypoints):
        wd, ht = img.size
        st_size = 1.0 * min(wd, ht)
        # resize the image to fit the crop size
        if st_size < self.c_size:
            rr = 1.0 * self.c_size / st_size
            wd = round(wd * rr)
            ht = round(ht * rr)
            st_size = 1.0 * min(wd, ht)
            img = img.resize((wd, ht), Image.BICUBIC)
            keypoints = keypoints * rr
        assert st_size >= self.c_size, print(wd, ht)
        assert len(keypoints) >= 0
        i, j, h, w = random_crop(ht, wd, self.c_size, self.c_size)
        img = F.crop(img, i, j, h, w)
        if len(keypoints) > 0:
            keypoints = keypoints - [j, i]
            idx_mask = (keypoints[:, 0] >= 0) * (keypoints[:, 0] <= w) * \
                       (keypoints[:, 1] >= 0) * (keypoints[:, 1] <= h)
            keypoints = keypoints[idx_mask]
        else:
            keypoints = np.empty([0, 2])

        gt_discrete = gen_discrete_map(h, w, keypoints)
        down_w = w // self.d_ratio
        down_h = h // self.d_ratio
        gt_discrete = gt_discrete.reshape([down_h, self.d_ratio, down_w, self.d_ratio]).sum(axis=(1, 3))
        assert np.sum(gt_discrete) == len(keypoints)

        if len(keypoints) > 0:
            if random.random() > 0.5:
                img = F.hflip(img)
                gt_discrete = np.fliplr(gt_discrete)
                keypoints[:, 0] = w - keypoints[:, 0] - 1
        else:
            if random.random() > 0.5:
                img = F.hflip(img)
                gt_discrete = np.fliplr(gt_discrete)
        gt_discrete = np.expand_dims(gt_discrete, 0)

        return self.trans(img), torch.from_numpy(keypoints.copy()).float(), torch.from_numpy(
            gt_discrete.copy()).float()


class Crowd_mall(Base):
    def __init__(self, root_path, crop_size,
                 downsample_ratio=8,
                 method='train'):
        super().__init__(root_path, crop_size, downsample_ratio)
        self.method = method
        if method not in ['train', 'val']:
            raise Exception("not implement")

        self.im_list = np.load(os.path.join(root_path, 'img_list.npy'), allow_pickle=True)[1:]
        self.gt_list = sio.loadmat(os.path.join(root_path, 'gt_points.mat'))["gt_points"][0][1:]
        self.roi = sio.loadmat(os.path.join(root_path, 'roi.mat'))['roi']

        if method == 'train':
            self.tmp_list = self.im_list[0:999]
            self.gt = self.gt_list[0:999]
        elif method == 'val':
            self.tmp_list = self.im_list[799:]
            self.gt = self.gt_list[799:]

    def __len__(self):
        return len(self.tmp_list)

    def __getitem__(self, item):
        img_path = self.tmp_list[item]
        pre_img_path = self.get_previous_frame_path(img_path)
        cur_img = cv2.imread(img_path)
        pre_img = cv2.imread(pre_img_path)
        gt_points = self.gt[item]

        diff = self.cal_frameDiff(pre_img, cur_img)

        cur_img = Image.fromarray(cv2.cvtColor(cur_img, cv2.COLOR_BGR2RGB))
        diff = Image.fromarray(diff)

        if self.method == 'train':
            # 返回四个随机裁剪区域的信息，列表中包含四个元素
            return self.train_transform_video(cur_img, diff, gt_points)
        elif self.method == 'val':
            wd, ht = cur_img.size
            ground_truth_map = generate_density_map(ht, wd, gt_points)
            cur_img = self.trans(cur_img)
            diff = self.trans_diff(diff)
            return cur_img, diff, len(gt_points), ground_truth_map, 'bbb'

    def train_transform_video(self, img, diff, keypoints):
        wd, ht = img.size  # 注意：PIL 图片尺寸格式为 (width, height)
        st_size = 1.0 * min(wd, ht)
        # 若图像尺寸小于裁剪尺寸，则进行放大
        if st_size < self.c_size:
            rr = 1.0 * self.c_size / st_size
            wd = round(wd * rr)
            ht = round(ht * rr)
            st_size = 1.0 * min(wd, ht)
            img = img.resize((wd, ht), Image.BICUBIC)
            keypoints = keypoints * rr
        assert wd >= self.c_size and ht >= self.c_size, f"图像尺寸太小: {(wd, ht)}"
        assert len(keypoints) >= 0

        i, j, h, w = random_crop(ht, wd, self.c_size, self.c_size)
        img = F.crop(img.copy(), i, j, h, w)
        diff = F.crop(diff.copy(), i, j, h, w)
        if len(keypoints) > 0:
            keypoints = keypoints - [j, i]
            idx_mask = (keypoints[:, 0] >= 0) * (keypoints[:, 0] <= w) * \
                       (keypoints[:, 1] >= 0) * (keypoints[:, 1] <= h)
            keypoints = keypoints[idx_mask]
        else:
            keypoints = np.empty([0, 2])

        gt_discrete = gen_discrete_map(h, w, keypoints)
        dmap = generate_density_map(h, w, keypoints)
        down_w = w // self.d_ratio
        down_h = h // self.d_ratio
        gt_discrete = gt_discrete.reshape([down_h, self.d_ratio, down_w, self.d_ratio]).sum(axis=(1, 3))
        dmap = dmap.reshape([down_h, self.d_ratio, down_w, self.d_ratio]).sum(axis=(1, 3))
        assert np.sum(gt_discrete) == len(keypoints)

        if len(keypoints) > 0:
            if random.random() > 0.5:
                img = F.hflip(img)
                diff = F.hflip(diff)
                gt_discrete = np.fliplr(gt_discrete)
                dmap = np.fliplr(dmap)
                keypoints[:, 0] = w - keypoints[:, 0] - 1
        else:
            if random.random() > 0.5:
                img = F.hflip(img)
                diff = F.hflip(diff)
                gt_discrete = np.fliplr(gt_discrete)
                dmap = np.fliplr(dmap)
        gt_discrete = np.expand_dims(gt_discrete, 0)
        dmap = np.expand_dims(dmap, 0)
        img = self.trans(img)
        diff = self.trans_diff(diff)
        return (img,
                diff,
                torch.from_numpy(keypoints.copy()).float(),
                torch.from_numpy(gt_discrete.copy()).float(),
                torch.from_numpy(dmap.copy()).float())

    def get_previous_frame_path(self, current_frame_path):
        # 解析当前帧路径
        current_frame_filename = os.path.basename(current_frame_path)
        # 提取当前帧的编号（假设文件名格式为 "seq01_1490.jpg"）
        current_frame_number = int(current_frame_filename.split("_")[1].split(".")[0])
        # 计算上一帧的编号
        previous_frame_number = current_frame_number - 1
        formatted_frame_number = f"{previous_frame_number:06d}"  # 使用f-string格式化或zfill方法
        # 生成上一帧的文件名
        previous_frame_filename = f"seq_{formatted_frame_number}.jpg"
        # 生成上一帧的完整路径
        previous_frame_path = os.path.join(os.path.dirname(current_frame_path), previous_frame_filename)
        return previous_frame_path

    def cal_frameDiff(self, pre_img, cur_img):
        # 将图像转换为灰度图（若输入为彩色图像）
        if len(cur_img.shape) == 3:
            cur_img = cv2.cvtColor(cur_img, cv2.COLOR_BGR2GRAY)
        if len(pre_img.shape) == 3:
            pre_img = cv2.cvtColor(pre_img, cv2.COLOR_BGR2GRAY)
        # 计算帧差
        p1 = np.array(cur_img, dtype=np.float32)
        p2 = np.array(pre_img, dtype=np.float32)
        # diff = np.abs(p1 - p2)
        eps = np.ones_like(p1).astype('float32')
        diff = np.abs(np.log(p1 + eps) - np.log(p2 + eps))
        diff_norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        diff_filter = cv2.medianBlur(diff_norm, 9)
        return diff_filter



class Crowd_ucsd(Base):
    def __init__(self, root_path, crop_size,
                 downsample_ratio=8,
                 method='train'):
        super().__init__(root_path, crop_size, downsample_ratio)
        self.method = method
        if method not in ['train', 'val']:
            raise Exception("not implement")

        self.im_list = np.load(os.path.join(root_path, 'img_list.npy'), allow_pickle=True)[1:]
        self.gt_list = sio.loadmat(os.path.join(root_path, 'gt_points.mat'))["gt_points"][0][1:]
        self.roi = sio.loadmat(os.path.join(root_path, 'roi.mat'))['roi']

        if method == 'train':
            self.tmp_list = np.concatenate([self.im_list[0:800], self.im_list[900:]])
            self.gt = np.concatenate([self.gt_list[0:800], self.gt_list[900:]])
        elif method == 'val':
            self.tmp_list = self.im_list[600:1400]
            self.gt = self.gt_list[600:1400]

    def __len__(self):
        return len(self.tmp_list)

    def __getitem__(self, item):
        img_path = self.tmp_list[item]
        pre_img_path = self.get_previous_frame_path(img_path)
        cur_img = cv2.imread(img_path)
        pre_img = cv2.imread(pre_img_path)
        gt_points = self.gt[item]
        cur_img[self.roi == 0] = 0
        pre_img[self.roi == 0] = 0
        filtered_gt = []
        for (x, y) in gt_points:
            x_int = int(round(x))  # 将 x 转换为整数
            y_int = int(round(y))  # 将 y 转换为整数
            if 0 <= y_int < self.roi.shape[0] and 0 <= x_int < self.roi.shape[1]:  # 检查坐标是否在图像范围内
                if self.roi[y_int, x_int] == 1:  # 检查点是否在 ROI 内
                    filtered_gt.append([x, y])  # 保留原始坐标（浮点数）
        # 转换为 numpy 数组
        gt_points = np.array(gt_points, dtype=np.float32)

        # 上采样 3 倍
        height, width = cur_img.shape[:2]
        cur_img = cv2.resize(cur_img, (width * 4, height * 4), interpolation=cv2.INTER_CUBIC)
        pre_img = cv2.resize(pre_img, (width * 4, height * 4), interpolation=cv2.INTER_CUBIC)
        gt_points = gt_points * 4

        diff = self.cal_frameDiff(pre_img, cur_img)

        cur_img = Image.fromarray(cv2.cvtColor(cur_img, cv2.COLOR_BGR2RGB))

        diff = Image.fromarray(diff)

        if self.method == 'train':
            return self.train_transform_video(cur_img, diff, gt_points)
        elif self.method == 'val':
            wd, ht = cur_img.size
            st_size = 1.0 * min(wd, ht)
            if st_size < self.c_size:
                rr = 1.0 * self.c_size / st_size
                wd = round(wd * rr)
                ht = round(ht * rr)
                st_size = 1.0 * min(wd, ht)
                cur_img = cur_img.resize((wd, ht), Image.BICUBIC)

            cur_img = self.trans(cur_img)
            diff = self.trans_diff(diff)

            return cur_img, diff, len(gt_points), "aaa", "bbb"

    def train_transform_video(self, img, diff, keypoints):
        wd, ht = img.size
        st_size = 1.0 * min(wd, ht)
        # resize the image to fit the crop size
        if st_size < self.c_size:
            rr = 1.0 * self.c_size / st_size
            wd = round(wd * rr)
            ht = round(ht * rr)
            st_size = 1.0 * min(wd, ht)
            img = img.resize((wd, ht), Image.BICUBIC)
            keypoints = keypoints * rr
        assert st_size >= self.c_size, print(wd, ht)
        assert len(keypoints) >= 0
        i, j, h, w = random_crop(ht, wd, self.c_size, self.c_size)
        img = F.crop(img.copy(), i, j, h, w)
        diff = F.crop(diff.copy(), i, j, h, w)
        if len(keypoints) > 0:
            keypoints = keypoints - [j, i]
            idx_mask = (keypoints[:, 0] >= 0) * (keypoints[:, 0] <= w) * \
                       (keypoints[:, 1] >= 0) * (keypoints[:, 1] <= h)
            keypoints = keypoints[idx_mask]
        else:
            keypoints = np.empty([0, 2])

        gt_discrete = gen_discrete_map(h, w, keypoints)
        dmap = generate_density_map(h,w,keypoints)
        down_w = w // self.d_ratio
        down_h = h // self.d_ratio
        gt_discrete = gt_discrete.reshape([down_h, self.d_ratio, down_w, self.d_ratio]).sum(axis=(1, 3))
        dmap = dmap.reshape([down_h, self.d_ratio, down_w, self.d_ratio]).sum(axis=(1, 3))

        assert np.sum(gt_discrete) == len(keypoints)

        if len(keypoints) > 0:
            if random.random() > 0.5:
                img = F.hflip(img)
                diff = F.hflip(diff)
                gt_discrete = np.fliplr(gt_discrete)
                dmap = np.fliplr(dmap)
                keypoints[:, 0] = w - keypoints[:, 0] - 1
        else:
            if random.random() > 0.5:
                img = F.hflip(img)
                diff = F.hflip(diff)
                gt_discrete = np.fliplr(gt_discrete)
                dmap = np.fliplr(dmap)

        gt_discrete = np.expand_dims(gt_discrete, 0)
        dmap = np.expand_dims(dmap, 0)
        img = self.trans(img)
        diff = self.trans_diff(diff)
        return (img,
                diff,
                torch.from_numpy(keypoints.copy()).float(),
                torch.from_numpy(gt_discrete.copy()).float(),
                torch.from_numpy(dmap.copy()).float())

    def get_previous_frame_path(self, current_frame_path):
        # 解析当前帧路径
        base_dir = os.path.dirname(current_frame_path)  # 获取目录部分
        file_name = os.path.basename(current_frame_path)  # 获取文件名部分

        # 提取 XXX 和 YYY
        parts = file_name.split('_')
        xxx = parts[2]  # 提取 XXX（如 '006'）
        yyy = int(parts[-1].split('.')[0][1:])  # 提取 YYY（如 '200'），并转换为整数

        # 计算上一帧的 YYY
        previous_yyy = yyy - 1

        # 如果 YYY 小于 1，需要切换到上一份的最后一帧
        if previous_yyy < 1:
            previous_yyy = 200  # 上一份的最后一帧
            previous_xxx = f"{int(xxx) - 1:03d}"
            # 更新 base_dir
            parent_dir = os.path.dirname(base_dir)  # 获取父目录
            current_dir_name = os.path.basename(base_dir)  # 获取当前目录名
            current_xxx = current_dir_name.split('_')[-1].split('.')[0]  # 提取当前 XXX
            previous_dir_name = current_dir_name.replace(current_xxx, previous_xxx)  # 替换为上一份的 XXX
            base_dir = os.path.join(parent_dir, previous_dir_name)  # 更新 base_dir
            # 计算上一份的 XXX
        else:
            previous_xxx = xxx  # 保持当前份

        # 构造上一帧的文件名
        previous_file_name = f"vidf1_33_{previous_xxx}_f{previous_yyy:03d}.png"

        # 构造上一帧的完整路径
        previous_frame_path = os.path.join(base_dir, previous_file_name)

        return previous_frame_path

    def cal_frameDiff(self, pre_img, cur_img):
        # 将图像转换为灰度图（如果输入是彩色图像）
        if len(cur_img.shape) == 3:
            cur_img = cv2.cvtColor(cur_img, cv2.COLOR_BGR2GRAY)
        if len(pre_img.shape) == 3:
            pre_img = cv2.cvtColor(pre_img, cv2.COLOR_BGR2GRAY)

        # 计算帧差
        p1 = np.array(cur_img, dtype=np.float32)
        p2 = np.array(pre_img, dtype=np.float32)
        eps = np.ones_like(p1).astype('float32')
        diff = np.abs(np.log(p1 + eps) - np.log(p2 + eps))
        diff_norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        diff_filter = cv2.medianBlur(diff_norm, 5)
        return diff_filter


class Crowd_classroom(Base):
    def __init__(self, root_path, crop_size,
                 downsample_ratio=8,
                 method='train'):
        super().__init__(root_path, crop_size, downsample_ratio)
        self.method = method
        if method not in ['train', 'val']:
            raise Exception("not implement")

        self.im_list = np.load(os.path.join(root_path, 'img_list.npy'), allow_pickle=True)[1:]
        self.gt_list = sio.loadmat(os.path.join(root_path, 'gt_points.mat'))["gt_points"][0][1:]
        # self.roi = sio.loadmat(os.path.join(root_path, 'roi.mat'))['roi']

        if method == 'train':
            self.tmp_list = self.im_list[0:644]
            self.gt = self.gt_list[0:644]
        elif method == 'val':
            self.tmp_list = self.im_list[644:]
            self.gt = self.gt_list[644:]

    def __len__(self):
        return len(self.tmp_list)

    def __getitem__(self, item):
        img_path = self.tmp_list[item]
        pre_img_path = self.get_previous_frame_path(img_path)
        cur_img = cv2.imread(img_path)
        pre_img = cv2.imread(pre_img_path)
        gt_points = self.gt[item]

        diff = self.cal_frameDiff(pre_img, cur_img)

        cur_img = Image.fromarray(cv2.cvtColor(cur_img, cv2.COLOR_BGR2RGB))

        diff = Image.fromarray(diff)

        if self.method == 'train':
            return self.train_transform_video(cur_img, diff, gt_points)
        elif self.method == 'val':

            wd, ht = cur_img.size
            ground_truth_map = generate_density_map(ht, wd, gt_points)
            cur_img = self.trans(cur_img)
            diff = self.trans_diff(diff)
            return cur_img, diff, len(gt_points), ground_truth_map, 'bbb'

    def train_transform_video(self, img, diff, keypoints):
        wd, ht = img.size
        st_size = 1.0 * min(wd, ht)
        # resize the image to fit the crop size
        if st_size < self.c_size:
            rr = 1.0 * self.c_size / st_size
            wd = round(wd * rr)
            ht = round(ht * rr)
            st_size = 1.0 * min(wd, ht)
            img = img.resize((wd, ht), Image.BICUBIC)
            keypoints = keypoints * rr
        assert st_size >= self.c_size, print(wd, ht)
        assert len(keypoints) >= 0
        i, j, h, w = random_crop(ht, wd, self.c_size, self.c_size)
        img = F.crop(img.copy(), i, j, h, w)
        diff = F.crop(diff.copy(), i, j, h, w)
        if len(keypoints) > 0:
            keypoints = keypoints - [j, i]
            idx_mask = (keypoints[:, 0] >= 0) * (keypoints[:, 0] <= w) * \
                       (keypoints[:, 1] >= 0) * (keypoints[:, 1] <= h)
            keypoints = keypoints[idx_mask]
        else:
            keypoints = np.empty([0, 2])

        gt_discrete = gen_discrete_map(h, w, keypoints)
        dmap = generate_density_map(h, w, keypoints)

        down_w = w // self.d_ratio
        down_h = h // self.d_ratio
        gt_discrete = gt_discrete.reshape([down_h, self.d_ratio, down_w, self.d_ratio]).sum(axis=(1, 3))
        dmap = dmap.reshape([down_h, self.d_ratio, down_w, self.d_ratio]).sum(axis=(1, 3))

        assert np.sum(gt_discrete) == len(keypoints)

        if len(keypoints) > 0:
            if random.random() > 0.5:
                img = F.hflip(img)
                diff = F.hflip(diff)
                gt_discrete = np.fliplr(gt_discrete)
                dmap = np.fliplr(dmap)
                keypoints[:, 0] = w - keypoints[:, 0] - 1
        else:
            if random.random() > 0.5:
                img = F.hflip(img)
                diff = F.hflip(diff)
                gt_discrete = np.fliplr(gt_discrete)
                dmap = np.fliplr(dmap)
        gt_discrete = np.expand_dims(gt_discrete, 0)
        dmap = np.expand_dims(dmap, 0)
        img = self.trans(img)
        diff = self.trans_diff(diff)
        return (img,
                diff,
                torch.from_numpy(keypoints.copy()).float(),
                torch.from_numpy(gt_discrete.copy()).float(),
                torch.from_numpy(dmap.copy()).float())

    def get_previous_frame_path(self, current_frame_path):
        # 解析当前帧路径
        current_frame_filename = os.path.basename(current_frame_path)

        # 提取当前帧的编号（假设文件名格式为 "seq01_1490.jpg"）
        current_frame_number = int(current_frame_filename.split("_")[1].split(".")[0])

        # 计算上一帧的编号
        previous_frame_number = current_frame_number - 10

        # 生成上一帧的文件名
        previous_frame_filename = f"seq01_{previous_frame_number}.jpg"

        # 生成上一帧的完整路径
        previous_frame_path = os.path.join(os.path.dirname(current_frame_path), previous_frame_filename)

        if current_frame_number == 6450:
            previous_frame_path = previous_frame_path.replace('test', 'train')

        return previous_frame_path

    def cal_frameDiff(self, pre_img, cur_img):
        # 将图像转换为灰度图（如果输入是彩色图像）
        if len(cur_img.shape) == 3:
            cur_img = cv2.cvtColor(cur_img, cv2.COLOR_BGR2GRAY)
        if len(pre_img.shape) == 3:
            pre_img = cv2.cvtColor(pre_img, cv2.COLOR_BGR2GRAY)

        # 计算帧差
        p1 = np.array(cur_img, dtype=np.float32)
        p2 = np.array(pre_img, dtype=np.float32)
        # diff = np.abs(p1 - p2)
        eps = np.ones_like(p1).astype('float32')
        diff = np.abs(np.log(p1 + eps) - np.log(p2 + eps))
        diff_norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        diff_filter = cv2.medianBlur(diff_norm, 11)
        return diff_filter


class Crowd_bus(Base):
    def __init__(self, root_path, crop_size,
                 downsample_ratio=8,
                 method='train'):
        super().__init__(root_path, crop_size, downsample_ratio)
        self.method = method
        if method not in ['train', 'val']:
            raise Exception("not implement")

        self.im_list = np.load(os.path.join(root_path, 'img_list.npy'), allow_pickle=True)[1:]
        self.gt_list = sio.loadmat(os.path.join(root_path, 'gt_points.mat'))["gt_points"][0][1:]
        self.roi = sio.loadmat(os.path.join(root_path, 'roi.mat'))['roi']

        if method == 'train':
            self.tmp_list = self.im_list[0:1413]
            self.gt = self.gt_list[0:1413]
        elif method == 'val':
            self.tmp_list = self.im_list[1413:]
            self.gt = self.gt_list[1413:]

    def __len__(self):
        return len(self.tmp_list)

    def __getitem__(self, item):
        img_path = self.tmp_list[item]
        pre_img_path = self.get_previous_frame_path(img_path)
        cur_img = cv2.imread(img_path)
        pre_img = cv2.imread(pre_img_path)
        gt_points = self.gt[item]
        # cur_img[self.roi == 0] = 0
        # pre_img[self.roi == 0] = 0
        filtered_gt = []
        for (x, y) in gt_points:
            x_int = int(round(x))  # 将 x 转换为整数
            y_int = int(round(y))  # 将 y 转换为整数
            if 0 <= y_int < self.roi.shape[0] and 0 <= x_int < self.roi.shape[1]:  # 检查坐标是否在图像范围内
                if self.roi[y_int, x_int] == 1:  # 检查点是否在 ROI 内
                    filtered_gt.append([x, y])  # 保留原始坐标（浮点数）
        # 转换为 numpy 数组
        gt_points = np.array(filtered_gt)

        diff = self.cal_frameDiff(cur_img, pre_img)

        cur_img = Image.fromarray(cv2.cvtColor(cur_img, cv2.COLOR_BGR2RGB))

        pre_img = Image.fromarray(cv2.cvtColor(pre_img, cv2.COLOR_BGR2RGB))

        diff = Image.fromarray(diff)

        if self.method == 'train':
            return self.train_transform_video(cur_img, pre_img, diff, gt_points)
        elif self.method == 'val':
            wd, ht = cur_img.size
            st_size = 1.0 * min(wd, ht)
            if st_size < self.c_size:
                rr = 1.0 * self.c_size / st_size
                wd = round(wd * rr)
                ht = round(ht * rr)
                st_size = 1.0 * min(wd, ht)
                img_cur = cur_img.resize((wd, ht), Image.BICUBIC)

            cur_img = self.trans(cur_img)
            pre_img = self.trans(pre_img)
            diff = self.trans_diff(diff)

            return pre_img, cur_img, diff, len(gt_points), 'name'

    def train_transform_video(self, img, img_pre, diff, keypoints):
        wd, ht = img.size
        st_size = 1.0 * min(wd, ht)
        # resize the image to fit the crop size
        if st_size < self.c_size:
            rr = 1.0 * self.c_size / st_size
            wd = round(wd * rr)
            ht = round(ht * rr)
            st_size = 1.0 * min(wd, ht)
            img = img.resize((wd, ht), Image.BICUBIC)
            keypoints = keypoints * rr
        assert st_size >= self.c_size, print(wd, ht)
        assert len(keypoints) >= 0
        i, j, h, w = random_crop(ht, wd, self.c_size, self.c_size)
        img = F.crop(img.copy(), i, j, h, w)
        img_pre = F.crop(img_pre.copy(), i, j, h, w)
        diff = F.crop(diff.copy(), i, j, h, w)
        if len(keypoints) > 0:
            keypoints = keypoints - [j, i]
            idx_mask = (keypoints[:, 0] >= 0) * (keypoints[:, 0] <= w) * \
                       (keypoints[:, 1] >= 0) * (keypoints[:, 1] <= h)
            keypoints = keypoints[idx_mask]
        else:
            keypoints = np.empty([0, 2])

        gt_discrete = gen_discrete_map(h, w, keypoints)
        down_w = w // self.d_ratio
        down_h = h // self.d_ratio
        gt_discrete = gt_discrete.reshape([down_h, self.d_ratio, down_w, self.d_ratio]).sum(axis=(1, 3))
        assert np.sum(gt_discrete) == len(keypoints)

        if len(keypoints) > 0:
            if random.random() > 0.5:
                img = F.hflip(img)
                diff = F.hflip(diff)
                gt_discrete = np.fliplr(gt_discrete)
                keypoints[:, 0] = w - keypoints[:, 0] - 1
        else:
            if random.random() > 0.5:
                img = F.hflip(img)
                diff = F.hflip(diff)
                gt_discrete = np.fliplr(gt_discrete)
        gt_discrete = np.expand_dims(gt_discrete, 0)

        img = self.trans(img)
        img_pre = self.trans(img_pre)
        diff = self.trans_diff(diff)
        return img_pre, img, diff, torch.from_numpy(keypoints.copy()).float(), torch.from_numpy(
            gt_discrete.copy()).float()

    def get_previous_frame_path(self, current_frame_path):
        # 解析当前帧路径
        current_frame_filename = os.path.basename(current_frame_path)

        # 提取当前帧的编号（假设文件名格式为 "seq01_1490.jpg"）
        current_frame_number = int(current_frame_filename.split("_")[1].split(".")[0])

        # 计算上一帧的编号
        previous_frame_number = current_frame_number - 10

        # 生成上一帧的文件名
        previous_frame_filename = f"bus_{previous_frame_number}.jpg"

        # 生成上一帧的完整路径
        previous_frame_path = os.path.join(os.path.dirname(current_frame_path), previous_frame_filename)

        if current_frame_number == 14130:
            previous_frame_path = previous_frame_path.replace('test', 'train')

        return previous_frame_path

    def cal_frameDiff(self, pre_img, cur_img):
        # 将图像转换为灰度图（如果输入是彩色图像）
        if len(cur_img.shape) == 3:
            cur_img = cv2.cvtColor(cur_img, cv2.COLOR_BGR2GRAY)
        if len(pre_img.shape) == 3:
            pre_img = cv2.cvtColor(pre_img, cv2.COLOR_BGR2GRAY)

        # 计算帧差
        p1 = np.array(cur_img, dtype=np.float32)
        p2 = np.array(pre_img, dtype=np.float32)
        eps = np.ones_like(p1).astype('float32')
        diff = np.abs(np.log(p1 + eps) - np.log(p2 + eps))
        diff_norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        diff_filter = cv2.medianBlur(diff_norm, 5)
        return diff_filter


class Crowd_canteen(Base):
    def __init__(self, root_path, crop_size,
                 downsample_ratio=8,
                 method='train'):
        super().__init__(root_path, crop_size, downsample_ratio)
        self.method = method
        if method not in ['train', 'val']:
            raise Exception("not implement")

        self.im_list = np.load(os.path.join(root_path, 'img_list.npy'), allow_pickle=True)[1:]
        self.gt_list = sio.loadmat(os.path.join(root_path, 'gt_points.mat'))["gt_points"][0][1:]
        self.roi = sio.loadmat(os.path.join(root_path, 'roi.mat'))['roi']

        if method == 'train':
            self.tmp_list = self.im_list[0:2999]
            self.gt = self.gt_list[0:2999]
        elif method == 'val':
            self.tmp_list = self.im_list[2999:]
            self.gt = self.gt_list[2999:]

    def __len__(self):
        return len(self.tmp_list)

    def __getitem__(self, item):
        img_path = self.tmp_list[item]
        pre_img_path = self.get_previous_frame_path(img_path)
        cur_img = cv2.imread(img_path)
        pre_img = cv2.imread(pre_img_path)
        gt_points = self.gt[item]
        # cur_img[self.roi == 0] = 0
        # pre_img[self.roi == 0] = 0
        filtered_gt = []
        for (x, y) in gt_points:
            x_int = int(round(x))  # 将 x 转换为整数
            y_int = int(round(y))  # 将 y 转换为整数
            if 0 <= y_int < self.roi.shape[0] and 0 <= x_int < self.roi.shape[1]:  # 检查坐标是否在图像范围内
                if self.roi[y_int, x_int] == 1:  # 检查点是否在 ROI 内
                    filtered_gt.append([x, y])  # 保留原始坐标（浮点数）
        # 转换为 numpy 数组
        gt_points = np.array(gt_points, dtype=np.float32)

        height, width = cur_img.shape[:2]
        cur_img = cv2.resize(cur_img, (width * 2, height * 2), interpolation=cv2.INTER_CUBIC)
        pre_img = cv2.resize(pre_img, (width * 2, height * 2), interpolation=cv2.INTER_CUBIC)
        gt_points = gt_points * 2

        diff = self.cal_frameDiff(cur_img, cur_img)

        cur_img = Image.fromarray(cv2.cvtColor(cur_img, cv2.COLOR_BGR2RGB))

        pre_img = Image.fromarray(cv2.cvtColor(pre_img, cv2.COLOR_BGR2RGB))

        diff = Image.fromarray(diff)

        if self.method == 'train':
            return self.train_transform_video(cur_img, pre_img, diff, gt_points)
        elif self.method == 'val':
            wd, ht = cur_img.size
            st_size = 1.0 * min(wd, ht)
            if st_size < self.c_size:
                rr = 1.0 * self.c_size / st_size
                wd = round(wd * rr)
                ht = round(ht * rr)
                st_size = 1.0 * min(wd, ht)
                img_cur = cur_img.resize((wd, ht), Image.BICUBIC)

            cur_img = self.trans(cur_img)
            pre_img = self.trans(pre_img)
            diff = self.trans_diff(diff)

            return pre_img, cur_img, diff, len(gt_points), 'name'

    def train_transform_video(self, img, img_pre, diff, keypoints):
        wd, ht = img.size
        st_size = 1.0 * min(wd, ht)
        # resize the image to fit the crop size
        if st_size < self.c_size:
            rr = 1.0 * self.c_size / st_size
            wd = round(wd * rr)
            ht = round(ht * rr)
            st_size = 1.0 * min(wd, ht)
            img = img.resize((wd, ht), Image.BICUBIC)
            keypoints = keypoints * rr
        assert st_size >= self.c_size, print(wd, ht)
        assert len(keypoints) >= 0
        i, j, h, w = random_crop(ht, wd, self.c_size, self.c_size)
        img = F.crop(img.copy(), i, j, h, w)
        img_pre = F.crop(img_pre.copy(), i, j, h, w)
        diff = F.crop(diff.copy(), i, j, h, w)
        if len(keypoints) > 0:
            keypoints = keypoints - [j, i]
            idx_mask = (keypoints[:, 0] >= 0) * (keypoints[:, 0] <= w) * \
                       (keypoints[:, 1] >= 0) * (keypoints[:, 1] <= h)
            keypoints = keypoints[idx_mask]
        else:
            keypoints = np.empty([0, 2])

        gt_discrete = gen_discrete_map(h, w, keypoints)
        down_w = w // self.d_ratio
        down_h = h // self.d_ratio
        gt_discrete = gt_discrete.reshape([down_h, self.d_ratio, down_w, self.d_ratio]).sum(axis=(1, 3))
        assert np.sum(gt_discrete) == len(keypoints)

        if len(keypoints) > 0:
            if random.random() > 0.5:
                img = F.hflip(img)
                diff = F.hflip(diff)
                gt_discrete = np.fliplr(gt_discrete)
                keypoints[:, 0] = w - keypoints[:, 0] - 1
        else:
            if random.random() > 0.5:
                img = F.hflip(img)
                diff = F.hflip(diff)
                gt_discrete = np.fliplr(gt_discrete)
        gt_discrete = np.expand_dims(gt_discrete, 0)

        img = self.trans(img)
        img_pre = self.trans(img_pre)
        diff = self.trans_diff(diff)
        return img_pre, img, diff, torch.from_numpy(keypoints.copy()).float(), torch.from_numpy(
            gt_discrete.copy()).float()

    def get_previous_frame_path(self, current_frame_path):
        # 解析当前帧路径
        current_frame_filename = os.path.basename(current_frame_path)

        # '/home/amax/High_Performance_Disk/xxj_datasets/canteen/train/images/juyuan_26_1_30010.jpg'
        current_frame_number = int(current_frame_filename.split("_")[-1].split(".")[0])

        # 计算上一帧的编号
        previous_frame_number = current_frame_number - 10

        # 生成上一帧的文件名
        previous_frame_filename = f"juyuan_26_1_{previous_frame_number}.jpg"

        # 生成上一帧的完整路径
        previous_frame_path = os.path.join(os.path.dirname(current_frame_path), previous_frame_filename)

        if current_frame_number == 60000:
            previous_frame_path = previous_frame_path.replace('test', 'train')

        return previous_frame_path

    def cal_frameDiff(self, pre_img, cur_img):
        # 将图像转换为灰度图（如果输入是彩色图像）
        if len(cur_img.shape) == 3:
            cur_img = cv2.cvtColor(cur_img, cv2.COLOR_BGR2GRAY)
        if len(pre_img.shape) == 3:
            pre_img = cv2.cvtColor(pre_img, cv2.COLOR_BGR2GRAY)

        # 计算帧差
        p1 = np.array(cur_img, dtype=np.float32)
        p2 = np.array(pre_img, dtype=np.float32)
        eps = np.ones_like(p1).astype('float32')
        diff = np.abs(np.log(p1 + eps) - np.log(p2 + eps))
        diff_norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        diff_filter = cv2.medianBlur(diff_norm, 5)
        return diff_filter


class Crowd_fdst(Base):
    def __init__(self, root_path, crop_size,
                 downsample_ratio=8,
                 method='train'):
        super().__init__(root_path, crop_size, downsample_ratio)
        self.method = method
        if method not in ['train', 'val']:
            raise Exception("not implement")

        self.im_list = np.load(os.path.join(root_path, 'img_list.npy'), allow_pickle=True)
        self.gt_list = sio.loadmat(os.path.join(root_path, 'gt_points.mat'))["gt_points"][0]

        if method == 'train':
            self.tmp_list = self.im_list[0:8943]
            self.gt = self.gt_list[0:8943]
        elif method == 'val':
            self.tmp_list = self.im_list[8943:]
            self.gt = self.gt_list[8943:]

    def __len__(self):
        return len(self.tmp_list)

    def __getitem__(self, item):
        img_path = self.tmp_list[item]
        pre_img_path = self.get_previous_frame_path(img_path)
        cur_img = cv2.imread(img_path)
        pre_img = cv2.imread(pre_img_path)

        # cur_img = cv2.resize(cur_img,
        #            (cur_img.shape[1] // 2, cur_img.shape[0] // 2),
        #            interpolation=cv2.INTER_AREA)
        #
        # pre_img = cv2.resize(pre_img,
        #            (pre_img.shape[1] // 2, pre_img.shape[0] // 2),
        #            interpolation=cv2.INTER_AREA)
        gt_points = self.gt[item]

        # 转换为 numpy 数组
        gt_points = np.array(gt_points, dtype=np.float32)

        diff = self.cal_frameDiff(pre_img, cur_img)

        cur_img = Image.fromarray(cv2.cvtColor(cur_img, cv2.COLOR_BGR2RGB))

        diff = Image.fromarray(diff)
        if self.method == 'train':
            return self.train_transform_video(cur_img, diff, gt_points)
        elif self.method == 'val':
            wd, ht = cur_img.size
            ground_truth_map = generate_density_map(ht, wd, gt_points)

            cur_img = self.trans(cur_img)
            diff = self.trans_diff(diff)
            return cur_img, diff, len(gt_points), ground_truth_map, "bbb"

    def train_transform_video(self, img, diff, keypoints):
        wd, ht = img.size
        st_size = 1.0 * min(wd, ht)
        # resize the image to fit the crop size
        if st_size < self.c_size:
            rr = 1.0 * self.c_size / st_size
            wd = round(wd * rr)
            ht = round(ht * rr)
            st_size = 1.0 * min(wd, ht)
            img = img.resize((wd, ht), Image.BICUBIC)
            keypoints = keypoints * rr
        assert st_size >= self.c_size, print(wd, ht)
        assert len(keypoints) >= 0
        i, j, h, w = random_crop(ht, wd, self.c_size, self.c_size)
        img = F.crop(img.copy(), i, j, h, w)
        diff = F.crop(diff.copy(), i, j, h, w)
        if len(keypoints) > 0:
            keypoints = keypoints - [j, i]
            idx_mask = (keypoints[:, 0] >= 0) * (keypoints[:, 0] <= w) * \
                       (keypoints[:, 1] >= 0) * (keypoints[:, 1] <= h)
            keypoints = keypoints[idx_mask]
        else:
            keypoints = np.empty([0, 2])

        gt_discrete = gen_discrete_map(h, w, keypoints)
        dmap = generate_density_map(h, w, keypoints)
        down_w = w // self.d_ratio
        down_h = h // self.d_ratio
        gt_discrete = gt_discrete.reshape([down_h, self.d_ratio, down_w, self.d_ratio]).sum(axis=(1, 3))
        dmap = dmap.reshape([down_h, self.d_ratio, down_w, self.d_ratio]).sum(axis=(1, 3))

        assert np.sum(gt_discrete) == len(keypoints)

        if len(keypoints) > 0:
            if random.random() > 0.5:
                img = F.hflip(img)
                diff = F.hflip(diff)
                gt_discrete = np.fliplr(gt_discrete)
                dmap = np.fliplr(dmap)
                keypoints[:, 0] = w - keypoints[:, 0] - 1
        else:
            if random.random() > 0.5:
                img = F.hflip(img)
                diff = F.hflip(diff)
                gt_discrete = np.fliplr(gt_discrete)
                dmap = np.fliplr(dmap)
        gt_discrete = np.expand_dims(gt_discrete, 0)
        dmap = np.expand_dims(dmap, 0)
        img = self.trans(img)
        diff = self.trans_diff(diff)
        return (img,
                diff,
                torch.from_numpy(keypoints.copy()).float(),
                torch.from_numpy(gt_discrete.copy()).float(),
                torch.from_numpy(dmap.copy()).float())

    def get_previous_frame_path(self, current_frame_path):
        # 解析当前帧路径
        current_frame_filename = os.path.basename(current_frame_path)

        # '/home/amax/High_Performance_Disk/xxj_datasets/canteen/train/images/juyuan_26_1_30010.jpg'
        current_frame_number = int(current_frame_filename.split(".")[0])

        # 计算上一帧的编号
        previous_frame_number = current_frame_number - 1

        # 生成上一帧的文件名
        previous_frame_filename = f"{previous_frame_number:03d}.jpg"

        # 生成上一帧的完整路径
        previous_frame_path = os.path.join(os.path.dirname(current_frame_path), previous_frame_filename)

        return previous_frame_path

    def cal_frameDiff(self, pre_img, cur_img):
        # 将图像转换为灰度图（如果输入是彩色图像）
        if len(cur_img.shape) == 3:
            cur_img = cv2.cvtColor(cur_img, cv2.COLOR_BGR2GRAY)
        if len(pre_img.shape) == 3:
            pre_img = cv2.cvtColor(pre_img, cv2.COLOR_BGR2GRAY)

        # 计算帧差
        p1 = np.array(cur_img, dtype=np.float32)
        p2 = np.array(pre_img, dtype=np.float32)
        eps = np.ones_like(p1).astype('float32')
        diff = np.abs(np.log(p1 + eps) - np.log(p2 + eps))
        diff_norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        diff_filter = cv2.medianBlur(diff_norm, 5)
        return diff_filter


if __name__ == '__main__':
    data_dir = '/home/zns/xxj/CCTrans/datasets/ucsd_dataset'
    datasets = {
        "train": Crowd_ucsd(
            data_dir, 384, method="train"
        ),
        "val": Crowd_ucsd(
            data_dir, 384, method="val"
        ),
    }
    img_pre, img_cur, diff, kpoints, _ = datasets['train'][0]
    diff = np.array(diff)
    diff = cv2.cvtColor(diff, cv2.COLOR_RGB2BGR)  # 调整通道顺序

    img_cur = np.array(img_cur)
    img_cur = cv2.cvtColor(img_cur, cv2.COLOR_RGB2BGR)  # 调整通道顺序
    for point in kpoints:  # 遍历每组点
        x, y = int(point[0]), int(point[1])  # 提取 x 和 y
        cv2.circle(img_cur, (x, y), radius=4, color=(0, 255, 0), thickness=-1)  # 画绿色圆点
        cv2.circle(diff, (x, y), radius=4, color=(0, 255, 0), thickness=-1)  # 画绿色圆点

    # 保存图像
    output_path = f"./ucsd_dataset/"  # 保存路径
    cv2.imwrite(output_path + "image_test.jpg", img_cur)
    cv2.imwrite(output_path + "diff_test.jpg", diff)

    print(f"图像已保存到: {output_path}")
