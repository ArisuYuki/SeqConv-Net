import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from thop import profile
import time
from dataLoader import read_las_file, LasDataLoader
import numpy as np
import os
import laspy


def select_classes(label_mat, classes=[3, 4, 5, 6, 7]):
    """
    屏蔽掉classes中的类别
    """
    mask = torch.zeros_like(label_mat)
    for c in classes:
        mask = mask * (label_mat==c)

    return mask.sum()!=0


def eval_fps(net, size=(96, 96), device="cpu"):
    print(f"FPS Calculating on {device}")
    model = net.to(device)
    input = torch.randint(
        1, net.elevation_resolution + 1, (1, 100, size[0], size[1])
    ).to(device)
    start_time = time.time()
    for _ in range(0, 10):
        pred_output(model, input, device=device)
    end_time = time.time()

    print("    FPS:", 10 / (end_time - start_time))
    print("    InputSize:", input.shape)
    print("----------------------------------")


def statistics(
    net,
    size=(96, 96),
    device="cpu",
):
    print("Net Statistics:")
    model = net.to(device)
    model.eval()
    input = torch.randint(
        1, net.elevation_resolution + 1, (1, 100, size[0], size[1])
    ).to(device)
    label = torch.randint(0, net.num_classes, (1, 100, size[0], size[1])).to(device)
    flops, params = profile(model, inputs=(input, label))
    print(f"    FLOPs: {flops/1e9} G")
    print(f"    Params: {params/1e6} M")
    print("----------------------------------")


class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """带遮蔽的softmax交叉熵损失函数"""

    def __init__(self, weights=None, smooth=1e-5):
        super().__init__(weight=weights, reduction="none")
        self.smooth = smooth

    def forward(self, pred, label, valid_len):
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(pred, label)
        weighted_loss = unweighted_loss * valid_len
        cross_loss = weighted_loss.sum() / valid_len.sum()
        dice_loss = self.dice_loss(pred, label, valid_len)
        return cross_loss + dice_loss

    def dice_loss(self, pred, target, valid_len):
        # 得到概率分布
        target_vector = target.permute(0, 2, 3, 1).reshape(-1)
        valid_len_vector = valid_len.permute(0, 2, 3, 1).reshape(-1)
        # valid_len_vector = valid_len.permute(0, 2, 3, 1).reshape(-1)
        class_num = pred.shape[1]
        input_soft = (
            F.softmax(pred, dim=1).permute(0, 2, 3, 4, 1).reshape(-1, class_num)
        )
        index = torch.arange(0, target_vector.shape[0], dtype=torch.int64).to(
            target_vector.device
        )
        # 得到每个位置的概率
        pred_vector = input_soft[index, target_vector]

        dice_loss = 0
        num_current_classes = 0
        for current_class in range(class_num):
            # 创建这个类别的遮罩
            label_mask = target_vector == current_class
            label_mask = label_mask * valid_len_vector
            if label_mask.sum() == 0:
                continue
            else:
                num_current_classes += 1
            dice_loss = (
                dice_loss
                + 1
                - (2 * ((pred_vector * label_mask).sum()) + self.smooth)
                / ((pred_vector * label_mask).sum() + label_mask.sum() + self.smooth)
            )
        return dice_loss / num_current_classes


def pred_output(net, input, max_step=None, device="cpu"):
    """
    对空间序列进行预测

    :param net: 网络
    :param input: 空间序列
    :param max_step: 预测的最大步数
    :param device: 计算设备
    """
    net.eval()
    with torch.no_grad():
        net = net.to(device)
        input = input.to(device)
        batch_size, num_steps, num_rows, num_cols = input.shape
        # 如果不设置开始标签，那么最大预测长度和输入一样
        if max_step == None:
            max_step = num_steps - 1

        # 创建开始标签
        current = torch.zeros((batch_size * num_rows * num_cols, 1))
        current = torch.fill(current, net.num_classes + 1).to(torch.int64).to(device)
        output = torch.zeros_like(input)

        init_state = net.encode(input)
        current_state = init_state.clone()
        for i in range(max_step):
            pred, current_state = net.decoder(current, current_state, init_state)
            # current = pred.argmax(dim=2)
            current = pred[:, :, 1:].argmax(dim=2) + 1
            output[:, i] = current.reshape(batch_size, num_rows, num_cols, 1).permute(
                0, 3, 1, 2
            )

        return output


def compute_acc(pred, label, num_classes, valid_len):

    acc = []
    recall = []
    iou = []
    # 每个类别计算精度，从1开始
    for i in range(num_classes):
        current_class = i + 1
        # 预测为此类的
        pred_mask = (pred == current_class) * valid_len

        # 标签为此类的
        label_mask = (label == current_class) * valid_len
        # 预测正确的
        acc_mask = (pred_mask * label_mask) * valid_len
        # 如果区域里没有这个类别，设置为0
        # 计算精度
        if pred_mask.sum().item() == 0:
            acc.append(0.0)
        else:
            acc.append((acc_mask.sum() / pred_mask.sum()).item())
        # 计算召回率
        if label_mask.sum().item() == 0:
            recall.append(0.0)
        else:
            recall.append((acc_mask.sum() / label_mask.sum()).item())
        # 计算iou
        if (pred_mask.sum() + label_mask.sum() - acc_mask.sum()).item() == 0:
            iou.append(0.0)
        else:
            iou.append(
                (
                    acc_mask.sum()
                    / (pred_mask.sum() + label_mask.sum() - acc_mask.sum())
                ).item()
            )
    return acc, recall, iou


def compute_mean(data):
    acc = np.array(data)
    acc_mask = acc != 0.0
    acc_mask_sum = np.sum(acc_mask, axis=0)
    acc_mask_sum = np.where(acc_mask_sum == 0, 1, acc_mask_sum)
    acc = np.sum(acc, axis=0) / acc_mask_sum
    return acc, np.mean(acc)


def eval_net_point(net, test_loader, device="cpu"):
    """
    以点的形式评估网络精度

    :param net: 网络
    :param test_loader: 测试数据
    :param device: 设备
    """
    print(f"-----------------Eval Point--------------")
    net.eval()
    num_classes = net.num_classes
    dataset = test_loader.dataset
    with torch.no_grad():
        acc = []
        recall = []
        iou = []
        with tqdm(dataset.files, desc="Validating", leave=False) as pbar_loss:
            for file_name in pbar_loss:
                file_path = os.path.join(dataset.data_path, file_name)
                current_acc, current_recall, current_iou = pred_file_once(
                    net,
                    file_path,
                    dataset.area_size,
                    dataset.num_z,
                    dataset.xy_resolution,
                    dataset.z_resolution,
                    device=device,
                    only_pred=False,
                )
                acc.append(current_acc)
                recall.append(current_recall)
                iou.append(current_iou)
                pbar_loss.set_description(f"mIOU:{compute_mean(iou)[-1]:.5f}")

        acc, mACC = compute_mean(acc)
        recall, mRecall = compute_mean(recall)
        iou, mIoU = compute_mean(iou)

        print(
            "Class:        ",
            *[f"{i+1:>5d}" for i in range(num_classes)],
            "  AVG",
            sep=" | ",
        )
        print(
            "Valid acc:    ",
            *[f"{i:.3f}" for i in acc],
            f"{mACC:.3f}",
            sep=" | ",
        )
        print(
            "Valid recall: ",
            *[f"{i:.3f}" for i in recall],
            f"{mRecall:.3f}",
            sep=" | ",
        )
        print(
            "Valid iou:    ",
            *[f"{i:.3f}" for i in iou],
            f"{mIoU:.3f}",
            sep=" | ",
        )

        return mACC, mRecall, mIoU


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def create_path(part_dir):
    for i in range(len(part_dir)):
        create_dir("/".join(part_dir[: i + 1]))


def padding_point(x, y, area_data):
    # 找到离位置最近的点
    distance = torch.abs(area_data[:, 0] - x) + torch.abs(area_data[:, 1] - y)
    index = torch.argmin(distance)
    padding = area_data[index]
    padding[0] = x
    padding[1] = y
    return padding.unsqueeze(0)


def preprocess(dataloader, resolution=100, need_label=True, half=True):
    """
    切块与预处理生成标签

    :param dataloader: 数据加载器
    :param resolution: 每一块的大小
    :param need_label: 是否还需要生成标签
    """
    # 创建切块数据目录
    part_dir = dataloader.data_path.split("/")
    if part_dir[0] == ".":
        part_dir = part_dir[1:]
    part_dir[0] = part_dir[0] + "_chunk"
    create_path(part_dir)
    data_path = "/".join(part_dir)
    if half:
        half_resolution = resolution / 2
    else:
        half_resolution = resolution
    data_name_list = []
    # 创建标签目录
    part_dir = dataloader.label_path.split("/")
    if part_dir[0] == ".":
        part_dir = part_dir[1:]
    part_dir[0] = part_dir[0] + "_chunk"
    create_path(part_dir)
    label_path = "/".join(part_dir)

    with tqdm(
        dataloader.files, desc="Chunking", leave=False, total=len(dataloader.files)
    ) as pbar_loss:
        for file_name in pbar_loss:
            file_path = os.path.join(dataloader.data_path, file_name)
            data, raw_header = read_las_file(file_path, include_header=True)
            data = torch.tensor(data)
            max_coord, _ = torch.max(data, dim=0)
            min_coord, _ = torch.min(data, dim=0)
            # 算算能划分多少快,+1不够划分的部分单独划分一块
            x_range = (max_coord[0] - min_coord[0]) // half_resolution + 1
            y_range = (max_coord[1] - min_coord[1]) // half_resolution + 1

            x_coord = half_resolution * torch.arange(x_range)
            y_coord = half_resolution * torch.arange(y_range)
            for x in torch.arange(x_range, dtype=torch.int64):
                for y in torch.arange(y_range, dtype=torch.int64):
                    # 得到矩形框的四个范围
                    start_x = x_coord[x] + min_coord[0]
                    start_y = y_coord[y] + min_coord[1]
                    end_x = start_x + resolution
                    end_y = start_y + resolution

                    # 如果超出了，那就从后往前划分
                    if end_x > max_coord[0]:
                        start_x = max_coord[0] - resolution
                        end_x = max_coord[0]
                    if end_y > max_coord[1]:
                        start_y = max_coord[1] - resolution
                        end_y = max_coord[1]

                    area_file_name = file_name.split(".")[0] + f"_{x}_{y}.las"
                    area_file_path = os.path.join(data_path, area_file_name)
                    # 拿到在矩形框内的部分
                    x_mask = (start_x <= data[:, 0]) * (data[:, 0] < end_x)
                    y_mask = (start_y <= data[:, 1]) * (data[:, 1] < end_y)
                    if (x_mask * y_mask).sum() == 0:
                        continue
                    data_name_list.append(area_file_name)
                    area_data = data[x_mask * y_mask]
                    start_point = padding_point(start_x, start_y, area_data)
                    end_point = padding_point(
                        end_x - dataloader.xy_resolution,
                        end_y - dataloader.xy_resolution,
                        area_data,
                    )
                    area_data = torch.concat((area_data, start_point, end_point), dim=0)
                    save_result(area_data, area_file_path, raw_header)
    if not need_label:
        return
    # 生成标签
    with tqdm(
        data_name_list,
        desc="Generating label",
        leave=False,
        total=len(data_name_list),
    ) as pbar_loss:
        for file_name in pbar_loss:
            file_path = os.path.join(data_path, file_name)
            data = read_las_file(file_path)
            data = torch.tensor(data)
            valid_input_len, input_mat, label_mat, force_teach_mat = (
                dataloader.generate_mat(data, file_name)
            )
            label_mat = label_mat.permute(1, 2, 0)
            force_teach_mat = force_teach_mat.permute(1, 2, 0)
            label_name = f"{file_name}.label"
            teach_name = f"{file_name}.force"
            torch.save(label_mat, os.path.join(label_path, label_name))
            torch.save(force_teach_mat, os.path.join(label_path, teach_name))


def save_input_mat(las_path, input_mat, label_mat, xy_resolution, z_resolution):
    """
    将空间序列保存成las文件

    :param las_path: 文件路径
    :param input_mat: 空间序列
    :param label_mat: 标签
    :param xy_resolution: 平面分辨率
    :param z_resolution: 高程分辨率
    """
    input_mat = torch.where(input_mat == input_mat.max(), 0, input_mat)
    point = []
    for i in range(input_mat.shape[1]):
        for j in range(input_mat.shape[2]):
            for k in range(input_mat.shape[0]):
                if input_mat[k, i, j] != 0:
                    x = i * xy_resolution
                    y = j * xy_resolution
                    z = input_mat[k, i, j] * z_resolution
                    c = label_mat[k, i, j]
                    point.append([x, y, z, c])
    point = torch.tensor(point)
    save_result(point, las_path)


def save_result(
    data,
    file_path,
    raw_header=None,
):
    """
    将点云保存成las文件

    :param data: 点云数据
    :param file_path: 文件路径
    :param raw_header: 原始文件头
    """
    # 创建文件然后写入
    if raw_header is None:
        header = laspy.LasHeader(point_format=3, version="1.2")
        header.offsets = np.array([0, 0, 0])
        header.scales = np.array([0.001, 0.001, 0.001])
    else:
        header = laspy.LasHeader(point_format=3, version="1.2")
        header.offsets = raw_header.offsets
        header.scales = raw_header.scales

    # 如果是tensor
    if torch.is_tensor(data):
        data = data.numpy()
    las_data = laspy.LasData(header=header)
    las_data.x = data[:, 0]
    las_data.y = data[:, 1]
    las_data.z = data[:, 2]
    las_data.classification = data[:, 3].astype(np.int64)

    las_data.write(file_path)


def pred_data(data, net, area_size, num_z, xy_resolution, z_resolution, device="cpu"):
    """
    对tensor里的点云进行预测

    :param data: tensor格式的点云
    :param net: 网络
    :param area_size: 每次加载的分块的大小
    :param num_z: 最大高程离散化区段数
    :param xy_resolution: 每个体素平面分辨率（单位：米）
    :param z_resolution: 每个体素平面分辨率（单位：米）
    :param device: 设备
    :return: （点，分类结果）
    """
    data_loader = LasDataLoader(
        data,
        area_size,
        num_z,
        xy_resolution,
        z_resolution,
    )
    loader = torch.utils.data.DataLoader(data_loader, batch_size=1)
    _, num_rows, num_cols = data_loader.input_mat.shape
    (
        data,
        input_mat,
        sort_input,
        sort_index,
        full_indices,
        num_rows,
        num_cols,
        x_indices,
        y_indices,
        z_indices,
        min_coord,
        max_coord,
    ) = data_loader.preprocess_info
    # pred_mat用来储存生成的结果
    pred_mat = torch.zeros(
        (num_rows, num_cols, num_z + 1),
        dtype=torch.int64,
    )
    full_indices = torch.arange(len(full_indices))
    for valid_input_len, input_mat, pos in loader:
        input_mat = input_mat.to(device)
        valid_input_len = valid_input_len.to(device).to(torch.float32)
        output = pred_output(net, input_mat, device=device)
        pred = output * valid_input_len
        pred = pred.squeeze(0).permute(1, 2, 0)
        pred_mat[
            pos[0] : pos[0] + area_size[0],
            pos[1] : pos[1] + area_size[1],
            : pred.shape[-1],
        ] = pred
    # pred_mat的位置不对，这里用反排序索引恢复高程的位置
    inverse_indices = torch.argsort(sort_index)
    total_label_mat = torch.take_along_dim(pred_mat, inverse_indices, dim=-1)
    # 拿到真正的每个点的结果
    pred_point_label = total_label_mat[x_indices, y_indices, z_indices]
    # 最后两个点是填充的，丢掉
    return data[full_indices], pred_point_label[full_indices]


def pred_file(
    net,
    las_path,
    area_size,
    num_z,
    xy_resolution,
    z_resolution,
    resolution=100,
    device="cpu",
    only_pred=True,
):
    """
    分块地对整个文件进行预测

    :param net: 网络
    :param las_path: 文件路径
    :param area_size: 每次加载的分块的大小
    :param num_z: 最大高程离散化区段数
    :param xy_resolution: 每个体素平面分辨率（单位：米）
    :param z_resolution: 每个体素平面分辨率（单位：米）
    :param device: 设备
    :param only_pred: 是否只进行预测不计算精度
    :return: （预测结果，las文件头）
    """
    # 对点云进行切片处理，划分间隔resolution*resolution米一块
    data, header = read_las_file(las_path, include_header=True)
    half_resolution = resolution
    data = torch.tensor(data)
    max_coord, _ = torch.max(data, dim=0)
    min_coord, _ = torch.min(data, dim=0)
    # 算算能划分多少快
    x_range = (max_coord[0] - min_coord[0]) // half_resolution
    y_range = (max_coord[1] - min_coord[1]) // half_resolution
    if x_range == 0:
        x_range = 1
    if y_range == 0:
        y_range = 1

    x_coord = half_resolution * torch.arange(x_range)
    y_coord = half_resolution * torch.arange(y_range)
    point = []
    pred_number_list = []
    label_number_list = []
    acc_number_list = []
    for x in torch.arange(x_range, dtype=torch.int64):
        for y in torch.arange(y_range, dtype=torch.int64):
            # 得到矩形框的四个范围
            start_x = x_coord[x] + min_coord[0]
            start_y = y_coord[y] + min_coord[1]
            end_x = start_x + resolution
            end_y = start_y + resolution

            # 如果超出了，那就从后往前划分,不够划分的部分连着前面的一起划分为一块
            if end_x + resolution > max_coord[0]:
                # start_x = max_coord[0] - resolution
                # if start_x < min_coord[0]:
                #     start_x = min_coord[0]
                end_x = max_coord[0]
            if end_y + resolution > max_coord[1]:
                # start_y = max_coord[1] - resolution
                # if start_y < min_coord[1]:
                #     start_y = min_coord[1]
                end_y = max_coord[1]

            # 拿到在矩形框内的部分
            x_mask = (start_x <= data[:, 0]) * (data[:, 0] < end_x)
            y_mask = (start_y <= data[:, 1]) * (data[:, 1] < end_y)
            if (x_mask * y_mask).sum() == 0:
                continue
            area_data = data[x_mask * y_mask]
            start_point = padding_point(start_x, start_y, area_data)
            end_point = padding_point(
                end_x - xy_resolution,
                end_y - xy_resolution,
                area_data,
            )
            area_data = torch.concat((area_data, start_point, end_point), dim=0)
            raw_points, pred_label = pred_data(
                area_data, net, area_size, num_z, xy_resolution, z_resolution, device
            )
            # +1恢复到1...num_classes区间里
            point.append(
                torch.cat((raw_points[:, :3], pred_label.unsqueeze(-1)), dim=1).numpy()
            )
            # 统计精度
            if not only_pred:
                pred_number, label_number, acc_number = count_number(
                    pred_label, raw_points[:, 3], net.num_classes
                )
                pred_number_list.append(pred_number)
                label_number_list.append(label_number)
                acc_number_list.append(acc_number)

    # 统计精度
    if not only_pred:
        acc, recall, iou = compute_area_acc(
            pred_number_list, label_number_list, acc_number_list
        )
        return acc, recall, iou
    else:
        point_cloud = np.row_stack(point)
        return point_cloud, header


def pred_file_once(
    net,
    las_path,
    area_size,
    num_z,
    xy_resolution,
    z_resolution,
    device="cpu",
    only_pred=True,
):
    """
    一次性对整个文件进行预测

    :param net: 网络
    :param las_path: 文件路径
    :param area_size: 每次加载的分块的大小
    :param num_z: 最大高程离散化区段数
    :param xy_resolution: 每个体素平面分辨率（单位：米）
    :param z_resolution: 每个体素平面分辨率（单位：米）
    :param device: 设备
    :param only_pred: 是否只进行预测不计算精度
    :return: （预测点云，las文件头）
    """
    # 对点云进行切片处理，划分间隔resolution*resolution米一块
    data, header = read_las_file(las_path, include_header=True)
    data = torch.tensor(data)
    # data[:, 3] -= 1
    raw_points, pred_label = pred_data(
        data, net, area_size, num_z, xy_resolution, z_resolution, device
    )
    cloud_points = torch.cat(
        (raw_points[:, :3], pred_label.unsqueeze(-1)), dim=1
    ).numpy()
    if not only_pred:
        pred_number, label_number, acc_number = count_number(
            pred_label, raw_points[:, 3], net.num_classes
        )
        acc, recall, iou = compute_area_acc(pred_number, label_number, acc_number)
        return acc, recall, iou
    # else:
    return cloud_points, header


def count_number(pred, label, num_classes):
    pred_number = []
    label_number = []
    acc_number = []
    # 每个类别计算精度，从1开始
    for i in range(num_classes):
        current_class = i+1
        # 预测为此类的
        pred_mask = pred == current_class
        # 标签为此类的
        label_mask = label == current_class
        # 预测正确的
        acc_mask = pred_mask * label_mask
        # 如果区域里没有这个类别，设置为0
        # 计算精度
        pred_number.append(pred_mask.sum().item())
        label_number.append(label_mask.sum().item())
        acc_number.append(acc_mask.sum().item())

    return pred_number, label_number, acc_number


def compute_area_acc(pred_number_list, label_number_list, acc_number_list):
    # 求总数
    pred_number = np.array(pred_number_list)
    label_number = np.array(label_number_list)
    acc_number = np.array(acc_number_list)
    # 避免除0
    pred_number = pred_number + 1
    label_number = label_number + 1
    # 计算
    acc = acc_number / pred_number
    recall = acc_number / label_number
    iou = acc_number / (label_number + pred_number - acc_number)
    return acc, recall, iou


def reindex_label(dataloader):
    labels = []
    for las_path in dataloader.files:
        las_path = os.path.join(dataloader.data_path, las_path)
        point_cloud = read_las_file(las_path)
        labels.append(point_cloud[:, -1])
    labels = np.concatenate(labels)
    unique_label = np.unique(labels)
    labels = torch.tensor(unique_label).to(torch.int64)

    unique_labels = sorted(set(labels))  # 去重并排序
    mapping = {
        old_label: new_label + 1 for new_label, old_label in enumerate(unique_labels)
    }

    with tqdm(
        dataloader.files,
        desc="reindex",
        leave=False,
        total=len(dataloader.files),
    ) as pbar_loss:
        for file_name in pbar_loss:
            file_path = os.path.join(dataloader.data_path, file_name)
            data, raw_header = read_las_file(file_path, include_header=True)
            data = torch.tensor(data)
            data_clone = data.clone()
            for raw_label in mapping:
                new_label = mapping[raw_label]
                data_clone = torch.where(data == raw_label, new_label, data_clone)
            save_result(data_clone, file_path, raw_header)
