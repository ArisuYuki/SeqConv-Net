from torch.utils.data import IterableDataset
import os
import torch
import laspy
import numpy as np
import torchvision.transforms as transforms
import random


def read_las_file(
    las_path,
    include_returns=False,
    include_rgb=False,
    include_label=True,
    include_header=False,
    exclude_label=[],
):
    """
    读取LAS文件并根据参数返回不同的属性组合。

    :param las_path: LAS文件路径
    :param include_returns: 是否包含返回次数
    :param include_rgb: 是否包含RGB颜色信息
    :param include_label: 是否包含类别信息
    :param exclude_label: 排除指定标签的点
    :return: 包含所需属性的numpy数组
    """
    las_file = laspy.read(las_path)

    # 创建一个布尔掩码，用于排除指定的标签
    mask = ~np.isin(las_file.classification, exclude_label)

    # 根据掩码过滤所有属性
    attributes = [las_file.xyz[mask]]

    if include_returns:
        attributes.append(las_file.number_of_returns[mask])

    if include_rgb:
        attributes.extend(
            [las_file.red[mask], las_file.green[mask], las_file.blue[mask]]
        )

    if include_label:
        attributes.append(las_file.classification[mask])

    # 将所有属性合并成一个numpy数组
    points = np.column_stack(attributes)
    if include_header:
        return (points, las_file.header)
    else:
        return points


class PointCloudLoader(IterableDataset):

    def __init__(
        self,
        root_path,
        num_classes,
        area_size,
        num_z,
        xy_resolution,
        z_resolution,
        num_len=10,
        random=True,
        file_name=None,
        max_z_voxel=None,
    ):
        """
        从文件里加载点云

        :param root_path: 根路径
        :param num_classes: 点云类别数
        :param area_size: 每次加载的分块的大小
        :param num_z: 最大高程离散化区段数
        :param xy_resolution: 每个体素平面分辨率（单位：米）
        :param z_resolution: 每个体素平面分辨率（单位：米）
        :param num_len: 每个文件加载的分块数
        :param random: 是否随机加载文件
        :param file_name: 文件名（指定此文件名，只加载此文件）
        :param xmax_z_voxel: 每个位置上最长的体素长度
        """
        self.root_path = root_path
        self.data_path = os.path.join(self.root_path, "data")
        self.label_path = os.path.join(self.root_path, "label")
        if file_name is None:
            self.files = self.collect_las_files(self.data_path)
        else:
            self.files = [file_name]
        self.num_len = num_len
        self.area_size = area_size
        self.num_z = num_z
        self.max_z_voxel = max_z_voxel
        self.xy_resolution = xy_resolution
        self.z_resolution = z_resolution

        self.num_classes = num_classes
        # 设置填充token、开始token、结束token
        self.label_start_token = num_classes + 1
        self.label_padding_token = 0

        self.end_token = num_z + 1
        self.padding_token = 0

        self.random = random

    def get_weight(self):
        """计算标签的权重"""
        labels = []
        for las_path in self.files:
            las_path = os.path.join(self.data_path, las_path)
            point_cloud = read_las_file(las_path)
            labels.append(point_cloud[:, -1])
        labels = np.concatenate(labels)
        unique_label, counts = np.unique(labels, return_counts=True)

        label_weights = counts.astype(np.float32) + 0.001
        label_weights /= np.sum(label_weights)
        label_weights = np.power(np.amax(label_weights) / label_weights, 1 / 3)

        labels = torch.tensor(unique_label).to(torch.int64)
        weights = torch.tensor(label_weights)
        weights_tensor = torch.ones(self.num_classes + 1, dtype=torch.float32)

        for i in range(labels.shape[0]):
            weights_tensor[labels[i]] = weights[i]
        return weights_tensor

    def __iter__(self):
        for file_name in self.files:
            if self.random:
                index = torch.randint(0, len(self.files), (1,))[0]
                file_name = self.files[index]

            # 读取文件并生成输入和标签
            file_path = os.path.join(self.data_path, file_name)
            data = read_las_file(file_path)
            data = torch.tensor(data)

            valid_input_len, input_mat, label_mat, force_teach_mat = self.generate_mat(
                data, file_name
            )
            _, num_rows, num_cols = input_mat.shape
            # half_area = (self.area_size[0]//2, self.area_size[1]//2)
            # half_area = (self.area_size[0] // 2, self.area_size[1] // 2)
            # 计算分块的总行列数
            row_len = num_rows // self.area_size[0] + 1
            col_len = num_cols // self.area_size[1] + 1
            if (
                input_mat.shape[1] < self.area_size[0]
                or input_mat.shape[2] < self.area_size[1]
            ):
                continue
            # 如果不随机，按照顺序返回
            if not self.random:

                for i in range(row_len):
                    for j in range(col_len):

                        row_begin = i * self.area_size[0]
                        col_begin = j * self.area_size[1]

                        row_end = row_begin + self.area_size[0]
                        col_end = col_begin + self.area_size[1]

                        if col_end > num_cols:
                            col_end = num_cols
                            col_begin = num_cols - self.area_size[1]

                        if row_end > num_rows:
                            row_end = num_rows
                            row_begin = num_rows - self.area_size[0]

                        valid_input = valid_input_len[
                            :,
                            row_begin:row_end,
                            col_begin:col_end,
                        ]

                        valid_len = torch.sum(valid_input, dim=0).max()
                        if valid_len == 0:
                            continue
                        yield (
                            valid_input_len[
                                :valid_len,
                                row_begin:row_end,
                                col_begin:col_end,
                            ],
                            input_mat[
                                :valid_len,
                                row_begin:row_end,
                                col_begin:col_end,
                            ],
                            label_mat[
                                :valid_len,
                                row_begin:row_end,
                                col_begin:col_end,
                            ],
                            (row_begin, col_begin),
                        )
            else:
                index = torch.randint(0, len(self.files), (1,))[0]
                another_file_name = self.files[index]

                # 读取另一个文件并生成输入和标签
                another_file_path = os.path.join(self.data_path, another_file_name)
                another_data = read_las_file(another_file_path)
                another_data = torch.tensor(another_data)

                (
                    another_valid_input_len,
                    another_input_mat,
                    another_label_mat,
                    another_force_teach_mat,
                ) = self.generate_mat(another_data, another_file_name)
                # 否则随机生成一块 area_size大小的部分返回
                for _ in range(self.num_len):
                    # valid_input, input, label, force_teach = self.crop(
                    #     valid_input_len, input_mat, label_mat, force_teach_mat, self.area_size
                    # )
                    valid_input, input, label, force_teach = self.data_enhance(
                        valid_input_len,
                        input_mat,
                        label_mat,
                        force_teach_mat,
                        another_valid_input_len,
                        another_input_mat,
                        another_label_mat,
                        another_force_teach_mat,
                    )
                    if valid_input.sum() == 0:
                        continue
                    yield (
                        valid_input,
                        input,
                        label,
                        force_teach,
                    )

    def preprocess_data(self, data):
        # 计算坐标范围
        max_coord, _ = torch.max(data, dim=0)
        min_coord, _ = torch.min(data, dim=0)
        boundary = max_coord - min_coord
        # 计算行列数
        num_rows = torch.floor(boundary[0] / self.xy_resolution).to(torch.int64) + 1
        num_cols = torch.floor(boundary[1] / self.xy_resolution).to(torch.int64) + 1
        # 计算每个点的矩阵索引
        x_indices = torch.floor((data[:, 0] - min_coord[0]) / self.xy_resolution).to(
            torch.int64
        )
        y_indices = torch.floor((data[:, 1] - min_coord[1]) / self.xy_resolution).to(
            torch.int64
        )
        z_indices = torch.floor((data[:, 2] - min_coord[2]) / self.z_resolution).to(
            torch.int64
        )
        # 对于高程超出了最大长度num_z的，设置高程为num_z-1

        mask = z_indices > self.num_z - 1
        z_indices[mask] = self.num_z - 1

        # data = data[mask]
        # x_indices = x_indices[mask]
        # y_indices = y_indices[mask]
        # z_indices = z_indices[mask]

        # 构建一维索引
        full_indices = (
            x_indices * num_cols * self.num_z + y_indices * self.num_z + z_indices
        )
        # 对每个有效位置的体素，都置1
        input_mask_mat = torch.zeros(
            (num_rows, num_cols, self.num_z), dtype=torch.uint8
        )
        input_mask_mat[x_indices, y_indices, z_indices] = 1
        # 如果没有设定z方向上最大的高程点数,自动计算整个区域内最长的序列长度
        valid_input_len = torch.sum(input_mask_mat, dim=-1)
        self.max_z_voxel = valid_input_len.max() + 1

        # 在最后一个维度上拼接一个结束占位符
        end_mask = torch.ones((num_rows, num_cols, 1), dtype=torch.uint8)
        input_end_mask_mat = torch.concat((input_mask_mat, end_mask), dim=-1)
        # 对矩阵mask排序，以排序获得的下标作为高程
        sort_input, sort_index = torch.sort(
            input_end_mask_mat, dim=-1, descending=True, stable=True
        )
        # sort_input为1代表高程有效,否则得0表示无效高程点，相乘获得点云矩阵形式
        # sort_index+1,把0留给填充点
        input_mat = sort_input * (sort_index + 1)
        # 裁剪到指定长度
        input_mat = input_mat[:, :, : self.max_z_voxel]
        
        return (
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
        )

    def data_enhance(
        self,
        valid_input_len,
        input_mat,
        label_mat,
        force_teach_mat,
        another_valid_input_len,
        another_input_mat,
        another_label_mat,
        another_force_teach_mat,
    ):
        # valid_input_len, input_mat, label_mat, force_teach_mat = self.rotate(
        #     valid_input_len, input_mat, label_mat, force_teach_mat
        # )
        # (
        #     another_valid_input_len,
        #     another_input_mat,
        #     another_label_mat,
        #     another_force_teach_mat,
        # ) = self.rotate(
        #     another_valid_input_len,
        #     another_input_mat,
        #     another_label_mat,
        #     another_force_teach_mat,
        # )

        x_pos = torch.randint(self.area_size[0] // 4, 3 * self.area_size[0] // 4, (1,))[
            0
        ]
        y_pos = torch.randint(self.area_size[1] // 4, 3 * self.area_size[1] // 4, (1,))[
            0
        ]

        rect_1 = (x_pos, y_pos)
        rect_2 = (x_pos, self.area_size[1] - y_pos)
        rect_3 = (self.area_size[0] - x_pos, y_pos)
        rect_4 = (self.area_size[0] - x_pos, self.area_size[1] - y_pos)

        area_1_valid_input_len, area_1_input, area_1_label, area_1_force_teach = (
            self.crop(valid_input_len, input_mat, label_mat, force_teach_mat, rect_1)
        )
        area_2_valid_input_len, area_2_input, area_2_label, area_2_force_teach = (
            self.crop(
                another_valid_input_len,
                another_input_mat,
                another_label_mat,
                another_force_teach_mat,
                rect_2,
            )
        )
        area_3_valid_input_len, area_3_input, area_3_label, area_3_force_teach = (
            self.crop(
                another_valid_input_len,
                another_input_mat,
                another_label_mat,
                another_force_teach_mat,
                rect_3,
            )
        )
        area_4_valid_input_len, area_4_input, area_4_label, area_4_force_teach = (
            self.crop(valid_input_len, input_mat, label_mat, force_teach_mat, rect_4)
        )

        input = self.montage(
            area_1_input, area_2_input, area_3_input, area_4_input, rect_1
        )
        label = self.montage(
            area_1_label, area_2_label, area_3_label, area_4_label, rect_1
        )
        force_teach = self.montage(
            area_1_force_teach,
            area_2_force_teach,
            area_3_force_teach,
            area_4_force_teach,
            rect_1,
        )
        valid_len = self.montage(
            area_1_valid_input_len,
            area_2_valid_input_len,
            area_3_valid_input_len,
            area_4_valid_input_len,
            rect_1,
        )
        l = torch.sum(valid_len, dim=0).max()
        return (
            valid_len[:l, :, :],
            input[:l, :, :],
            label[:l, :, :],
            force_teach[:l, :, :],
        )

    def crop(self, valid_input_len, input_mat, label_mat, force_teach_mat, rect_size):
        rect = transforms.RandomCrop.get_params(input_mat, rect_size)
        input = transforms.functional.crop(input_mat, *rect)
        label = transforms.functional.crop(label_mat, *rect)
        force_teach = transforms.functional.crop(force_teach_mat, *rect)
        valid_len = transforms.functional.crop(valid_input_len, *rect)
        return (valid_len, input, label, force_teach)

    def rotate(self, valid_input_len, input_mat, label_mat, force_teach_mat):
        angle = random.uniform(0, 360)
        input = transforms.functional.rotate(
            input_mat, 
            angle, 
            interpolation=transforms.InterpolationMode.NEAREST, 
            fill=0
        )
        label = transforms.functional.rotate(
            label_mat, 
            angle, 
            interpolation=transforms.InterpolationMode.NEAREST, 
            fill=0
        )
        force_teach = transforms.functional.rotate(
            force_teach_mat,
            angle,
            interpolation=transforms.InterpolationMode.NEAREST,
            fill=0,
        )
        valid_len = transforms.functional.rotate(
            valid_input_len,
            angle,
            interpolation=transforms.InterpolationMode.NEAREST,
            fill=0,
        )
        input[0, :, :] = torch.where(
            input[0, :, :] == 0, self.end_token, input[0, :, :]
        )
        return (valid_len, input, label, force_teach)

    def montage(self, area_1, area_2, area_3, area_4, rect):
        output = torch.zeros((self.num_z + 1, self.area_size[0], self.area_size[1]))
        output[: area_1.shape[0], : rect[0], : rect[1]] = area_1
        output[: area_2.shape[0], : rect[0], rect[1] :] = area_2
        output[: area_3.shape[0], rect[0] :, : rect[1]] = area_3
        output[: area_4.shape[0], rect[0] :, rect[1] :] = area_4

        return output.to(torch.int64)

    def generate_mat(self, data, file_name):

        # 有标签数据才读取标签，否则只加载数据
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
        ) = self.preprocess_data(data)

        label_path = os.path.join(self.label_path, f"{file_name}.label")
        teach_path = os.path.join(self.label_path, f"{file_name}.force")
        # 如果有标签数据，直接读取，否则生成
        if os.path.exists(label_path) and os.path.exists(teach_path):
            label_mat = torch.load(label_path)
            force_teach_mat = torch.load(teach_path)
        elif self.random:
            label_mat, force_teach_mat = self.generate_label(
                data, sort_index, full_indices, num_rows, num_cols
            )
        else:
            label_mat = torch.zeros_like(input_mat)
            force_teach_mat = label_mat
        sort_input = sort_input[:, :, 0 : self.max_z_voxel]

        padding = torch.zeros((sort_input.shape[0], sort_input.shape[1], 1))
        # 丢掉第一个有效长度，从而排除掉结束标签
        sort_input = torch.cat((sort_input[:, :, 1:], padding), dim=-1)
        return (
            sort_input.permute(2, 0, 1).to(torch.int64),
            input_mat.permute(2, 0, 1).to(torch.int64),
            label_mat.permute(2, 0, 1).to(torch.int64),
            force_teach_mat.permute(2, 0, 1).to(torch.int64),
        )

    def generate_label(self, data, sort_index, full_indices, num_rows, num_cols):

        # 对向量索引排序，把相同位置的点排在一起
        sort_indices, order = torch.sort(full_indices, dim=-1)
        # 重新索引点数据
        data_order = data[order]
        # 生成体素的[索引,标签]数组
        label = torch.concat(
            (sort_indices.reshape(-1, 1), data_order[:, -1].reshape(-1, 1)), dim=1
        ).to(torch.int64)
        # 计算每个体素中的点数量dx
        end_idx = torch.where(sort_indices[:-1] != sort_indices[1:])[0] + 1
        end_idx = torch.concat(
            (
                end_idx,
                torch.tensor([len(data)]),
            )
        )
        start_idx = torch.concat((torch.tensor([0]), end_idx[:-1]))
        dx = end_idx - start_idx
        # 分组并求众数
        group = torch.split(label, list(dx.numpy()))
        group_mat = torch.zeros(num_rows * num_cols * self.num_z, dtype=torch.uint8)

        for i in range(len(group)):
            value, _ = torch.mode(group[i], dim=0)
            # 把相应的位置赋值正确的类别
            group_mat[value[0]] = value[1]

        group_mat = group_mat.reshape((num_rows, num_cols, self.num_z))
        end_padding = torch.zeros((num_rows, num_cols, 1), dtype=torch.uint8)
        group_mat = torch.concat((group_mat, end_padding), dim=-1)
        # 获得label_mat标签矩阵
        label_mat = torch.take_along_dim(group_mat, sort_index, dim=-1)
        # 开头拼上开始token
        start_mask = torch.zeros((num_rows, num_cols, 1), dtype=torch.uint8)
        start_mask = torch.fill(start_mask, self.label_start_token)
        force_teach_mat = torch.concat((start_mask, label_mat), dim=-1)

        label_mat = label_mat[:, :, : self.max_z_voxel]
        force_teach_mat = force_teach_mat[:, :, : self.max_z_voxel]
        return label_mat, force_teach_mat

    def collect_las_files(self, file_path):
        """收集文件夹下所有 las 文件路径。"""
        if os.path.isfile(file_path) and file_path.endswith((".las", ".laz")):
            return [file_path]
        elif os.path.isdir(file_path):
            return [f for f in os.listdir(file_path) if f.endswith((".las", ".laz"))]
        else:
            raise ValueError(f"Invalid path: '{file_path}'")


class LasDataLoader(IterableDataset):

    def __init__(
        self,
        data,
        area_size,
        num_z,
        xy_resolution,
        z_resolution,
        max_z_voxel=None,
    ):
        """
        从tensor里加载点云

        :param data: tensor格式的点云
        :param area_size: 每次加载的分块的大小
        :param num_z: 最大高程离散化区段数
        :param xy_resolution: 每个体素平面分辨率（单位：米）
        :param z_resolution: 每个体素平面分辨率（单位：米）
        :param xmax_z_voxel: 每个位置上最长的体素长度
        """
        self.data = data
        self.area_size = area_size
        self.num_z = num_z
        self.max_z_voxel = max_z_voxel
        self.xy_resolution = xy_resolution
        self.z_resolution = z_resolution

        self.end_token = num_z + 1
        self.start_token = num_z + 2
        self.padding_token = 0

        self.random = random
        self.valid_input_mat, self.input_mat = self.generate_mat(data)

    def __iter__(self):
        valid_input_mat = self.valid_input_mat
        input_mat = self.input_mat

        _, num_rows, num_cols = input_mat.shape

        # 计算分块的总行列数
        row_len = num_rows // self.area_size[0] + 1
        col_len = num_cols // self.area_size[1] + 1
        for i in range(row_len):
            for j in range(col_len):

                row_begin = i * self.area_size[0]
                col_begin = j * self.area_size[1]

                row_end = row_begin + self.area_size[0]
                col_end = col_begin + self.area_size[1]

                if col_end > num_cols:
                    col_end = num_cols
                    col_begin = num_cols - self.area_size[1]

                if row_end > num_rows:
                    row_end = num_rows
                    row_begin = num_rows - self.area_size[0]

                valid_input = valid_input_mat[
                    :,
                    row_begin:row_end,
                    col_begin:col_end,
                ]

                valid_len = torch.sum(valid_input != 0, dim=0).max()
                if valid_len == 0:
                    continue
                yield (
                    valid_input_mat[
                        :valid_len,
                        row_begin:row_end,
                        col_begin:col_end,
                    ],
                    input_mat[
                        :valid_len,
                        row_begin:row_end,
                        col_begin:col_end,
                    ],
                    (row_begin, col_begin),
                )

    def preprocess_data(self, data):
        # data = clear_data(data)
        # 计算坐标范围
        max_coord, _ = torch.max(data, dim=0)
        min_coord, _ = torch.min(data, dim=0)
        self.min_coord = min_coord
        boundary = max_coord - min_coord
        # 计算行列数
        num_rows = torch.floor(boundary[0] / self.xy_resolution).to(torch.int64) + 1
        num_cols = torch.floor(boundary[1] / self.xy_resolution).to(torch.int64) + 1
        # 计算每个点的矩阵索引
        x_indices = torch.floor((data[:, 0] - min_coord[0]) / self.xy_resolution).to(
            torch.int64
        )
        y_indices = torch.floor((data[:, 1] - min_coord[1]) / self.xy_resolution).to(
            torch.int64
        )
        z_indices = torch.floor((data[:, 2] - min_coord[2]) / self.z_resolution).to(
            torch.int64
        )
        # 对于高程超出了最大长度num_z的，设置高程为num_z-1

        mask = z_indices > self.num_z - 1
        z_indices[mask] = self.num_z - 1

        # 构建一维索引
        full_indices = (
            x_indices * num_cols * self.num_z + y_indices * self.num_z + z_indices
        )
        # 对每个有效位置的体素，都置1
        input_mask_mat = torch.zeros(
            (num_rows, num_cols, self.num_z), dtype=torch.uint8
        )
        input_mask_mat[x_indices, y_indices, z_indices] = 1

        # 如果没有设定z方向上最大的高程点数,自动计算整个区域内最长的序列长度
        valid_input_len = torch.sum(input_mask_mat, dim=-1)
        self.max_z_voxel = valid_input_len.max() + 1

        # 在最后一个维度上拼接一个结束占位符
        end_mask = torch.ones((num_rows, num_cols, 1), dtype=torch.uint8)
        input_end_mask_mat = torch.concat((input_mask_mat, end_mask), dim=-1)
        # 对矩阵mask排序，以排序获得的下标作为高程
        sort_input, sort_index = torch.sort(
            input_end_mask_mat, dim=-1, descending=True, stable=True
        )
        # sort_input为1代表高程有效,否则得0表示无效高程点，相乘获得点云矩阵形式
        # sort_index+1,把0留给填充点
        input_mat = sort_input * (sort_index + 1)
        # 裁剪到指定长度
        input_mat = input_mat[:, :, : self.max_z_voxel]
        return (
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
        )

    def generate_mat(self, data):
        self.preprocess_info = self.preprocess_data(data)
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
        ) = self.preprocess_info

        sort_input = sort_input[:, :, 0 : self.max_z_voxel]
        padding = torch.zeros((sort_input.shape[0], sort_input.shape[1], 1))
        # 丢掉第一个有效长度，从而排除掉结束标签
        sort_input = torch.cat((sort_input[:, :, 1:], padding), dim=-1)
        return (
            sort_input.permute(2, 0, 1).to(torch.int64),
            input_mat.permute(2, 0, 1).to(torch.int64),
        )
