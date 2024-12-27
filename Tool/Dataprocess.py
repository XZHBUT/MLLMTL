import pywt  # 小波变换的pywt库
import matplotlib.pyplot as plt
import numpy as np
import os
import csv
import torch
import xlrd
from PIL import Image
from matplotlib.backends.backend_agg import FigureCanvasAgg

from scipy.io import loadmat
# 从scikit-learn库中导入preprocessing模块，用于数据预处理，包括标准化
from sklearn import preprocessing
# 读取数据
import pandas as pd
# 从scikit-learn库中导入StratifiedShuffleSplit模块，用于分层随机划分数据集以保持类别比例
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset



def read_OneOH(filepath, ClassNameList):
    '''
    # 获取一种负载文件下的的数据

    :return: dict{} key：类名， Velue：该的全部电波
    '''

    # 获得该文件夹下所有.mat文件名,获取指定文件夹下所有文件的文件名，并将它们存储在filenames变量
    filenames = os.listdir(filepath)

    # 用于从MATLAB文件中提取数据。它遍历了filenames中的所有文件名，加载每个文件中的数据
    # 并将数据存储在一个字典中，字典的键是文件名，值是文件中包含 'DE' 的数据。
    def capture(original_path, ClassNameList):
        files = {}
        for i in filenames:

            ClassName = ''
            for one in ClassNameList:
                if one in i:
                    ClassName = one
                    break
            # 文件路径
            file_path = os.path.join(filepath, i)
            file = loadmat(file_path)
            file_keys = file.keys()
            for key in file_keys:
                if 'DE' in key:
                    files[ClassName] = file[key].ravel()
        return files

    return capture(filepath, ClassNameList)


def SliceDate(data, slice_rate, SampleNumpy, SampleLength):
    '''
    :param data: dict (key:类别名， value： 完整的数据)
    :param slice_rate: 训练集：验证集：测试集
    :param SampleNumpy: 一个类别所拥有的 训练集+验证集+测试集的数量 ps：例如总的样本是 N 训练集0.7 一个负载下的类是：（N/负载数/类别数） * 0.7
    :return: 然后将这些数据存储在Train_Samples和Test_Samples字典中，其中键是文件名，值是切分后的数据
    '''
    keys = data.keys()
    step = int(SampleLength / 2)
    Train_Samples = {}
    Vaild_Samples = {}
    Test_Samples = {}
    # 遍历所有的数据文件
    for i in keys:

        slice_data = data[i]  # 每个键（数据文件）的数据，将其存储在 slice_data 变量中
        samp_train = int(SampleNumpy * slice_rate[0])  # 计算每一类用于训练的样本数量
        samp_vaild = int(SampleNumpy * slice_rate[1])  # 计算每一类用于验证的样本数量
        samp_test = int(SampleNumpy * slice_rate[2])  # 计算每一类用于测试的样本数量
        Train_sample = []
        Vaild_sample = []
        Test_sample = []  # 两个空列表 Train_sample 和 Test_sample 被初始化，用于存储划分后的训练集和测试集样本数据。
        # 遍历训练集中的样本，滑动窗口技巧，收集samp_train个
        for j in range(samp_train):
            sample = slice_data[j * step: j * step + SampleLength]
            Train_sample.append(sample)

        # 抓取验证数据，滑动窗口技巧
        for h in range(samp_vaild):
            sample = slice_data[
                     samp_train * step + SampleLength + h * step: samp_train * step + SampleLength + h * step + SampleLength]
            Vaild_sample.append(sample)

        local = samp_train + samp_vaild
        # 抓取测试数据
        for h in range(samp_test):
            sample = slice_data[
                     local * step + SampleLength + h * step: local * step + SampleLength + h * step + SampleLength]
            Test_sample.append(sample)

        Train_Samples[i] = Train_sample
        Vaild_Samples[i] = Vaild_sample
        Test_Samples[i] = Test_sample  # 训练集和测试集样本被存储在 Train_Samples 和 Test_Samples 字典中，以数据文件名 i 作为键

    return Train_Samples, Vaild_Samples, Test_Samples


# 输入字典，按文件读取字典，拼到一个数组里面
# 标签打上0，1，2，3.。。。代表不同文件，也就是不同情况的label
# 返回值是两个数组，一个样本，一个标签
def add_labels(data_Dict, ClassNameList):
    '''

    :param data_Dict: 一个负载下的字典
    :param ClassNameList:  类名
    :return:  两个列表
    '''
    X = []
    Y = []
    label = 0
    for i in ClassNameList:
        x = data_Dict[i]
        X += x
        lenx = len(x)
        Y += [label] * lenx
        label += 1
    return X, Y

    # 主要用于将训练集和测试集进行标准化处理 均值变为0，标准差变为1
    #  Train_X, Test_X二维数组


def scalar_stand(Train_X, Vaild_X, Test_X):
    data_all = np.vstack((Train_X, Vaild_X, Test_X))  # 将训练集和测试集的特征数据堆叠在一起，创建了一个包含所有数据的 data_all
    Standard = preprocessing.StandardScaler()
    Train_X = Standard.fit_transform(Train_X)
    Vaild_X = Standard.transform(Vaild_X)
    Test_X = Standard.transform(Test_X)
    '''
    使用训练集的均值和标准差来对训练集和测试集的数据进行标准化的原因如下：

    避免信息泄漏（Data Leakage）：在机器学习中，我们通常将数据分为训练集和测试集，以评估模型的性能。如果在标准化时使用了整个数据集（包括测试集）的均值和标准差，那么在某种程度上会将测试集的信息引入到训练过程中，这可能导致过于乐观的性能评估。因此，应该仅使用训练集的统计信息，以避免信息泄漏。

    模拟真实世界：在实际应用中，我们通常只能在训练阶段获取有限数量的数据用于模型训练，而测试数据是未来模型将要处理的新数据。因此，标准化过程应该模拟实际情况，即模型在训练期间只能访问训练数据的统计信息，然后将训练数据的分布应用于测试数据。

    保持一致性：使用训练集的统计信息对训练集和测试集进行标准化可以确保两者之间的一致性。这意味着模型在训练和测试期间看到的数据分布是相同的，从而使得模型更容易泛化到未见过的数据。
    '''
    return Train_X, Vaild_X, Test_X


def read_OrginCWRU(filepath, SampleNum=200, SampleLength=1024, normal=False, Rate=[0.7, 0.2, 0.1]):
    ClassName = ['B007', 'B014', 'B021', 'IR007', 'IR014', 'IR021', 'OR007', 'OR014', 'OR021', 'normal']

    # 获取指定路径下的所有文件夹名称
    folder_names = [f for f in os.listdir(filepath) if os.path.isdir(os.path.join(filepath, f))]

    Train_X, Train_Y = [], []
    Valid_X, Valid_Y = [], []
    Test_X, Test_Y = [], []
    # 按照不同负载条件分别读取
    for OneOH_File in folder_names:
        OneOH_Dict = read_OneOH(filepath + '/' + OneOH_File, ClassName)
        # OneOH_Dict: 字典， key：一个负载下的所有类别， value： 该类别下的所有数据（未切片）

        Train_Samples_OneOH, Vaild_Samples_OneOH, Test_Samples_OneOH = SliceDate(OneOH_Dict, Rate, int(SampleNum / 4),
                                                                                 SampleLength)
        # Train_Samples_OneOH： 字典 key：一个负载下的所有类别， value： 二维数组，num x 1024
        Train_X_OneOH, Train_Y_OneOH = add_labels(Train_Samples_OneOH, ClassName)
        Valid_X_OneOH, Valid_Y_OneOH = add_labels(Vaild_Samples_OneOH, ClassName)
        Test_X_OneOH, Test_Y_OneOH = add_labels(Test_Samples_OneOH, ClassName)

        Train_X += Train_X_OneOH
        Train_Y += Train_Y_OneOH
        Valid_X += Valid_X_OneOH
        Valid_Y += Valid_Y_OneOH
        Test_X += Test_X_OneOH
        Test_Y += Test_Y_OneOH

    # 训练数据/测试数据 是否标准化.
    if normal:
        Train_X, Valid_X, Test_X = scalar_stand(Train_X, Valid_X, Test_X)

    Train_X = np.asarray(Train_X)
    Valid_X = np.asarray(Valid_X)
    Test_X = np.asarray(Test_X)

    Test_Y = np.asarray(Test_Y, dtype=np.int32)
    Valid_Y = np.asarray(Valid_Y, dtype=np.int32)
    Train_Y = np.asarray(Train_Y, dtype=np.int32)

    return Train_X, Train_Y, Valid_X, Valid_Y, Test_X, Test_Y

def SliceDate_Shot(data, Train_shot=3,Test_shot=10, SampleLength=1024):

    keys = data.keys()
    step = int(SampleLength / 2)
    Train_Samples = {}
    Test_Samples = {}
    # 遍历所有的数据文件
    for i in keys:
        slice_data = data[i]  # 每个键（数据文件）的数据，将其存储在 slice_data 变量中
        samp_train = Train_shot  # 计算每一类用于训练的样本数量
        samp_test = Test_shot  # 计算每一类用于测试的样本数量
        Train_sample = []
        Test_sample = []  # 两个空列表 Train_sample 和 Test_sample 被初始化，用于存储划分后的训练集和测试集样本数据。
        # 遍历训练集中的样本，滑动窗口技巧，收集samp_train个
        for j in range(samp_train):
            sample = slice_data[j * step: j * step + SampleLength]
            Train_sample.append(sample)

        local = samp_train
        # 抓取测试数据
        for h in range(samp_test):
            sample = slice_data[
                     local * step + SampleLength + h * step: local * step + SampleLength + h * step + SampleLength]
            Test_sample.append(sample)

        Train_Samples[i] = Train_sample
        Test_Samples[i] = Test_sample  # 训练集和测试集样本被存储在 Train_Samples 和 Test_Samples 字典中，以数据文件名 i 作为键

    return Train_Samples, Test_Samples


def read_OrginCWRU_fromOneOH_fewshot(filepath, Train_shot=3,Test_shot=10, SampleLength=1024, normal=False, Select_OH='0HP', Select_shot=None):
    ClassName = ['B007', 'B014', 'B021', 'IR007', 'IR014', 'IR021', 'OR007', 'OR014', 'OR021', 'normal']
    if Select_shot == None:
        Select_shot = [0, 1, 2,3,4,5,6,7,8,9]
        Select_ClassName_shot = [ClassName[i] for i in Select_shot]
    else:
        Select_ClassName_shot = [ClassName[i] for i in Select_shot]

    # 获取指定路径下的所有文件夹名称
    folder_names = [f for f in os.listdir(filepath) if os.path.isdir(os.path.join(filepath, f))]
    OneOH_Dict = read_OneOH(filepath + '/' + Select_OH, ClassName)

    Train_Samples_OneOH,  Test_Samples_OneOH = SliceDate_Shot(OneOH_Dict, Train_shot=Train_shot,Test_shot=Test_shot,SampleLength=SampleLength)
    Select_Train_Samples_OneOH = {key: Train_Samples_OneOH[key] for key in Select_ClassName_shot}
    Select_Test_Samples_OneOH = {key: Test_Samples_OneOH[key] for key in Select_ClassName_shot}

    Train_X_OneOH, Train_Y_OneOH = add_labels(Select_Train_Samples_OneOH, Select_ClassName_shot)
    Test_X_OneOH, Test_Y_OneOH = add_labels(Select_Test_Samples_OneOH, Select_ClassName_shot)


    # 训练数据/测试数据 是否标准化.

    Train_X = np.asarray(Train_X_OneOH)
    Test_X = np.asarray(Test_X_OneOH)

    Test_Y = np.asarray(Test_Y_OneOH, dtype=np.int32)
    Train_Y = np.asarray(Train_Y_OneOH, dtype=np.int32)
    return Train_X, Train_Y,  Test_X, Test_Y




##################################### 江南大学

def extract_data_from_JNU_folder(folder_path):
    '''
    返回是一个字典，key是文件名不带后缀，value是的从第17行第一列数据的数组
    '''
    # 初始化一个空字典来存储结果
    result_dict = {}

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            # 构建完整的文件路径
            file_path = os.path.join(folder_path, filename)

            # 读取 CSV 文件，并只选择第一列数据，并指定分隔符
            data = pd.read_csv(file_path, skiprows=16, usecols=[0])

            # 从文件名中提取不带后缀的文件名作为字典的键
            file_name_without_extension = os.path.splitext(filename)[0]

            # 将数据存储到结果字典中
            result_dict[file_name_without_extension] = data.values.flatten()

    return result_dict


def extract_data(data_dict, Classname, SampleNumpy, SampleLength):
    result_dict = {}

    for classname in Classname:
        combined_data = []

        for filename, data in data_dict.items():
            if classname in filename:
                # 使用滑动窗口提取信息
                step = int(SampleLength / 2)
                sample_onefile = []
                for j in range(SampleNumpy):
                    sample = data[j * step: j * step + SampleLength]
                    sample_onefile.append(sample)

                combined_data.extend(sample_onefile)

        result_dict[classname] = combined_data

    return result_dict


def split_data_JNU(data_dict, train_ratio=0.7, valid_ratio=0.2, test_ratio=0.1):
    # 获取所有类别
    categories = list(data_dict.keys())

    # 初始化用于存储划分后数据的列表
    Train_X, Train_Y, Valid_X, Valid_Y, Test_X, Test_Y = [], [], [], [], [], []

    # 分配标签
    label_mapping = {category: i for i, category in enumerate(categories)}

    for category, label in label_mapping.items():
        samples = data_dict[category]
        num_samples = len(samples)
        num_train = int(num_samples * train_ratio)
        num_valid = int(num_samples * valid_ratio)
        num_test = int(num_samples * test_ratio)

        # 划分数据
        train_samples = samples[:num_train]
        valid_samples = samples[num_train:num_train + num_valid]
        test_samples = samples[num_train + num_valid:num_train + num_valid + num_test]

        # 添加到相应的列表中
        Train_X.extend(train_samples)
        Train_Y.extend([label] * len(train_samples))
        Valid_X.extend(valid_samples)
        Valid_Y.extend([label] * len(valid_samples))
        Test_X.extend(test_samples)
        Test_Y.extend([label] * len(test_samples))

    return Train_X, Train_Y, Valid_X, Valid_Y, Test_X, Test_Y


def Read_Data_From_JNU(filepath, SampleNum=500, SampleLength=1024, normal=False, Rate=None):
    # 读取路径下所有csv文件，返回字典 key：文件名 value：所有数据
    if Rate is None:
        Rate = [0.7, 0.2, 0.1]
    data_dict_everyfile = extract_data_from_JNU_folder(filepath)

    # 按类 样本数量 数据长度 返回字典
    classname = ['ib', 'n', 'ob', 'tb']
    data_dict_everyclass = extract_data(data_dict_everyfile, classname, SampleNum // 3, SampleLength)

    # 切分数据集
    Train_X, Train_Y, Valid_X, Valid_Y, Test_X, Test_Y = split_data_JNU(data_dict_everyclass, Rate[0], Rate[1], Rate[2])

    # 训练数据/测试数据 是否标准化.
    if normal:
        Train_X, Valid_X, Test_X = scalar_stand(Train_X, Valid_X, Test_X)

    Train_X = np.asarray(Train_X)
    Valid_X = np.asarray(Valid_X)
    Test_X = np.asarray(Test_X)

    Test_Y = np.asarray(Test_Y, dtype=np.int32)
    Valid_Y = np.asarray(Valid_Y, dtype=np.int32)
    Train_Y = np.asarray(Train_Y, dtype=np.int32)

    return Train_X, Train_Y, Valid_X, Valid_Y, Test_X, Test_Y

######################################## 东南大学

def min_max_normalize(data):
    """
    对每个通道进行最大最小归一化。

    参数:
    - data: 一个二维NumPy数组，形状为(L, 8)，代表8个通道的序列。

    返回:
    - 归一化后的数据，形状同样为(L, 8)。
    """
    # 初始化一个和原数据形状相同的数组来存放归一化后的数据

    data = data.transpose()
    normalized_data = np.zeros(data.shape)

    # 对每个通道进行最大最小归一化
    for i in range(data.shape[0]):  # 遍历每个通道
        channel_min = data[i, :].min()  # 计算当前通道的最小值
        channel_max = data[i, :].max()  # 计算当前通道的最大值

        # 执行最大最小归一化
        normalized_data[i, :] = (data[i, :] - channel_min) / (channel_max - channel_min)

    normalized_data = normalized_data.transpose()

    return normalized_data

import numpy as np

def standardize(data):
    """
    对每个通道进行标准化（Z-score 标准化）。

    参数:
    - data: 一个二维NumPy数组，形状为(L, 8)，代表8个通道的序列。

    返回:
    - 标准化后的数据，形状同样为(L, 8)。
    """
    # 转置数据以方便逐通道操作
    data = data.transpose()
    standardized_data = np.zeros(data.shape)

    # 对每个通道进行标准化
    for i in range(data.shape[0]):  # 遍历每个通道
        channel_mean = data[i, :].mean()  # 计算当前通道的均值
        channel_std = data[i, :].std()    # 计算当前通道的标准差

        # 执行标准化
        standardized_data[i, :] = (data[i, :] - channel_mean) / channel_std

    # 转置回原始形状
    standardized_data = standardized_data.transpose()

    return standardized_data


def Read_csv_Data(file_path, row_num=102400, lienum=8, isTime=True):
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        data_started = False
        data_rows = []
        row_count = 0

        for row in reader:
            # 找到数据开始的标记
            if row and row[0].strip().lower() == 'data':
                data_started = True
                continue

            # 从标记行开始读取数据
            if data_started and row:
                # 检查是否所有数据都在第一列，且使用\t分隔
                if len(row) == 1 and '\t' in row[0]:
                    # 如果是，按\t分割字符串
                    row = row[0].split('\t')

                # 处理前8列数据，对于空字符串使用0.0填充
                if isTime:
                    processed_row = [float(value) if value else 0.0 for value in row[:lienum]]
                else:
                    processed_row = [float(value) if value else 0.0 for value in row[1:lienum]]
                data_rows.append(processed_row)
                row_count += 1  # 更新行数计数器

            # 如果达到了10,000行，就停止读取
            if row_count == row_num:
                break

        # # 转置二维数组以匹配要求的格式[8, L]
        # data_rows_transposed = list(zip(*data_rows))
        return data_rows


def Biuld_Sample(Data_list, SampleNumpy, SampleLength, step=None):
    Sample_List = []
    if step == None:
        step = int(SampleLength / 2)

    for j in range(SampleNumpy):
        sample = Data_list[j * step: j * step + SampleLength]
        Sample_List.append(list(map(list, zip(*sample))))

    # print(Sample_List[0][-1])
    return Sample_List


def Read_Data_From_SEU(filepath, SampleNum=200, SampleLength=1024, Rate=[0.7, 0.2, 0.1], stander=False):
    # 默认取第4列
    print("Creating Data Sets.....SEU")
    if Rate is None:
        Rate = [0.7, 0.2, 0.1]

    Data_class = {}
    # 轴承故障数据
    Bearing_class_name = ['ball', 'comb', 'health', 'inner', 'outer']
    Bearing_dilepath = filepath + '/' + 'bearingset'
    # 获取文件夹下所有的文件名
    Bearing_file_names = [f for f in os.listdir(Bearing_dilepath)]
    # print(Bearing_file_names)
    SampleNum_onescv = int(SampleNum / 2)
    for Bear_fault_i in Bearing_class_name:
        Data_class[Bear_fault_i] = []
        for Bearing_csv_i in Bearing_file_names:
            if Bear_fault_i in Bearing_csv_i:
                # 文件路径
                file_path_csv = os.path.join(Bearing_dilepath, Bearing_csv_i)

                Data_csv_rows = Read_csv_Data(file_path_csv, row_num=int(
                    SampleLength + ((SampleNum_onescv - 1) * 0.5 * SampleLength)),
                                              lienum=8, isTime=True)  # 这个时候还是按行存储的数据
                Data_csv_rows = np.array(Data_csv_rows)
                if stander:
                    Data_csv_rows = standardize(Data_csv_rows)
                Bearing_csv_i_Sample = Biuld_Sample(Data_csv_rows, SampleNum_onescv, SampleLength)  # 数据变成数据集
                Data_class[Bear_fault_i].extend(Bearing_csv_i_Sample)  # 合并同一种类别的数据集

    # 齿轮故障数据
    Gear_class_name = ['Chipped', 'Miss', 'Root', 'Surface']
    Gear_dilepath = filepath + '/' + 'gearset'
    # 获取文件夹下所有的文件名
    Gear_file_names = [f for f in os.listdir(Gear_dilepath)]
    # print(Gear_file_names)
    for Gear_fault_i in Gear_class_name:
        Data_class[Gear_fault_i] = []
        for Gear_csv_i in Gear_file_names:
            if Gear_fault_i in Gear_csv_i:
                # 文件路径
                file_path_csv = os.path.join(Gear_dilepath, Gear_csv_i)
                Data_csv_rows = Read_csv_Data(file_path_csv, row_num=int(
                    SampleLength + ((SampleNum_onescv - 1) * 0.5 * SampleLength)),
                                              lienum=8, isTime=True)  # 这个时候还是按行存储的数据
                Data_csv_rows = np.array(Data_csv_rows)
                if stander:
                    Data_csv_rows = standardize(Data_csv_rows)
                Gear_csv_i_Sample = Biuld_Sample(Data_csv_rows, int(SampleNum / 2), SampleLength)  # 数据变成数据集
                Data_class[Gear_fault_i].extend(Gear_csv_i_Sample)  # 合并同一种类别的数据集

    Train_X = []
    Train_Y = []
    Valid_X = []
    Valid_Y = []
    Test_X = []
    Test_Y = []
    OneClassTrainNum = int(SampleNum * Rate[0])
    OneClassVaildNum = int(SampleNum * Rate[1])
    OneClassTestNum = int(SampleNum * Rate[2])
    for index, (key, value) in enumerate(Data_class.items()):
        Train_X += Data_class[key][0: OneClassTrainNum]
        Train_Y += [index] * OneClassTrainNum

        Valid_X += Data_class[key][OneClassTrainNum: OneClassTrainNum + OneClassVaildNum]
        Valid_Y += [index] * OneClassVaildNum

        Test_X += Data_class[key][
                  OneClassTrainNum + OneClassVaildNum: OneClassTrainNum + OneClassVaildNum + OneClassTestNum]
        Test_Y += [index] * OneClassTestNum

    Train_X = np.asarray(Train_X)
    Valid_X = np.asarray(Valid_X)
    Test_X = np.asarray(Test_X)

    Test_Y = np.asarray(Test_Y, dtype=np.int32)
    Valid_Y = np.asarray(Valid_Y, dtype=np.int32)
    Train_Y = np.asarray(Train_Y, dtype=np.int32)


    return Train_X[:, 4, :], Train_Y, Valid_X[:, 4, :], Valid_Y, Test_X[:, 4, :], Test_Y

########################################## HUST数据集
def Read_xls_data(file_path, row_num=102400, isTime=True):
    # 打开Excel文件
    workbook = xlrd.open_workbook(file_path)
    # 选择第一个工作表
    sheet = workbook.sheet_by_index(0)

    data_started = False
    data_rows = []
    row_count = 0

    # 遍历工作表中的每一行
    for i in range(sheet.nrows):
        row = sheet.row_values(i)

        # 检查是否是数据开始的标记行
        if row and str(row[0]).strip().lower() == 'data':
            data_started = True
            continue

        # 从标记行之后开始处理数据
        if data_started and row:
            # 检查是否所有数据都在第一列，且使用\t分隔
            if len(row) == 1 and '\t' in str(row[0]):
                # 如果是，按\t分割字符串
                row = str(row[0]).split('\t')

            # 处理前8列数据，对于空字符串使用0.0填充
            if isTime:
                processed_row = [float(value) if value else 0.0 for value in row[:5]]
            else:
                processed_row = [float(value) if value else 0.0 for value in row[1:5]]
            data_rows.append(processed_row)
            row_count += 1  # 更新行数计数器

            # 如果达到了指定的行数，就停止读取
            if row_count == row_num:
                break

    return data_rows


def Read_txt_from(file_path, num_rows=102400, lienum=5, isTime=True):
    data_started = False
    data_rows = []
    with open(file_path, 'r') as file:
        for line in file:
            # 找到数据部分的开始
            if line.strip() == "Time (seconds) and Data Channels":
                data_started = True
                continue

            # 从数据部分开始读取数据
            if data_started:
                # 分割行中的每个数据项
                row = line.strip().split()
                if row:  # 确保不是空行
                    # 选择指定的列范围（注意Python中列表切片不包括end_col索引的元素）
                    if isTime:
                        selected_row = row[:lienum]
                    else:
                        selected_row = row[1:lienum]
                    data_rows.append([float(x) for x in selected_row])  # 将每个项转换为float
                    if len(data_rows) == num_rows:  # 只读取指定的行数
                        break
    return data_rows


def Read_Data_From_HUST(filepath, SampleNum=200, SampleLength=1024, Rate=None, stander=True, lienum=5, isTime=True):
    print("Creating Data Sets.....HUST")
    if Rate is None:
        Rate = [0.7, 0.2, 0.1]

    Data_class = {}
    # 轴承故障数据
    Bearing_class_name = ['0.5X_B', '0.5X_C', '0.5X_I', '0.5X_O', 'B', 'C', 'I', 'O', 'H']
    Bearing_dilepath = filepath + '/' + 'bearing'
    # 获取文件夹下所有的文件名
    Bearing_file_names = [f for f in os.listdir(Bearing_dilepath)]
    Bearing_file_type = [0 for f in os.listdir(Bearing_dilepath)]
    # print(Bearing_file_names)
    # print(Bearing_file_type)
    SampleNum_onescv = int(SampleNum / 1)
    for Bear_fault_i in Bearing_class_name:
        Data_class[Bear_fault_i] = []
        for Bearing_csv_i in Bearing_file_names:
            if Bear_fault_i in Bearing_csv_i and Bearing_file_type[Bearing_file_names.index(Bearing_csv_i)] == 0:
                Bearing_file_type[Bearing_file_names.index(Bearing_csv_i)] = 1

                file_path_csv = os.path.join(Bearing_dilepath, Bearing_csv_i)

                Data_csv_rows = Read_csv_Data(file_path_csv, row_num=int(
                    SampleLength + ((SampleNum_onescv - 1) * 0.5 * SampleLength)),
                                              lienum=lienum, isTime=True)  # 这个时候还是按行存储的数据
                Data_csv_rows = np.array(Data_csv_rows)
                if stander:
                    Data_csv_rows = standardize(Data_csv_rows)
                Bearing_csv_i_Sample = Biuld_Sample(Data_csv_rows, SampleNum_onescv, SampleLength)  # 数据变成数据集
                Data_class[Bear_fault_i].extend(Bearing_csv_i_Sample)  # 合并同一种类别的数据集

    # 齿轮故障数据
    Gear_class_name = ['B_', 'M_']
    Gear_dilepath = filepath + '/' + 'gearbox'
    # 获取文件夹下所有的文件名
    Gear_file_names = [f for f in os.listdir(Gear_dilepath)]

    for Gear_fault_i in Gear_class_name:
        Data_class[Gear_fault_i] = []
        for Gear_csv_i in Gear_file_names:
            if Gear_fault_i in Gear_csv_i:
                # print(Gear_csv_i)
                # 文件路径
                file_path_csv = os.path.join(Gear_dilepath, Gear_csv_i)
                Data_csv_rows = Read_txt_from(file_path_csv, num_rows=int(
                    SampleLength + ((SampleNum_onescv - 1) * 0.5 * SampleLength)),
                                              lienum=lienum, isTime=True)  # 这个时候还是按行存储的数据
                Data_csv_rows = np.array(Data_csv_rows)
                if stander:
                    Data_csv_rows = standardize(Data_csv_rows)
                Gear_csv_i_Sample = Biuld_Sample(Data_csv_rows, int(SampleNum / 1), SampleLength)  # 数据变成数据集
                Data_class[Gear_fault_i].extend(Gear_csv_i_Sample)  # 合并同一种类别的数据集

    Train_X = []
    Train_Y = []
    Valid_X = []
    Valid_Y = []
    Test_X = []
    Test_Y = []

    OneClassTrainNum = int(SampleNum * Rate[0])
    OneClassVaildNum = int(SampleNum * Rate[1])
    OneClassTestNum = int(SampleNum * Rate[2])
    for index, (key, value) in enumerate(Data_class.items()):
        Train_X += Data_class[key][0: OneClassTrainNum]
        Train_Y += [index] * OneClassTrainNum

        Valid_X += Data_class[key][OneClassTrainNum: OneClassTrainNum + OneClassVaildNum]
        Valid_Y += [index] * OneClassVaildNum

        Test_X += Data_class[key][
                  OneClassTrainNum + OneClassVaildNum: OneClassTrainNum + OneClassVaildNum + OneClassTestNum]
        Test_Y += [index] * OneClassTestNum

    Train_X = np.asarray(Train_X)
    Valid_X = np.asarray(Valid_X)
    Test_X = np.asarray(Test_X)

    Test_Y = np.asarray(Test_Y, dtype=np.int32)
    Valid_Y = np.asarray(Valid_Y, dtype=np.int32)
    Train_Y = np.asarray(Train_Y, dtype=np.int32)


    return Train_X[:, 2, :], Train_Y, Valid_X[:, 2, :], Valid_Y, Test_X[:, 2, :], Test_Y


###################################################### HIT

def Read_npy_from(file_path, one_Sample_Length=1204):
    # 设置随机种子以确保每次运行结果相同
    np.random.seed(42)
    data_all = np.load(file_path, encoding="latin1")  # 加载文件
    # 保留前6个通道
    data = data_all[:, :6, :one_Sample_Length]
    label = data_all[0, 7, 0]
    shuffled_data = data[np.random.permutation(int(len(data)))]
    # data (N, )
    return shuffled_data, label


def Read_Data_From_HIT(filepath, SampleNum=200, SampleLength=1024, Rate=None):
    print("Creating Data Sets.....HIT")
    if Rate is None:
        Rate = [0.7, 0.2, 0.1]

    Data_class = {}
    # 获取文件夹下所有的文件名
    file_names = [f for f in os.listdir(filepath)]
    # print(file_names)
    for file_names_i in file_names:
        file_path_csv = os.path.join(filepath, file_names_i)
        data, label = Read_npy_from(file_path_csv,one_Sample_Length=SampleLength)
        if str(label) in Data_class:

            Data_class[str(label)] = np.concatenate((Data_class[str(label)], data), axis=0)
        else:
            Data_class[str(label)] = data


    Train_X = []
    Train_Y = []
    Valid_X = []
    Valid_Y = []
    Test_X = []
    Test_Y = []

    OneClassTrainNum = int(SampleNum * Rate[0])
    OneClassVaildNum = int(SampleNum * Rate[1])
    OneClassTestNum = int(SampleNum * Rate[2])

    for index, (key, value) in enumerate(Data_class.items()):
        Train_X += Data_class[key][0: OneClassTrainNum].tolist()
        Train_Y += [index] * OneClassTrainNum

        Valid_X += Data_class[key][OneClassTrainNum: OneClassTrainNum + OneClassVaildNum].tolist()
        Valid_Y += [index] * OneClassVaildNum

        Test_X += Data_class[key][
                  OneClassTrainNum + OneClassVaildNum: OneClassTrainNum + OneClassVaildNum + OneClassTestNum].tolist()
        Test_Y += [index] * OneClassTestNum

    print("Created Successfully; Number of channels:6;Class:{};SampleNum:{};SampleLength:{}".format(
        len(Data_class), SampleNum, SampleLength))

    Train_X = np.asarray(Train_X)
    Valid_X = np.asarray(Valid_X)
    Test_X = np.asarray(Test_X)

    Test_Y = np.asarray(Test_Y, dtype=np.int32)
    Valid_Y = np.asarray(Valid_Y, dtype=np.int32)
    Train_Y = np.asarray(Train_Y, dtype=np.int32)

    return Train_X[:, 5, :], Train_Y, Valid_X[:, 5, :], Valid_Y, Test_X[:, 5, :], Test_Y





if __name__ == '__main__':

    read_OrginCWRU_fromOneOH_fewshot(filepath='../data/CWRU',Select_shot=[0,1,2])

    # Train_X, Train_Y, Valid_X, Valid_Y, Test_X, Test_Y = read_OrginCWRU(filepath='../data/CWRU',
    #                                                                     SampleLength=1024,
    #                                                                     SampleNum=200,
    #                                                                     normal=False,
    #                                                                     Rate=[0.7, 0.2, 0.1]
    #                                                                     )
    # #
    # # Train_X = addNoiseBatch(Train_X, -4)
    # # Valid_X = addNoiseBatch(Valid_X, -4)
    # # Test_X = addNoiseBatch(Test_X, -4)
    # print(Train_X.shape)
    # print(Train_Y.shape)
    # print(Valid_X.shape)
    # print(Valid_Y.shape)
    # print(Test_X.shape)
    # print(Test_Y.shape)
    # import matplotlib.pyplot as plt
    # #
    # # # 假设 Train_X 的第一行是你想要绘制的信号
    # signal_to_plot = Train_X[5, :]
    #
    # # 设置图形的颜色
    # color = 'b'  # 这里使用蓝色，你可以根据需要更改颜色
    #
    # # 设置图形大小
    # plt.figure(figsize=(10, 6))
    #
    # # 绘制信号图
    # # plt.plot(signal_to_plot, color=color)
    # # plt.title('Signal Plot')
    # # plt.xlabel('Sample Index')
    # # plt.ylabel('Amplitude')
    # # plt.show()
    #
    #
    #
    # Train_X, Train_Y, Valid_X, Valid_Y, Test_X, Test_Y = read_OrginJNU(filepath='../data/JNU',
    #                                                                     SampleLength=1024,
    #                                                                     SampleNum=200,
    #                                                                     normal=False,
    #                                                                     Rate=[0.7, 0.2, 0.1]
    #                                                                     )
    #
    # # Train_X = addPinkNoiseBatch(Train_X, -4)
    # # Valid_X = addPinkNoiseBatch(Valid_X, -4)
    # # Test_X = addPinkNoiseBatch(Test_X, -4)
    # print(Train_X.shape)
    # print(Train_Y.shape)
    # print(Valid_X.shape)
    # print(Valid_Y.shape)
    # print(Test_X.shape)
    # print(Test_Y.shape)
    #
    #
    # folder_path = "../data/SEU"
    #
    # Train_X, Train_Y, Valid_X, Valid_Y, Test_X, Test_Y = Read_Data_From_SEU(folder_path, SampleNum=200, SampleLength=1024, Rate=[0.7, 0.2, 0.1], isMaxMin=False)
    #
    # # 将NumPy数组转换为PyTorch张量，标签使用dtype=torch.int64
    # Train_X = torch.tensor(Train_X, dtype=torch.float32)
    # Train_Y = torch.tensor(Train_Y, dtype=torch.int64)
    # Valid_X = torch.tensor(Valid_X, dtype=torch.float32)
    # Valid_Y = torch.tensor(Valid_Y, dtype=torch.int64)
    # Test_X = torch.tensor(Test_X, dtype=torch.float32)
    # Test_Y = torch.tensor(Test_Y, dtype=torch.int64)
    #
    # print("SEU",Train_X.shape)
    #
    Train_X, Train_Y, Valid_X, Valid_Y, Test_X, Test_Y = Read_Data_From_HUST('../data/HUST', SampleNum=200, SampleLength=1024, Rate=None,  lienum=5, isTime=True)

    Train_X = torch.tensor(Train_X, dtype=torch.float32)
    Train_Y = torch.tensor(Train_Y, dtype=torch.int64)
    Valid_X = torch.tensor(Valid_X, dtype=torch.float32)
    Valid_Y = torch.tensor(Valid_Y, dtype=torch.int64)
    Test_X = torch.tensor(Test_X, dtype=torch.float32)
    Test_Y = torch.tensor(Test_Y, dtype=torch.int64)
    print(Train_X.shape)
    #
    # Train_X, Train_Y, Valid_X, Valid_Y, Test_X, Test_Y = Read_Data_From_HIT('../data/HIT', SampleNum=200, SampleLength=1024, Rate=None)
    #
    # Train_X = torch.tensor(Train_X, dtype=torch.float32)
    # Train_Y = torch.tensor(Train_Y, dtype=torch.int64)
    # Valid_X = torch.tensor(Valid_X, dtype=torch.float32)
    # Valid_Y = torch.tensor(Valid_Y, dtype=torch.int64)
    # Test_X = torch.tensor(Test_X, dtype=torch.float32)
    # Test_Y = torch.tensor(Test_Y, dtype=torch.int64)
    # print(Train_X.shape)