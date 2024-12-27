import csv
import os
import random
import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat

def Read_npy_from(file_path, one_Sample_Length=1204):
    np.random.seed(42)
    data_all = np.load(file_path, encoding="latin1")
    data = data_all[:, :6, :one_Sample_Length]
    label = data_all[0, 7, 0]
    shuffled_data = data[np.random.permutation(int(len(data)))]
    return shuffled_data, label

def Read_csv_Data(file_path, row_num=102400, lienum=8, isTime=True):
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        data_started = False
        data_rows = []
        row_count = 0
        for row in reader:
            if row and row[0].strip().lower() == 'data':
                data_started = True
                continue
            if data_started and row:
                if len(row) == 1 and '\t' in row[0]:
                    row = row[0].split('\t')
                if isTime:
                    processed_row = [float(value) if value else 0.0 for value in row[:lienum]]
                else:
                    processed_row = [float(value) if value else 0.0 for value in row[1:lienum]]
                data_rows.append(processed_row)
                row_count += 1
            if row_count == row_num:
                break
        return data_rows

def Read_txt_from(file_path, num_rows=102400, lienum=5, isTime=True):
    data_started = False
    data_rows = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip() == "Time (seconds) and Data Channels":
                data_started = True
                continue
            if data_started:
                row = line.strip().split()
                if row:
                    if isTime:
                        selected_row = row[:lienum]
                    else:
                        selected_row = row[1:lienum]
                    data_rows.append([float(x) for x in selected_row])
                    if len(data_rows) == num_rows:
                        break
    return data_rows

def Biuld_Sample(Data_list, SampleNumpy, SampleLength, step=None):
    Sample_List = []
    if step == None:
        step = int(SampleLength / 2)
    for j in range(SampleNumpy):
        sample = Data_list[j * step: j * step + SampleLength]
        Sample_List.append(list(map(list, zip(*sample))))
    return Sample_List

def Read_HUST_OneClass_sample(filepath, class_name, SampleLength, SampleNum=200, step=1024):
    if class_name != 'B_' and class_name != 'M_':
        Bearing_dilepath = filepath + '/' + 'bearing'
        Bearing_file_names = [f for f in os.listdir(Bearing_dilepath)]
        SampleNum_onescv = int(SampleNum / 1)
        Bearing_file_type = [0 for f in os.listdir(Bearing_dilepath)]
        for Bearing_csv_i in Bearing_file_names:
            if class_name in Bearing_csv_i and Bearing_file_type[Bearing_file_names.index(Bearing_csv_i)] == 0:
                Bearing_file_type[Bearing_file_names.index(Bearing_csv_i)] = 1
                file_path_csv = os.path.join(Bearing_dilepath, Bearing_csv_i)
                Data_csv_rows = Read_csv_Data(file_path_csv, row_num=int(SampleLength + ((SampleNum_onescv - 1) * 0.5 * SampleLength)),
                                              lienum=5, isTime=True)
                Data_csv_rows = np.array(Data_csv_rows)
                Bearing_csv_i_Sample = Biuld_Sample(Data_csv_rows, SampleNum_onescv, SampleLength, step=step)
                Bearing_csv_i_Sample = np.array(Bearing_csv_i_Sample)[:, 2, :]
                return Bearing_csv_i_Sample
    else:
        Gear_dilepath = filepath + '/' + 'gearbox'
        Gear_file_names = [f for f in os.listdir(Gear_dilepath)]
        SampleNum_onescv = int(SampleNum / 1)
        Bearing_file_type = [0 for f in os.listdir(Gear_dilepath)]
        for Gear_csv_i in Gear_file_names:
            if class_name in Gear_csv_i:
                file_path_csv = os.path.join(Gear_dilepath, Gear_csv_i)
                Data_csv_rows = Read_txt_from(file_path_csv, num_rows=int(SampleLength + ((SampleNum_onescv - 1) * 0.5 * SampleLength)),
                                              lienum=5, isTime=True)
                Data_csv_rows = np.array(Data_csv_rows)
                Gear_csv_i_Sample = Biuld_Sample(Data_csv_rows, SampleNum_onescv, SampleLength, step=step)
                Gear_csv_i_Sample = np.array(Gear_csv_i_Sample)[:, 2, :]
                return Gear_csv_i_Sample

def Read_JNU_OneClass_sample(filepath, class_name, SampleLength, SampleNum=200, step=1024):
    result_dict = {}
    file_name_list = os.listdir(filepath)
    for file_name in file_name_list:
        if file_name.endswith(".csv") and class_name in file_name:
            file_path = os.path.join(filepath, file_name)
            data = pd.read_csv(file_path, skiprows=16, usecols=[0])
            file_name_without_extension = os.path.splitext(file_name)[0]
            result_dict[file_name_without_extension] = data.values.flatten()
    combined_data = []
    for filename, data in result_dict.items():
        sample_onefile = []
        for j in range(SampleNum // 3):
            sample = data[j * step: j * step + SampleLength]
            sample_onefile.append(sample)
        combined_data.extend(sample_onefile)

    combined_data = np.vstack(combined_data)
    return combined_data

def Read_HIT_OneClass_sample(filepath, class_name, SampleLength, SampleNum=200, step=1024):
    if class_name == '0.0':
        file_path_csv = os.path.join(filepath, 'data1.npy')
        data1, label = Read_npy_from(file_path_csv, one_Sample_Length=SampleLength)
        file_path_csv = os.path.join(filepath, 'data2.npy')
        data2, label = Read_npy_from(file_path_csv, one_Sample_Length=SampleLength)
        data = np.concatenate((data1, data2), axis=0)[:SampleNum, 5, :]
        return data
    if class_name == '1.0':
        file_path_csv = os.path.join(filepath, 'data3.npy')
        data1, label = Read_npy_from(file_path_csv, one_Sample_Length=SampleLength)
        file_path_csv = os.path.join(filepath, 'data4.npy')
        data2, label = Read_npy_from(file_path_csv, one_Sample_Length=SampleLength)
        data = np.concatenate((data1, data2), axis=0)[:SampleNum, 5, :]
        return data

    if class_name == '2.0':
        file_path_csv = os.path.join(filepath, 'data5.npy')
        data1, label = Read_npy_from(file_path_csv, one_Sample_Length=SampleLength)
        data = data1[:SampleNum, 5, :]
        return data

def Read_CWRU_OneClass_sample(filepath, class_name, SampleLength, SampleNum=200, step=1024):
    Class_sample_list = []
    for i_OH in range(4):
        OH_name = str(i_OH) + 'HP'
        OneOH_Dict = read_OneOH(filepath + '/' + OH_name, [class_name])
        i_Class_sample = read_CWRU_Sample_from_oneSample_oneOH(OneOH_Dict=OneOH_Dict, Select_Class=class_name,
                                                               Sample_num=int(SampleNum // 4), step=step,
                                                               SampleLength=SampleLength)
        Class_sample_list += i_Class_sample

    combined_data = np.vstack(Class_sample_list)
    return combined_data



import random
import os
import numpy as np
import torch
from scipy.io import loadmat

def get_random_indices_and_values(lst, M):
    if M > len(lst):
        raise ValueError("M")
    indices = random.sample(range(len(lst)), M)
    values = [lst[i] for i in indices]
    return indices, values


def read_OneOH(filepath, ClassNameList):
    filenames = os.listdir(filepath)
    def capture(original_path, ClassNameList):
        files = {}
        for i in filenames:
            ClassName = ''
            for one in ClassNameList:
                if one in i:
                    ClassName = one
                    break
            file_path = os.path.join(filepath, i)
            file = loadmat(file_path)
            file_keys = file.keys()
            for key in file_keys:
                if 'DE' in key:
                    files[ClassName] = file[key].ravel()
        return files
    return capture(filepath, ClassNameList)


def read_CWRU_Sample_from_oneSample_oneOH(OneOH_Dict, Select_Class, Sample_num=200, step=150, SampleLength=2048):
    Class_sample = []
    slice_data_class = OneOH_Dict[Select_Class]
    for i in range(Sample_num):
        sample = slice_data_class[i * step: i * step + SampleLength]
        Class_sample.append(sample)
    return Class_sample


def Build_Task_from_HUST_JNU_HIT(filepath, Task=1000, SampleLength=2048, step=1024, Way=10, Shot=5, QueryShot=5):
    HUST_ClassName = ['HUST_0.5X_B', 'HUST_0.5X_C', 'HUST_0.5X_I', 'HUST_0.5X_O', 'HUST_B', 'HUST_C', 'HUST_I', 'HUST_O', 'HUST_H'] + ['HUST_B_', 'HUST_M_']
    JNU_ClassName = ['JNU_ib', 'JNU_n', 'JNU_ob', 'JNU_tb']
    HIT_ClassName = ['HIT_0.0', 'HIT_1.0', 'HIT_2.0']
    ClassName = HUST_ClassName + JNU_ClassName + HIT_ClassName
    Class_Sample_Dict = {}
    for i_class in ClassName:
        Dataset_name = i_class.split("_", 1)[0]
        class_name = i_class.split("_", 1)[1]
        if Dataset_name == 'HUST':
            i_class_sample = Read_HUST_OneClass_sample(filepath + '/HUST', class_name=class_name, SampleLength=SampleLength, SampleNum=200, step=step)
            Class_Sample_Dict[i_class] = i_class_sample
        if Dataset_name == 'JNU':
            i_class_sample = Read_JNU_OneClass_sample(filepath + '/JNU', class_name=class_name, SampleLength=SampleLength, SampleNum=200, step=step)
            Class_Sample_Dict[i_class] = i_class_sample
        if Dataset_name == 'HIT':
            i_class_sample = Read_HIT_OneClass_sample(filepath + '/HIT', class_name=class_name, SampleLength=SampleLength, SampleNum=200)
            Class_Sample_Dict[i_class] = i_class_sample

    Task_Train_X = []
    Task_Train_Y = []
    Task_Query_X = []
    Task_Query_Y = []

    for i_train_task in range(Task):
        indices_list, class_list = get_random_indices_and_values(lst=ClassName, M=Way)
        i_Task_Train_X = []
        i_Task_Train_Y = []
        i_Task_Query_X = []
        i_Task_Query_Y = []
        for index, i_class_name in enumerate(class_list):
            Sample_list = Class_Sample_Dict[i_class_name]
            indices_sample_list, class_sample_list = get_random_indices_and_values(lst=Sample_list, M=Shot + QueryShot)
            i_Task_Train_X.append(np.vstack(class_sample_list[:Shot]))
            i_Task_Train_Y += [index] * Shot
            i_Task_Query_X.append(np.vstack(class_sample_list[Shot:]))
            i_Task_Query_Y += [index] * QueryShot

        i_Task_Train_X = np.vstack(i_Task_Train_X)
        indices = np.arange(len(i_Task_Train_X))
        np.random.shuffle(indices)
        i_Task_Train_X_shuffled = i_Task_Train_X[indices]
        i_Task_Train_Y_shuffled = [i_Task_Train_Y[i] for i in indices]

        i_Task_Query_X = np.vstack(i_Task_Query_X)
        indices = np.arange(len(i_Task_Query_X))
        np.random.shuffle(indices)
        i_Task_Query_X_shuffled = i_Task_Query_X[indices]
        i_Task_Query_Y_shuffled = [i_Task_Query_Y[i] for i in indices]

        Task_Train_X.append(i_Task_Train_X_shuffled)
        Task_Train_Y.append(i_Task_Train_Y_shuffled)
        Task_Query_X.append(i_Task_Query_X_shuffled)
        Task_Query_Y.append(i_Task_Query_Y_shuffled)

    Task_Train_X = torch.from_numpy(np.stack(Task_Train_X)).to(torch.float32)
    Task_Train_Y = torch.from_numpy(np.array(Task_Train_Y)).to(torch.int64)
    Task_Query_X = torch.from_numpy(np.stack(Task_Query_X)).to(torch.float32)
    Task_Query_Y = torch.from_numpy(np.array(Task_Query_Y)).to(torch.int64)

    return Task_Train_X, Task_Train_Y, Task_Query_X, Task_Query_Y


def Build_Task_from_HUST_JNU_CWRU(filepath, Task=1000, SampleLength=2048, step=1024, Way=3, Shot=5, QueryShot=5):
    HUST_ClassName = ['HUST_0.5X_B', 'HUST_0.5X_C', 'HUST_0.5X_I', 'HUST_0.5X_O', 'HUST_B', 'HUST_C', 'HUST_I', 'HUST_O', 'HUST_H'] + ['HUST_B_', 'HUST_M_']
    JNU_ClassName = ['JNU_ib', 'JNU_n', 'JNU_ob', 'JNU_tb']
    CWRU_ClassName = ['CWRU_B007', 'CWRU_B014', 'CWRU_B021', 'CWRU_IR007', 'CWRU_IR014', 'CWRU_IR021', 'CWRU_OR007', 'CWRU_OR014', 'CWRU_OR021', 'CWRU_normal']
    ClassName = HUST_ClassName + JNU_ClassName + CWRU_ClassName
    Class_Sample_Dict = {}
    for i_class in ClassName:
        Dataset_name = i_class.split("_", 1)[0]
        class_name = i_class.split("_", 1)[1]
        if Dataset_name == 'HUST':
            i_class_sample = Read_HUST_OneClass_sample(filepath + '/HUST', class_name=class_name, SampleLength=SampleLength, SampleNum=200, step=step)
            Class_Sample_Dict[i_class] = i_class_sample
        if Dataset_name == 'JNU':
            i_class_sample = Read_JNU_OneClass_sample(filepath + '/JNU', class_name=class_name, SampleLength=SampleLength, SampleNum=200, step=step)
            Class_Sample_Dict[i_class] = i_class_sample
        if Dataset_name == 'CWRU':
            i_class_sample = Read_CWRU_OneClass_sample(filepath + '/CWRU', class_name=class_name, SampleLength=SampleLength, SampleNum=200, step=step)
            Class_Sample_Dict[i_class] = i_class_sample

    Task_Train_X = []
    Task_Train_Y = []
    Task_Query_X = []
    Task_Query_Y = []

    for i_train_task in range(Task):
        indices_list, class_list = get_random_indices_and_values(lst=ClassName, M=Way)
        i_Task_Train_X = []
        i_Task_Train_Y = []
        i_Task_Query_X = []
        i_Task_Query_Y = []
        for index, i_class_name in enumerate(class_list):
            Sample_list = Class_Sample_Dict[i_class_name]
            indices_sample_list, class_sample_list = get_random_indices_and_values(lst=Sample_list, M=Shot + QueryShot)
            i_Task_Train_X.append(np.vstack(class_sample_list[:Shot]))
            i_Task_Train_Y += [index] * Shot
            i_Task_Query_X.append(np.vstack(class_sample_list[Shot:]))
            i_Task_Query_Y += [index] * QueryShot

        i_Task_Train_X = np.vstack(i_Task_Train_X)
        indices = np.arange(len(i_Task_Train_X))
        np.random.shuffle(indices)
        i_Task_Train_X_shuffled = i_Task_Train_X[indices]
        i_Task_Train_Y_shuffled = [i_Task_Train_Y[i] for i in indices]

        i_Task_Query_X = np.vstack(i_Task_Query_X)
        indices = np.arange(len(i_Task_Query_X))
        np.random.shuffle(indices)
        i_Task_Query_X_shuffled = i_Task_Query_X[indices]
        i_Task_Query_Y_shuffled = [i_Task_Query_Y[i] for i in indices]

        Task_Train_X.append(i_Task_Train_X_shuffled)
        Task_Train_Y.append(i_Task_Train_Y_shuffled)
        Task_Query_X.append(i_Task_Query_X_shuffled)
        Task_Query_Y.append(i_Task_Query_Y_shuffled)

    Task_Train_X = torch.from_numpy(np.stack(Task_Train_X)).to(torch.float32)
    Task_Train_Y = torch.from_numpy(np.array(Task_Train_Y)).to(torch.int64)
    Task_Query_X = torch.from_numpy(np.stack(Task_Query_X)).to(torch.float32)
    Task_Query_Y = torch.from_numpy(np.array(Task_Query_Y)).to(torch.int64)

    return Task_Train_X, Task_Train_Y, Task_Query_X, Task_Query_Y




import random
import os
import numpy as np
import torch
from scipy.io import loadmat

def get_random_indices_and_values(lst, M):
    if M > len(lst):
        raise ValueError("M度")
    indices = random.sample(range(len(lst)), M)
    values = [lst[i] for i in indices]
    return indices, values

def read_OneOH(filepath, ClassNameList):
    filenames = os.listdir(filepath)
    def capture(original_path, ClassNameList):
        files = {}
        for i in filenames:
            ClassName = ''
            for one in ClassNameList:
                if one in i:
                    ClassName = one
                    break
            file_path = os.path.join(filepath, i)
            file = loadmat(file_path)
            file_keys = file.keys()
            for key in file_keys:
                if 'DE' in key:
                    files[ClassName] = file[key].ravel()
        return files
    return capture(filepath, ClassNameList)

def read_CWRU_Sample_from_oneSample_oneOH(OneOH_Dict, Select_Class, Sample_num=200, step=150, SampleLength=2048):
    Class_sample = []
    slice_data_class = OneOH_Dict[Select_Class]
    for i in range(Sample_num):
        sample = slice_data_class[i * step: i * step + SampleLength]
        Class_sample.append(sample)
    return Class_sample

def Build_Task_from_JNU_HIT_CWRU(filepath, Task=1000, SampleLength=2048, step=1024, Way=4, Shot=5, QueryShot=5):
    JNU_ClassName = ['JNU_ib', 'JNU_n', 'JNU_ob', 'JNU_tb']
    HIT_ClassName = ['HIT_0.0', 'HIT_1.0', 'HIT_2.0']
    CWRU_ClassName = ['CWRU_B007', 'CWRU_B014', 'CWRU_B021', 'CWRU_IR007', 'CWRU_IR014', 'CWRU_IR021', 'CWRU_OR007',
                      'CWRU_OR014', 'CWRU_OR021', 'CWRU_normal']
    ClassName = JNU_ClassName + HIT_ClassName + CWRU_ClassName
    Class_Sample_Dict = {}
    for i_class in ClassName:
        Dataset_name = i_class.split("_", 1)[0]
        class_name = i_class.split("_", 1)[1]
        if Dataset_name == 'JNU':
            i_class_sample = Read_JNU_OneClass_sample(filepath + '/JNU', class_name=class_name,
                                                      SampleLength=SampleLength, SampleNum=200, step=step)
            Class_Sample_Dict[i_class] = i_class_sample
        if Dataset_name == 'HIT':
            i_class_sample = Read_HIT_OneClass_sample(filepath + '/HIT', class_name=class_name,
                                                      SampleLength=SampleLength, SampleNum=200)
            Class_Sample_Dict[i_class] = i_class_sample
        if Dataset_name == 'CWRU':
            i_class_sample = Read_CWRU_OneClass_sample(filepath + '/CWRU', class_name=class_name,
                                                       SampleLength=SampleLength, SampleNum=200, step=step)
            Class_Sample_Dict[i_class] = i_class_sample
    Task_Train_X = []
    Task_Train_Y = []
    Task_Query_X = []
    Task_Query_Y = []
    for i_train_task in range(Task):
        indices_list, class_list = get_random_indices_and_values(lst=ClassName, M=Way)
        i_Task_Train_X = []
        i_Task_Train_Y = []
        i_Task_Query_X = []
        i_Task_Query_Y = []
        for index, i_class_name in enumerate(class_list):
            Sample_list = Class_Sample_Dict[i_class_name]
            indices_sample_list, class_sample_list = get_random_indices_and_values(lst=Sample_list, M=Shot + QueryShot)
            i_Task_Train_X.append(np.vstack(class_sample_list[:Shot]))
            i_Task_Train_Y += [index] * Shot
            i_Task_Query_X.append(np.vstack(class_sample_list[Shot:]))
            i_Task_Query_Y += [index] * QueryShot
        i_Task_Train_X = np.vstack(i_Task_Train_X)
        indices = np.arange(len(i_Task_Train_X))
        np.random.shuffle(indices)
        i_Task_Train_X_shuffled = i_Task_Train_X[indices]
        i_Task_Train_Y_shuffled = [i_Task_Train_Y[i] for i in indices]
        i_Task_Query_X = np.vstack(i_Task_Query_X)
        indices = np.arange(len(i_Task_Query_X))
        np.random.shuffle(indices)
        i_Task_Query_X_shuffled = i_Task_Query_X[indices]
        i_Task_Query_Y_shuffled = [i_Task_Query_Y[i] for i in indices]
        Task_Train_X.append(i_Task_Train_X_shuffled)
        Task_Train_Y.append(i_Task_Train_Y_shuffled)
        Task_Query_X.append(i_Task_Query_X_shuffled)
        Task_Query_Y.append(i_Task_Query_Y_shuffled)
    Task_Train_X = torch.from_numpy(np.stack(Task_Train_X)).to(torch.float32)
    Task_Train_Y = torch.from_numpy(np.array(Task_Train_Y)).to(torch.int64)
    Task_Query_X = torch.from_numpy(np.stack(Task_Query_X)).to(torch.float32)
    Task_Query_Y = torch.from_numpy(np.array(Task_Query_Y)).to(torch.int64)
    return Task_Train_X, Task_Train_Y, Task_Query_X, Task_Query_Y




