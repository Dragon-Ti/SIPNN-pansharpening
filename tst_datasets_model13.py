import h5py
import numpy as np
import torch
import cv2
import scipy.io as sio
import os, glob, re
from pathlib import Path
import matlab.engine
from MyModel.metrics import ref_evaluate, no_ref_evaluate


def get_edge(data):  # get high-frequency
    rs = np.zeros_like(data)
    if len(rs.shape) == 3:
        for i in range(data.shape[2]):
            rs[:, :, i] = data[:, :, i] - cv2.boxFilter(data[:, :, i], -1, (5, 5))
    elif len(rs.shape) == 4:
        for i in range(data.shape[0]):
            for j in range(data.shape[3]):
                rs[i, :, :, j] = data[i, :, :, j] - cv2.boxFilter(data[i, :, :, j], -1, (5, 5))
    else:
        rs = data - cv2.boxFilter(data, -1, (5, 5))
    return rs

def load_set(file_path):
    data = h5py.File(file_path)  # B*C*H*W
    # tensor type:
    lms = torch.Tensor(np.array(data['lms'])) / 2047.0  # CxHxW = 8x256x256

    ms = torch.Tensor(np.array(data['ms']) / 2047.0)  # CxHxW= 8x64x64
    pan = torch.Tensor(np.array(data['pan']) / 2047.0)  # HxW = 256x256

    return ms, pan, lms

# 在这里改文件夹和模型名字



def datasets_test(sensor, datasets_type, model, load_folder, load_epoch):

    test_file_path, test_file_path_full = tst_path(sensor)
    if datasets_type == '':
        file_path = test_file_path
    elif datasets_type == '_full':
        file_path = test_file_path_full
    outname = 'Test_' + str(sensor) + '_data'

    ms, pan, lms = load_set(file_path)
    if isinstance(model, str):
        model = torch.load(model).eval()
    else:
        model = model.eval()

    with torch.no_grad():

        x1, x2, x3 = ms, pan, lms   # read data: CxHxW (numpy type)
        print(x1.shape)
        # 不需要unsqueeze
        x1 = x1.cuda().float()
        x2 = x2.cuda().float()

        sr = model(x1, x2)

        # convert to numpy type with permute and squeeze: HxWxC (go to cpu for easy saving)
        sr = sr.permute(0, 2, 3, 1).cpu().detach().numpy()  # HxWxC

        print(sr.shape)

        save_path = increment_path(load_folder + str(load_epoch) + "/results" + str(datasets_type), exist_ok=False)

        for i in range(sr.shape[0]):
            save_name = os.path.join(save_path, outname + str(i + 1) + ".mat")  # fixed! save as .mat format that will used in Matlab!
            i_sr = sr[i, :, :, :]
            sio.savemat(save_name, {"result": i_sr})  # fixed!
            # if datasets_type == '_full':
            #     full_file_path = Path("D:\pythonProject\PanNet_Quaternion\test_datasets\WV3\full_examples_2\Test_WV3_data" + str(i + 1) + ".mat");
            #     sio.loadmat(full_file_path)
            #     no_ref_evaluate(i_sr, pan, )
        # # MATLAB调用
        # eng = matlab.engine.start_matlab()
        # eng.cd('D:/pythonProject/PanNet_Quaternion/DLPan-Toolbox-main/02-Test-toolbox-for-traditional-and-DL(Matlab)')
        # if datasets_type == '':
        #     eng.single_model_qi_small(str(save_path.absolute()), model_which[1:], nargout=0)
        # elif datasets_type == '_full':
        #     eng.single_model_qi_full(str(save_path.absolute()), model_which[1:], nargout=0)

        with open(str(save_path) + '/arguments.txt', 'a') as f:
            f.write(str(load_epoch))

def increment_path(path, exist_ok=False, sep='', mkdir=True):
    """
    Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    :param path: file or directory path to increment
    :param exist_ok: existing project/name ok, do not increment
    :param sep: separator for directory name
    :param mkdir: create directory
    :return: incremented path
    """
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        suffix = path.suffix
        path = path.with_suffix('')
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        path = Path(f"{path}{sep}{n}{suffix}")  # update path
    dir_ = path if path.suffix == '' else path.parent  # directory
    if not dir_.exists() and mkdir:
        dir_.mkdir(parents=True, exist_ok=True)  # make directory
    return path

def tst_path(satellite):
    test_file_path = '../test_datasets/inONE/' + satellite.upper() + \
                     '/reduced_examples/test_' + satellite.lower() + '_multiExm1.h5'
    test_file_path_full = '../test_datasets/inONE/' + satellite.upper() + \
                          '/full_examples/test_' + satellite.lower() + '_OrigScale_multiExm1.h5'
    return test_file_path, test_file_path_full

def datasets_tst_fr_rr(satellite, model_which, load_epoch):
    load_folder = "./" + satellite + "_model" + model_which + "_runs/"
    ckpt = load_folder + "/Weights/" + load_epoch + ".pth"  # chose model
    net = torch.load(ckpt).eval()
    # file_path_WV3 = '../test_datasets/inONE/WV3/reduced_examples/test_wv3_multiExm1.h5'
    # file_path_WV3_full = '../test_datasets/inONE/WV3/full_examples/test_wv3_OrigScale_multiExm1.h5'
    # file_path_WV2 = '../test_datasets/inONE/WV2/reduced_examples/test_wv2_multiExm1.h5'
    # file_path_WV2_full = '../test_datasets/inONE/WV2/full_examples/test_wv2_OrigScale_multiExm1.h5'
    datasets_test(satellite, datasets_type='', model=net, load_folder=load_folder, load_epoch=load_epoch)
    datasets_test(satellite, datasets_type='_full', model=net, load_folder=load_folder, load_epoch=load_epoch)
    # 试试看能不能直接调MATLAB
    # eng = matlab.engine.start_matlab()
    # eng.cd()


if __name__ == '__main__':
    for i in range(400, 501, 10):
        load_epoch = i
        datasets_tst_fr_rr('QB', '13', str(load_epoch))