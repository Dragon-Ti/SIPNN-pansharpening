import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data_model13 import Dataset_Pro
from d2l import torch as d2l
import numpy as np
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
import os, glob, re
from pathlib import Path
from tst_datasets_model13 import datasets_tst_fr_rr

# model_num = '13a1'

# 训练主函数
def train(net, train_iter, test_iter, num_epochs, lr, lr_decay=1, start_epoch=0, step_size=100):

    print('start training...')

    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    if start_epoch == 0:
        net.apply(init_weights)
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=0)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size, gamma=lr_decay)  # 学习率调整
    loss_fn = nn.MSELoss(size_average=True).cuda()
    # 保存超参
    with open(str(save_dir) + '/arguments.txt', 'a') as f:
        f.write("model:{}\nlearing rate:{}\nepoch:{}\nbatch_size:{}\nresblocknum:{}\nblocknum:{}\n ".format(
            model_num, lr, epochs, batch_size, res_block_num, block_num))
    for epoch in range(start_epoch, num_epochs, 1):
        epoch += 1
        epoch_train_loss, epoch_val_loss = [], []  # 本次epoch的loss
        metric = d2l.Accumulator(3)
        net.train()
        for i, batch in enumerate(train_iter, start=1):
            # gt Nx8x64x64
            # lms Nx8x64x64
            # ms_hp Nx8x16x16
            # pan_hp Nx1x64x64
            gt, ms, pan, lms = batch[0].cuda(), batch[1].cuda(), batch[3].cuda(), batch[5].cuda()
            optimizer.zero_grad()
            sr = net(ms, pan)
            loss = loss_fn(sr, gt)
            epoch_train_loss.append(loss.item())  # 将每个batch的loss保存为一个向量
            loss.backward()
            optimizer.step()
            # for name, layer in model.named_parameters():  # 看不懂捏
            #     writer.add_histogram('net/'+name + '_data_weight_decay', layer, epoch*i)

        total_loss = np.nanmean(np.array(epoch_train_loss))
        writer.add_scalar('mse_loss/t_loss', total_loss, epoch)  # write to tensorboard to check
        print('Epoch: {}/{} training loss: {:.7f}'.format(epochs, epoch, total_loss))  # print loss for each epoch

        # 保存模型
        if epoch % ckpt == 0:  # if each ckpt epochs, then start to save model
            torch.save(net, net_save_path + '/' + "{}.pth".format(epoch))

        # eval time
        net.eval()
        with torch.no_grad():
            for i, batch in enumerate(validate_data_loader, 1):
                gt, ms, pan, lms = batch[0].cuda(), batch[1].cuda(), batch[3].cuda(), batch[5].cuda()

            sr = net(ms, pan)
            loss = loss_fn(sr, gt)
            epoch_val_loss.append(loss.item())

        if epoch % 10 == 0:
            v_loss = np.nanmean(np.array(epoch_val_loss))
            writer.add_scalar('val/v_loss', v_loss, epoch)
            print('             validate loss: {:.7f}'.format(v_loss))
            # 记录最佳epoch
            # if v_loss < best_val_loss:
            #     best_val_loss = v_loss
            #     best_val_epoch = epoch
                # torch.save(net, net_save_path + '/' + "best in {} epoch.pth".format(epoch))
        scheduler.step()
    writer.close()


# yolo抄来的增量目录
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


# datasets, 只改数据集不改网络换个名字就行
satellite = 'WV3'
spectral_num = 8

# HYPER PRAMS
lr = 0.0003
epochs = 500
ckpt = 10  # 检查点，进行网络保存
batch_size = 32
load_epoch = 0  # 在这改载入的epoch

# MODEL SELECT
model_num = '13a5'  # 如果只改模型内参数，在这里改模型名，不重新写model文件
from model13 import SIPNN, summaries  # 在这里改模型
model_path = satellite + "_model" + model_num + "_runs/Weights/" + str(load_epoch) + ".pth"

# MODEL PARAMS
res_block_num = 2
block_num = 2


if __name__ == "__main__":
    # load data and model + optimizer + loss_fn
    net = SIPNN(res_block_num, block_num, spectral_num).cuda()
    if os.path.isfile(model_path):
        net = torch.load(model_path)  ## Load the pretrained Encoder
        print('Net is Successfully Loaded from %s' % (model_path))
    # summaries(net, grad=True)  # 这是what
    summary(net, [(spectral_num, 16, 16), (1, 64, 64)])

    # 目录
    save_dir = increment_path(satellite + '_model' + model_num + '_runs', exist_ok=False)
    writer = SummaryWriter(save_dir)
    net_save_path = str(save_dir) + '/Weights'
    os.mkdir(net_save_path)

    train_path = '../training_data/train_' + satellite + '.h5'
    train_set = Dataset_Pro(train_path)
    training_data_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=batch_size, shuffle=True,
                                      pin_memory=True, drop_last=True)
    valid_path = '../training_data/valid_' + satellite + '.h5'
    validate_set = Dataset_Pro(valid_path)
    validate_data_loader = DataLoader(dataset=validate_set, num_workers=0, batch_size=batch_size, shuffle=True,
                                      pin_memory=True, drop_last=True)

    train(net, training_data_loader, validate_data_loader, epochs, lr, start_epoch=load_epoch)

    # datasets_test_fr_rr(satellite, net, str(save_dir.absolute()))
    # datasets_test(file_path_WV3, 'Test_WV3_data', '')
    # datasets_test(file_path_WV3_full, 'Test_WV3_data', '_full')