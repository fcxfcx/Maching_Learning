# Numerical Operations
import math
import numpy as np

# Reading/Writing Data
import pandas as pd
import os
import csv

# For Progress Bar
from tqdm import tqdm

# Pytorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# For plotting learning curve
from torch.utils.tensorboard import SummaryWriter


def same_seed(seed):
    """
    该方法的目的是去除程序随机性，包括：
    1. 配置cudnn（NVIDIA打造的针对深度神经网络的加速库）的参数，使其使用固定的卷积算法。
    2. 将整个程序的seed设置成相同，以保证生成的随机数是相等的。这里把可能用到random方法的类都设置了相同的seed。
    """
    # 配置cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # 配置seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_valid_split(data_set, valid_ratio, seed):
    """
    从数据集中分割出用于检验效果的校验集
    根据valid_ratio确定校验集需要多少个数据，然后通过随机分组确定训练集和校验集
    """
    valid_set_size = int(valid_ratio * len(data_set))
    train_set_size = len(data_set) - valid_set_size
    train_set, valid_set = random_split(data_set, [train_set_size, valid_set_size],
                                        generator=torch.Generator().manual_seed(seed))
    return np.array(train_set), np.array(valid_set)


def predict(test_loader, model, device):
    model.eval()  # Set your model to evaluation mode.
    preds = []
    for x in tqdm(test_loader):
        x = x.to(device)
        with torch.no_grad():
            pred = model(x)
            preds.append(pred.detach().cpu())
    preds = torch.cat(preds, dim=0).numpy()
    return preds


class COVID19Dataset(Dataset):
    """
    x: Features.
    y: Targets, if none, do prediction.
    包装数据为Tensor格式，提供基本的数据集类
    """

    def __init__(self, x, y=None):
        if y is None:
            self.y = y
        else:
            self.y = torch.FloatTensor(y)
        self.x = torch.FloatTensor(x)

    def __getitem__(self, idx):
        if self.y is None:
            return self.x[idx]
        else:
            return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)


class My_Model(nn.Module):
    def __init__(self, input_dim):
        """
        这里是定义模型的网络中每一层的操作，对操作顺序的定义置于Sequential类中
        首先需要了解这里出现的两个层操作：
        1. Linear操作用于改变feature张量的维度，这个过程是通过随机的权重矩阵进行矩阵乘法实现的
        2. ReLU函数在课程中出现过，也可理解为将小于零的数值截止至0

        因此这个部分的网络结构可以理解为，
        首先将输入的feature降维至16维，然后进行一次ReLU，
        之后将16维的feature降维至8维，再进行一次ReLU，
        最后将8维的feature降维至1维，输出结果

        这里可以回去看第一周课程中首次介绍深度神经网络的部分
        """
        super(My_Model, self).__init__()
        # TODO: modify model's structure, be aware of dimensions.
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        x = self.layers(x)
        x = x.squeeze(1)  # (B, 1) -> (B) 进行维度压缩，去除维度为1的部分
        return x


def select_feat(train_data, valid_data, test_data, select_all=True):
    """
    根据问题给出的数据格式，最后一列为我们需要预测的阳性病例数量
    对数据集来说取所有行的最后一列值为y，剩下的为x，然后在x中选取指定的列作为feature
    """
    y_train, y_valid = train_data[:, -1], valid_data[:, -1]
    raw_x_train, raw_x_valid, raw_x_test = train_data[:, :-1], valid_data[:, :-1], test_data

    if select_all:
        feat_idx = list(range(raw_x_train.shape[1]))
    else:
        feat_idx = [0, 1, 2, 3, 4]  # TODO: Select suitable feature columns.

    return raw_x_train[:, feat_idx], raw_x_valid[:, feat_idx], raw_x_test[:, feat_idx], y_train, y_valid


def trainer(train_loader, valid_loader, model, config, device):
    criterion = nn.MSELoss(reduction='mean')  # 选择均方误差的方式计算loss

    """
     在这里去定义模型的优化算法，同时确定学习率和动量momentum
     SGD就是其中的一种，名为随机梯度下降，参数详细说明可以参考：https://blog.csdn.net/weixin_46221946/article/details/122644487
    """
    # TODO: Please check https://pytorch.org/docs/stable/optim.html to get more available algorithms.
    # TODO: L2 regularization (optimizer(weight decay...) or implement by your self).
    optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9)

    writer = SummaryWriter()  # Tensorboard用于可视化训练数据

    if not os.path.isdir('./models'):
        os.mkdir('./models')  # 用于存放模型文件的文件夹

    n_epochs, best_loss, step, early_stop_count = config['n_epochs'], math.inf, 0, 0

    for epoch in range(n_epochs):
        model.train()  # Set your model to train mode.
        loss_record = []

        # tqdm is a package to visualize your training progress.
        train_progress_bar = tqdm(train_loader, position=0, leave=True)

        """
        这里是典型的一个epoch内的训练操作，包括了：
        1.将梯度归零（防止梯度的累加）
        2.使用GPU加速
        3.获得预测值并计算loss
        4.根据loss计算梯度，并根据学习率等更新参数
        """
        for x, y in train_progress_bar:
            optimizer.zero_grad()  # Set gradient to zero.
            x, y = x.to(device), y.to(device)  # Move your data to device.
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()  # Compute gradient(backpropagation).
            optimizer.step()  # Update parameters.
            step += 1
            loss_record.append(loss.detach().item())

            # Display current epoch number and loss on tqdm progress bar.
            train_progress_bar.set_description(f'Epoch [{epoch + 1}/{n_epochs}]')
            train_progress_bar.set_postfix({'loss': loss.detach().item()})

        mean_train_loss = sum(loss_record) / len(loss_record)
        writer.add_scalar('Loss/train', mean_train_loss, step)

        """
        验证过程，这里会得到验证集中的loss，并且始终维护一个best_loss值
        如果模型验证的loss低于best_loss，证明此时的模型是最优，则储存当前的模型
        如果出现多次训练出的模型loss高于best_loss则停止该过程（目前定义的是超过400次则停止）
        """
        model.eval()  # Set your model to evaluation mode.
        loss_record = []
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                loss = criterion(pred, y)

            loss_record.append(loss.item())

        mean_valid_loss = sum(loss_record) / len(loss_record)
        print(f'Epoch [{epoch + 1}/{n_epochs}]: Train loss: {mean_train_loss:.4f}, Valid loss: {mean_valid_loss:.4f}')
        writer.add_scalar('Loss/valid', mean_valid_loss, step)

        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(model.state_dict(), config['save_path'])  # Save your best model
            print('Saving model with loss {:.3f}...'.format(best_loss))
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count >= config['early_stop']:
            print('\nModel is not improving, so we halt the training session.')
            return


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = {
        'seed': 5201314,  # Your seed number, you can pick your lucky number. :)
        'select_all': True,  # Whether to use all features.
        'valid_ratio': 0.2,  # validation_size = train_size * valid_ratio
        'n_epochs': 3000,  # Number of epochs.
        'batch_size': 256,
        'learning_rate': 1e-5,
        'early_stop': 400,  # If model has not improved for this many consecutive epochs, stop training.
        'save_path': './models/model.ckpt'  # Your model will be saved here.
    }

    # Set seed for reproducibility
    same_seed(config['seed'])

    # train_data size: 2699 x 118 (id + 37 states + 16 features x 5 days)
    # test_data size: 1078 x 117 (without last day's positive rate)
    train_data, test_data = pd.read_csv('./covid.train.csv').values, pd.read_csv('./covid.test.csv').values
    train_data, valid_data = train_valid_split(train_data, config['valid_ratio'], config['seed'])

    # Print out the data size.
    print(f"""train_data size: {train_data.shape} 
    valid_data size: {valid_data.shape} 
    test_data size: {test_data.shape}""")

    # Select features
    x_train, x_valid, x_test, y_train, y_valid = select_feat(train_data, valid_data, test_data, config['select_all'])

    # Print out the number of features.
    print(f'number of features: {x_train.shape[1]}')

    train_dataset, valid_dataset, test_dataset = COVID19Dataset(x_train, y_train), \
                                                 COVID19Dataset(x_valid, y_valid), \
                                                 COVID19Dataset(x_test)

    # Pytorch data loader loads pytorch dataset into batches.
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)

    model = My_Model(input_dim=x_train.shape[1]).to(device)  # put your model and data on the same computation device.
    trainer(train_loader, valid_loader, model, config, device)