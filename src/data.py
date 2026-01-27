import torch
import numpy as np
from torch.autograd import Variable
from utils import *


class DataBasicLoader(object):
    def __init__(self, args):
        self.cuda = args.cuda
        self.P = args.window  # 窗口大小
        self.h = args.horizon # 预测步长
        self.add_his_day = True #是否额外添加一年前同一天的数据
        self.rawdat = np.loadtxt(open("data/{}.txt".format(args.dataset)), delimiter=',')
        print('data shape: ', self.rawdat.shape)

        self.orig_adj = None     # 原始邻接矩阵占位
        self.adj = None          # 对称归一化后的邻接矩阵占位     
        self.degree_adj = None   # 度数矩阵占位

        if args.sim_mat:
            self.load_sim_mat(args)   
            # 得到torch.tensor的self.adj和self.orig_adj，均在GPU上

        if (len(self.rawdat.shape)==1):
            self.rawdat = self.rawdat.reshape((self.rawdat.shape[0], 1))

        self.dat = np.zeros(self.rawdat.shape)   # 归一化后的数据矩阵占位
        self.n, self.m = self.dat.shape # n:样本数   m:节点数

        self._pre_train(int(args.train * self.n), int((args.train + args.val) * self.n), self.n)
        """这在索引划分上是正确的，它确保了：
        训练集在前。
        验证集在中间。
        测试集在最后。 这符合时间序列预测“不能用未来数据预测过去”的原则。
        """
        self._split(int(args.train * self.n), int((args.train + args.val) * self.n), self.n)
        print('size of train/val/test sets: ',len(self.train[0]),len(self.val[0]),len(self.test[0]))
        print('shape of X of train/val/test sets: ',self.train[0].shape,self.val[0].shape,self.test[0].shape)
        print('shape of Y of train/val/test sets: ',self.train[1].shape,self.val[1].shape,self.test[1].shape)
        print('-'*100)

    def load_sim_mat(self, args):
        
        adj_np = np.loadtxt(f"data/{args.sim_mat}.txt", delimiter=',')
        device = torch.device("cuda" if self.cuda and torch.cuda.is_available() else "cpu")
       
        if args.sparse:
            # --- 方案 A: 稀疏模式 (针对大矩阵) ---
            print("Using Sparse Mode...")
            # 调用 CPU 端的归一化
            adj_sp = normalize_adj2(adj_np)
            # 转换为 Torch 稀疏张量，再转化为密集矩阵并送入 GPU
            self.adj = sparse_mx_to_torch_sparse_tensor(adj_sp).to_dense().to(device)
        else :
            # --- 方案 B: 密集模式 (针对小矩阵，利用 GPU 广播加速) ---
            print("Using Dense Mode...")
            adj_tensor = torch.tensor(adj_np, dtype=torch.float32).to(device)
            d = adj_tensor.sum(dim=1)
            d_inv_sqrt = torch.pow(torch.clamp(d, min=1e-12), -0.5)
            d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
            # 广播计算
            self.adj = d_inv_sqrt[:, None] * adj_tensor * d_inv_sqrt[None, :]
        # 统一保存原始矩阵以便对比
        self.orig_adj = torch.tensor(adj_np, dtype=torch.float32).to(device)
        self.degree_adj = torch.sum(self.orig_adj, dim=-1).unsqueeze(0).unsqueeze(-1)  # shape [1, num_nodes, 1]

    # 数据预处理，划分数据集范围，计算训练集的最大最小值用于归一化，计算峰值阈值。
    # train：训练集的结束索引   valid：验证集的结束索引  test：测试集的结束索引（实际上这个参数没有用到）
    def _pre_train(self, train, valid, test):

        self.train_set = train_set = range(self.P+self.h-1, train)
        #self.train_set存的是输出（目标值）的索引
        #第一个有效样本的索引是从 P + h - 1开始的输出位置

        self.valid_set = valid_set = range(train, valid)
        self.test_set = test_set = range(valid, self.n)

        # 直接对原始数据的训练部分进行计算
        # 这样做既涵盖了所有输入，也涵盖了所有输出，逻辑最清晰
        train_slice = self.rawdat[:train, :]

        self.max = np.max(train_slice, axis=0)
        self.min = np.min(train_slice, axis=0)
        self.peak_thold = np.mean(train_slice, axis=0)

        # 执行归一化
        self.dat = (self.rawdat - self.min) / (self.max - self.min + 1e-12)

        """"
        # 另一种计算方式（但逻辑上不如上面清晰）
        #调用_batchify方法：将训练集索引转换为批次数据
        #返回值结构：[X, Y] 元组  X：形状 (n_train_samples, P, m) 的输入序列  Y：形状 (n_train_samples, m) 的目标值
        self.tmp_train = self._batchify(train_set, self.h, useraw=True)

        train_mx = torch.cat((self.tmp_train[0][0], self.tmp_train[1]), 0).numpy() #沿着第一个维度拼接
        self.max = np.max(train_mx, 0)
        self.min = np.min(train_mx, 0)
        self.peak_thold = np.mean(train_mx, 0)  
        self.dat  = (self.rawdat  - self.min ) / (self.max  - self.min + 1e-12)
        """
         
    def _split(self, train, valid, test):
        self.train = self._batchify(self.train_set, self.h) # torch.Size([179, 20, 47]) torch.Size([179, 47])
        self.val = self._batchify(self.valid_set, self.h)
        self.test = self._batchify(self.test_set, self.h)
        if (train == valid):       #此时没有验证集
            self.val = self.test   
 
    def _batchify(self, idx_set, horizon, useraw=False): 

        n = len(idx_set)
        Y = torch.zeros((n, self.m))
        if self.add_his_day and not useraw:
            X = torch.zeros((n, self.P+1, self.m)) # 额外添加一年前同一天的数据
        else:
            X = torch.zeros((n, self.P, self.m))
        
        for i in range(n):
            end = idx_set[i] - horizon + 1  #不包含
            start = end - self.P

            if useraw: # for normalization
                X[i,:self.P,:] = torch.from_numpy(self.rawdat[start:end, :])
                Y[i,:] = torch.from_numpy(self.rawdat[idx_set[i], :])
            else:
                his_window = self.dat[start:end, :]
                if self.add_his_day:
                    if idx_set[i] > 51 : # at least 52
                        his_day = self.dat[idx_set[i]-52:idx_set[i]-51, :] #
                    else: # no history day data
                        his_day = np.zeros((1,self.m))

                    his_window = np.concatenate([his_day,his_window])
                    # print(his_window.shape,his_day.shape,idx_set[i],idx_set[i]-52,idx_set[i]-51)
                    X[i,:self.P+1,:] = torch.from_numpy(his_window) # size (window+1, m)
                else:
                    X[i,:self.P,:] = torch.from_numpy(his_window) # size (window, m)
                Y[i,:] = torch.from_numpy(self.dat[idx_set[i], :])
        return [X, Y]

    def get_batches(self, data, batch_size, shuffle=True):
        dataset = torch.utils.data.TensorDataset(data[0], data[1])
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle,
            pin_memory=True if self.cuda else False  # 加上这一行性能更优
        )
        for batch_inputs, batch_targets in dataloader:
            if self.cuda:
                # 这里的 .cuda() 配合 pin_memory 效果最好
                batch_inputs = batch_inputs.cuda()
                batch_targets = batch_targets.cuda()
            yield batch_inputs, batch_targets

    """
    def get_batches(self, data, batch_size, shuffle=True):
        inputs = data[0]
        targets = data[1]
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length)   #生成随机排列索引
        else:
            index = torch.LongTensor(range(length))   #创建顺序索引 [0, 1, 2, ..., length-1]
        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt,:] 
            #虽然代码里只写了两个维度索引（行和列），但对于三维张量，最后一个维度如果没有写，
            #默认就是取全部（或者写成 inputs[excerpt, :, :]）
            Y = targets[excerpt,:]
            if (self.cuda):
                X = X.cuda()
                Y = Y.cuda()
            model_inputs = Variable(X)

            data = [model_inputs, Variable(Y)]
            yield data  # 使用 yield 而不是 return，让这个方法成为生成器,可以迭代获取批次，而不用一次性加载所有数据到内存
            
            start_idx += batch_size
        """

"""
数据集	  shuffle 应该设为	原因

训练集 (Train)	True	核心原因：打破样本间的顺序依赖。时间序列虽然有时序，但在随机梯度下降（SGD）时，              
                                如果 batch 总是按同样的顺序进入网络，模型容易产生“局部偏见”或陷入局部最优。
                                打乱顺序能让模型更健壮，提高泛化能力。

验证集 (Val)	False	验证是为了观察模型在不同阶段的稳定性。固定顺序可以让每次验证的评估结果具有严格的可比性。

测试集 (Test)	False	最重要：通常我们需要对比预测曲线和真实曲线（可视化），如果打乱了，预测结果和时间戳就对不上了。
                               此外，测试是为了模拟真实部署环境，数据通常是按顺序流入的。
"""