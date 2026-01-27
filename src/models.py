import torch
import torch.nn as nn   # nn.Module: PyTorch 神经网络基类
from torch.nn import Parameter
# 如果你想让一个张量在训练过程中被优化（更新权重），就必须把它包装成 nn.Parameter
import torch.nn.functional as F
from layers import *
from utils import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math

class cola_gnn(nn.Module):  
    def __init__(self, args, data): 
        super().__init__()
        self.x_h = 1  # 每个节点在每个时间步的输入特征维度
        self.m = data.m   # 节点数
        self.w = args.window
        if data.add_his_day:
            self.w += 1
        
        self.dropout = args.dropout
        
        # --- 核心修改：区分单向隐藏维度和最终拼接维度 ---
        self.rnn_hid = args.n_hidden  # 单层基础维度
        self.bi = args.bi
        # n_hidden 代表 RNN 输出的总维度
        self.n_hidden = (int(self.bi) + 1) * args.n_hidden 
        
        if args.rnn_model == 'LSTM':
            self.rnn = nn.LSTM(input_size=self.x_h, hidden_size=self.rnn_hid, 
                               num_layers=args.n_layer, dropout=args.dropout, 
                               batch_first=True, bidirectional=self.bi)
        elif args.rnn_model == 'GRU':
            self.rnn = nn.GRU(input_size=self.x_h, hidden_size=self.rnn_hid, 
                              num_layers=args.n_layer, dropout=args.dropout, 
                              batch_first=True, bidirectional=self.bi)
        elif args.rnn_model == 'RNN':
            self.rnn = nn.RNN(input_size=self.x_h, hidden_size=self.rnn_hid, 
                              num_layers=args.n_layer, dropout=args.dropout, 
                              batch_first=True, bidirectional=self.bi)
        else:
            raise LookupError('Only support LSTM, GRU and RNN')

        # 注意力机制参数
        half_hid = int(self.n_hidden / 2)
        self.V = Parameter(torch.Tensor(half_hid))
        self.bv = Parameter(torch.Tensor(1))
        self.W1 = Parameter(torch.Tensor(half_hid, self.n_hidden))#Ws
        self.b1 = Parameter(torch.Tensor(half_hid)) #bs
        self.W2 = Parameter(torch.Tensor(half_hid, self.n_hidden))#Wt
        self.Wb = Parameter(torch.Tensor(self.m, self.m)) #Wm
        self.wb = Parameter(torch.Tensor(1))#bm
        self.act = F.elu 

        # 空间卷积层
        self.k = args.k
        self.conv = nn.Conv1d(1, self.k, self.w) # 通过窗口大小为 w 的卷积，获取全局窗口的加权特征
        long_kernal = self.w // 2
        self.conv_long = nn.Conv1d(self.x_h, self.k, long_kernal, dilation=2)
        # dilation=2 原本卷积核覆盖下标为 [0, 1, 2] 的点，现在会覆盖 [0, 2, 4]  。捕捉到序列中的周期性或非连续的趋势

        long_out = self.w - 2 * (long_kernal - 1)
        
        self.n_spatial = 10  # 自己给的  空间特征维度
        # 注意：这里的输入维度要和后面 r_l 的维度匹配
        self.conv1 = GraphConvLayer((1 + long_out) * self.k, self.n_hidden) 
        self.conv2 = GraphConvLayer(self.n_hidden, self.n_spatial)

        self.out = nn.Linear(self.n_hidden + self.n_spatial, 1)  # nn.Linear 作用于高维张量时，只对最后一个维度进行运算。

        self.residual_window = args.res
        self.ratio = 1.0
        # 如果需要残差层，初始化
        if (self.residual_window > 0):
            self.residual_window = min(self.residual_window, args.window)
            self.residual = nn.Linear(self.residual_window, 1) 
        
        self.adj = data.adj # 对称归一化的邻接矩阵
        self.init_weights()

    def init_weights(self):
        with torch.no_grad(): # 确保初始化过程绝对不会干扰梯度链
            for p in self.parameters():
                if p.data.ndimension() >= 2:
                    nn.init.xavier_uniform_(p.data)
                    """
                    高维权重的 Xavier 初始化

                    Xavier (Glorot) 初始化：这种方法通过考虑输入和输出神经元的数量，自动调整初始值的方差。

                    核心目的：保持信号在各层间传递时方差一致，防止在深度网络中信号变得过大或过小。

                    适用场景：非常适合类似 tanh 或 sigmoid（以及 elu）这类对称的激活函数
                    """
                else:
                    stdv = 1. / math.sqrt(p.size(0))
                    p.data.uniform_(-stdv, stdv)
                    """
                    低维参数的均匀分布初始化
                    如果参数是 1 维（通常是偏置项 b1, bv, wb 或者投影向量 V）。
                    计算标准差 (stdv)：使用该维度大小倒数的平方根作为边界。
                    均匀分布 (uniform_)：在 $[-stdv, stdv]$ 范围内随机采样。
                    原因：对于偏置（Bias）或 1 维参数，通常不需要像权重矩阵那样复杂的方差缩放，简单的微小随机值即可打破对称性。
                    """

    def forward(self, x, feat=None):
        b, w, m = x.size()
        orig_x = x 
        
        # 1. Temporal Encoding (RNN)
        
        x_rnn = x.permute(0, 2, 1).contiguous().view(-1, w, 1) # 将输入变换为 (batch*m, window, 1)
        r_out, _ = self.rnn(x_rnn) # r_out: (b*m, w, n_hidden)
        
        # --- 处理双向 RNN 的状态提取 ---
        if self.bi:
            # 前向取最后一个步，后向取第一个步
            fwd_last = r_out[:, -1, :self.rnn_hid]
            bwd_last = r_out[:, 0, self.rnn_hid:]
            last_hid = torch.cat((fwd_last, bwd_last), dim=-1) 
        else:
            last_hid = r_out[:, -1, :] # (b*m, n_hidden)

        # 还原回 (batch, m, n_hidden)
        last_hid = last_hid.view(b, m, self.n_hidden)
        out_temporal = last_hid 

        # 2. Attention Matrix (Graph Structure Learning)
        hid_rpt_m = last_hid.unsqueeze(2).repeat(1, 1, m, 1) # (b, m, m, n_hidden) 每一行只包含同一个节点
        hid_rpt_w = last_hid.unsqueeze(1).repeat(1, m, 1, 1) # (b, m, m, n_hidden) 每一列只包含同一个节点
        
        a_mx = self.act(hid_rpt_m @ self.W1.t() + hid_rpt_w @ self.W2.t() + self.b1) @ self.V + self.bv 
        #self.W1.t()是转置
        a_mx = F.normalize(a_mx, p=2, dim=1)   # A

        # 3. Spatial Encoding (CNN + GCN)
        
        # 结合先验邻接矩阵
        adjs = self.adj.to(x.device).repeat(b, 1).view(b, m, m)
        c = torch.sigmoid(a_mx @ self.Wb + self.wb)
        adj = adjs * c + a_mx * (1 - c) # (b,m,m)

        # Multi-Scale Dilated Convolution
        r_l = []
        r_long_l = []
        for i in range(m):
            h_tmp = orig_x[:, :, i:i+1].permute(0, 2, 1).contiguous() # (b, 1, w)  
            # nn.Conv1d 的输入 Tensor 形状必须是：$$(\text{Batch}, \text{Channels}, \text{Length})$$
            r_l.append(self.conv(h_tmp)) # m个(b, k, 1)
            r_long_l.append(self.conv_long(h_tmp))
        
        r_l = torch.stack(r_l, dim=1) # 开辟了一个全新的维度（在这里是 dim=1），并把列表里的 m 个张量整齐地叠放进去。 (b, m, k, 1)
        r_long_l = torch.stack(r_long_l, dim=1)  # (b, m, k, long_out)
        r_l = torch.cat((r_l, r_long_l), dim=-1) # (b, m, k, 1 + long_out)
        r_l = r_l.view(b, m, -1)
        r_l = torch.relu(r_l)  # (b, m, (1 + long_out) * k)

        # Graph Convolution
        x_gcn = F.relu(self.conv1(r_l, adj))  # (b, m, n_hidden)
        x_gcn = F.dropout(x_gcn, self.dropout, training=self.training)
        """
        training=self.training: 这是最关键的开关。

        在 PyTorch 的 nn.Module 中，self.training 是一个布尔值。

        训练模式 (True): Dropout 生效，随机丢弃神经元。

        预测/验证模式 (False): Dropout 自动失效，所有神经元保持激活状态。
        """
        out_spatial = F.relu(self.conv2(x_gcn, adj))  # (b, m, n_spatial)

        # 4. Output Fusion
        out = torch.cat((out_spatial, out_temporal), dim=-1)  
        out = self.out(out).squeeze(-1) #  (b, m, n_spatial + n_hidden) -> (b, m, 1) -> (b, m)

        # 5. Residual connection
        if (self.residual_window > 0):
            z = orig_x[:, -self.residual_window:, :]
            #如果使用残差连接：从原始输入取最后residual_window个时间步
            z = z.permute(0, 2, 1).contiguous().view(-1, self.residual_window)
            #重塑为(batch*m, residual_window)格式
            z = self.residual(z).view(b, m)
            out = out * self.ratio + z

        return out, None
        



class ARMA(nn.Module): 
    def __init__(self, args, data):
        super(ARMA, self).__init__()   #这是 Python 2 风格的写法，Python 3 可以简写为 super().__init__()
        self.m = data.m
        self.w = args.window
        if data.add_his_day:
            self.w += 1
        self.n = 2 # larger worse
        self.w = 2*self.w - self.n + 1 
        self.weight = Parameter(torch.Tensor(self.w, self.m)) # 20 * 49
        self.bias = Parameter(torch.zeros(self.m)) # 49
        nn.init.xavier_normal(self.weight)

        self.output = None
        if (args.output_fun == 'sigmoid'):
            self.output = F.sigmoid;
        if (args.output_fun == 'tanh'):
            self.output = F.tanh;

    def forward(self, x):
        x_o = x
        x = x.permute(0,2,1).contiguous()
        n = self.n
        cumsum = torch.cumsum(x,dim=-1)
        cumsum[:,:,n:] = cumsum[:,:,n:] - cumsum[:,:,:-n]
        x = cumsum[:,:,n - 1:] / n
        x = x.permute(0,2,1).contiguous()
        x = torch.cat((x_o,x), dim=1)
        x = torch.sum(x * self.weight, dim=1) + self.bias
        if (self.output != None):
            x = self.output(x)
        return x, None

class AR(nn.Module):
    def __init__(self, args, data):
        super(AR, self).__init__()
        self.m = data.m
        self.w = args.window
        if data.add_his_day:
            self.w += 1
        self.weight = Parameter(torch.Tensor(self.w, self.m)) # 20 * 49
        self.bias = Parameter(torch.zeros(self.m)) # 49
        nn.init.xavier_normal(self.weight)

        self.output = None;
        if (args.output_fun == 'sigmoid'):
            self.output = F.sigmoid;
        if (args.output_fun == 'tanh'):
            self.output = F.tanh;

    def forward(self, x):
        batch_size = x.size(0);
        x = torch.sum(x * self.weight, dim=1) + self.bias
        if (self.output != None):
            x = self.output(x)
        return x,None

class VAR(nn.Module):
    def __init__(self, args, data):
        super(VAR, self).__init__()
        self.m = data.m
        self.w = args.window
        if data.add_his_day:
            self.w += 1
        self.linear = nn.Linear(self.m * self.w, self.m); 
        # 输入维度：m * w（将 w 个时间步的 m 个变量展平）
        # 输出维度：m（预测下一个时间步的 m 个变量）
        
        self.output = None;
        if (args.output_fun == 'sigmoid'):
            self.output = F.sigmoid;   
        if (args.output_fun == 'tanh'):
            self.output = F.tanh;

    def forward(self, x):
        x = x.view(-1, self.m * self.w); 
        x = self.linear(x);   # 通过线性层进行矩阵乘法：x = x @ W^T + b
        if (self.output != None):
            x = self.output(x);
        return x,None

class GAR(nn.Module):
    def __init__(self, args, data):
        super(GAR, self).__init__()
        self.m = data.m
        self.w = args.window
        if data.add_his_day:
            self.w += 1

        self.linear = nn.Linear(self.w, 1);
       
        self.output = None;
        if (args.output_fun == 'sigmoid'):
            self.output = F.sigmoid;
        if (args.output_fun == 'tanh'):
            self.output = F.tanh;

    def forward(self, x):
        batch_size = x.size(0);
        x = x.transpose(2,1).contiguous();
        x = x.view(batch_size * self.m, self.w);
        x = self.linear(x);
        x = x.view(batch_size, self.m);
        if (self.output != None):
            x = self.output(x);
        return x,None

class RNN(nn.Module):
    def __init__(self, args, data):
        super(RNN, self).__init__()
        n_input = 1
        self.m = data.m
        if args.rnn_model == 'LSTM':
            self.rnn = nn.LSTM( input_size=n_input, hidden_size=args.n_hidden, num_layers=args.n_layer, dropout=args.dropout,
                                batch_first=True, bidirectional=args.bi)
        elif args.rnn_model == 'GRU':
            self.rnn = nn.GRU( input_size=n_input, hidden_size=args.n_hidden, num_layers=args.n_layer, dropout=args.dropout,
                                batch_first=True, bidirectional=args.bi)
        elif args.rnn_model == 'RNN':
            self.rnn = nn.RNN( input_size=n_input, hidden_size=args.n_hidden, num_layers=args.n_layer, dropout=args.dropout,
                                batch_first=True, bidirectional=args.bi)
        else:
            raise LookupError(' only support LSTM, GRU and RNN')

        hidden_size = (int(args.bi) + 1) * args.n_hidden
        self.out = nn.Linear(hidden_size, 1) #n_output

    def forward(self, x):
        '''
        Args:
            x: (batch, time_step, m)  
        Returns:
            (batch, m)
        '''
        x = x.permute(0, 2, 1).contiguous().view(-1, x.size(1), 1)
        r_out, hc = self.rnn(x, None) # hidden state is the prediction TODO
        out = self.out(r_out[:,-1,:])
        out = out.view(-1, self.m)
        return out,None

class SelfAttnRNN(nn.Module):
    def __init__(self, args, data):
        super().__init__()
        self.n_input = 1
        self.m = data.m
        self.w = args.window
        if data.add_his_day:
            self.w += 1
        self.hid = args.n_hidden 
        self.rnn_cell =  nn.RNNCell(input_size=self.n_input, hidden_size=self.hid)
        self.V = Parameter(torch.Tensor(self.hid, 1))
        self.Wx = Parameter(torch.Tensor(self.hid, self.n_input))
        self.Wtlt = Parameter(torch.Tensor(self.hid, self.hid))
        self.Wh = Parameter(torch.Tensor(self.hid, self.hid))
        self.init_weights()
        self.out = nn.Linear(self.hid, 1)
    
    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data) # xavier_normal xavier_uniform_
            else:
                # nn.init.zeros_(p.data)
                stdv = 1. / math.sqrt(p.size(0))
                p.data.uniform_(-stdv, stdv)

    def forward(self, x):
        '''
        Args: x: (batch, time_step, m)  
        Returns: (batch, m)
        '''
        b, w, m = x.size()
        x = x.permute(0, 2, 1).contiguous().view(x.size(0)*x.size(2), x.size(1), self.n_input) # x, 20, 1
        Htlt = []
        H = []
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        for step in range(self.w): # forloop each history step
            x_tp1 = x[:,step,:] # [x, 1]
            if step == 0:
                hx = torch.zeros(b*m, self.hid).to(device)
                H.append(hx)
                h_tlt = torch.zeros(b*m, self.hid).to(device)
            else:
                h_tlt = Htlt[-1]
            h_his = torch.stack(H,dim=1)
            if step>0:
                x_tp1_rp = x_tp1.repeat(1,step+1).view(b*m,step+1,-1)
                h_tlt_rp = h_tlt.repeat(1,step+1).view(b*m,step+1,-1)
            else: 
                x_tp1_rp = x_tp1
                h_tlt_rp = h_tlt
            q1 = x_tp1_rp @ self.Wx.t() # [x, 20]
            q2 = h_tlt_rp @ self.Wtlt.t() # [x, 20]
            q3 = h_his @ self.Wh.t() # [x, 20]
            a = torch.tanh(q1 + q2 + q3) @ self.V # [x, 1]
            a = torch.softmax(a,dim=-1)
            h_tlt_t = h_his * a
            h_tlt_t = torch.sum(h_tlt_t,dim=1)
            Htlt.append(h_tlt_t)
            hx = self.rnn_cell(x_tp1, h_tlt_t) # [x, 20]
            H.append(hx)
        h = H[-1]
        out = self.out(h)
        out = out.view(b,m)
        return out,None

class CNNRNN_Res(nn.Module):
    def __init__(self, args, data): 
        super(CNNRNN_Res, self).__init__()
        self.ratio = 1.0   
        self.m = data.m  

        if args.rnn_model == 'LSTM':
            self.rnn = nn.LSTM( input_size=self.m, hidden_size=args.n_hidden, num_layers=args.n_layer, dropout=args.dropout, batch_first=True)
        elif args.rnn_model == 'GRU':
            self.rnn = nn.GRU( input_size=self.m, hidden_size=args.n_hidden, num_layers=args.n_layer, dropout=args.dropout, batch_first=True)
        elif args.rnn_model == 'RNN':
            self.rnn = nn.RNN( input_size=self.m, hidden_size=args.n_hidden, num_layers=args.n_layer, dropout=args.dropout, batch_first=True)
        else:
            raise LookupError(' only support LSTM, GRU and RNN')

        self.residual_window = 4

        self.mask_mat = Parameter(torch.Tensor(self.m, self.m))
        nn.init.xavier_normal(self.mask_mat)  
        self.adj = data.adj  

        self.dropout = nn.Dropout(p=args.dropout)
        self.linear1 = nn.Linear(args.n_hidden, self.m)
        if (self.residual_window > 0):
            self.residual_window = min(self.residual_window, args.window)
            self.residual = nn.Linear(self.residual_window, 1);
        
        self.output = None
        if (args.output_fun == 'sigmoid'):
            self.output = F.sigmoid
        if (args.output_fun == 'tanh'):
            self.output = F.tanh

    def forward(self, x):
        masked_adj = self.adj * self.mask_mat
        x = x.matmul(masked_adj)
        r_out, _ = self.rnn(x) #torch.Size([window, batch, n_hid]) torch.Size([batch, n_hid])
        r = self.dropout(r_out[:,-1,:])
        res = self.linear1(r) # ->[batch, m]
       
        if (self.residual_window > 0):
            z = x[:, -self.residual_window:, :]; #Step backward # [batch, res_window, m]
            z = z.permute(0,2,1).contiguous().view(-1, self.residual_window); #[batch*m, res_window]
            z = self.residual(z); #[batch*m, 1]
            z = z.view(-1,self.m); #[batch, m]
            res = res * self.ratio + z; #[batch, m]

        if self.output is not None:
            res = self.output(res).float()
        return res,None

class LSTNet(nn.Module):
    def __init__(self, args, data):
        super().__init__()
        self.use_cuda = args.cuda
        self.P = args.window;
        if data.add_his_day:
            self.P += 1
        self.m = data.m
        self.hidR = args.n_hidden;
        self.hidC = args.n_hidden;
        self.hidS = 5;
        self.Ck = 8;
        self.skip = 4;
        self.pt = (self.P - self.Ck)//self.skip
        self.hw = 4
        self.conv1 = nn.Conv2d(1, self.hidC, kernel_size = (self.Ck, self.m));
        self.GRU1 = nn.GRU(self.hidC, self.hidR);
        self.dropout = nn.Dropout(p = args.dropout);
        if (self.skip > 0):
            self.GRUskip = nn.GRU(self.hidC, self.hidS);
            self.linear1 = nn.Linear(self.hidR + self.skip * self.hidS, self.m);
        else:
            self.linear1 = nn.Linear(self.hidR, self.m);
        if (self.hw > 0):
            self.highway = nn.Linear(self.hw, 1);
        self.output = None;
        
 
    def forward(self, x):
        batch_size = x.size(0);
        
        #CNN
        c = x.view(-1, 1, self.P, self.m);
        c = F.relu(self.conv1(c));
        c = self.dropout(c);
        c = torch.squeeze(c, 3);
        
        # RNN 
        r = c.permute(2, 0, 1).contiguous();
        _, r = self.GRU1(r);
        r = self.dropout(torch.squeeze(r,0));

        
        #skip-rnn
        if (self.skip > 0):
            s = c[:,:, int(-self.pt * self.skip):].contiguous();
            # print(s.shape,self.pt)
            s = s.view(batch_size, self.hidC, self.pt, self.skip);
            s = s.permute(2,0,3,1).contiguous();
            s = s.view(self.pt, batch_size * self.skip, self.hidC);
            _, s = self.GRUskip(s);
            s = s.view(batch_size, self.skip * self.hidS);
            s = self.dropout(s);
            r = torch.cat((r,s),1);
        
        res = self.linear1(r);
        
        #highway
        if (self.hw > 0):
            z = x[:, -self.hw:, :];
            z = z.permute(0,2,1).contiguous().view(-1, self.hw);
            z = self.highway(z);
            z = z.view(-1,self.m);
            res = res + z;
            
        if (self.output):
            res = self.output(res);
        return res,None



'''STGCN'''
class TimeBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution to each node of
    a graph in isolation.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param out_channels: Desired number of output channels at each node in
        each time step.
        :param kernel_size: Size of the 1D temporal kernel.
        """
        super(TimeBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))

    def forward(self, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels)
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features_out=out_channels)
        """
        # Convert into NCHW format for pytorch to perform convolutions.
        X = X.permute(0, 3, 1, 2)
        temp = self.conv1(X) + torch.sigmoid(self.conv2(X))
        out = F.relu(temp + self.conv3(X))
        # Convert back from NCHW to NHWC
        out = out.permute(0, 2, 3, 1)
        return out


class STGCNBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution on each node in
    isolation, followed by a graph convolution, followed by another temporal
    convolution on each node.
    """

    def __init__(self, in_channels, spatial_channels, out_channels,
                 num_nodes):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param spatial_channels: Number of output channels of the graph
        convolutional, spatial sub-block.
        :param out_channels: Desired number of output features at each node in
        each time step.
        :param num_nodes: Number of nodes in the graph.
        """
        super(STGCNBlock, self).__init__()
        self.temporal1 = TimeBlock(in_channels=in_channels,
                                   out_channels=out_channels)
        self.Theta1 = nn.Parameter(torch.FloatTensor(out_channels,
                                                     spatial_channels))
        self.temporal2 = TimeBlock(in_channels=spatial_channels,
                                   out_channels=out_channels)
        self.batch_norm = nn.BatchNorm2d(num_nodes)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)

    def forward(self, X, A_hat):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features=out_channels).
        """
        t = self.temporal1(X)
        lfs = torch.einsum("ij,jklm->kilm", [A_hat, t.permute(1, 0, 2, 3)])
        # t2 = F.relu(torch.einsum("ijkl,lp->ijkp", [lfs, self.Theta1]))
        t2 = F.relu(torch.matmul(lfs, self.Theta1))
        t3 = self.temporal2(t2)
        return self.batch_norm(t3)
        # return t3


class STGCN(nn.Module):
    """
    Spatio-temporal graph convolutional network as described in
    https://arxiv.org/abs/1709.04875v3 by Yu et al.
    Input should have shape (batch_size, num_nodes, num_input_time_steps,
    num_features).
    """

    def __init__(self, args, data, num_nodes, num_features, num_timesteps_input,
                 num_timesteps_output):
        """
        :param num_nodes: Number of nodes in the graph.
        :param num_features: Number of features at each node in each time step.
        :param num_timesteps_input: Number of past time steps fed into the
        network.
        :param num_timesteps_output: Desired number of future time steps
        output by the network.
        """
        super(STGCN, self).__init__()
        self.block1 = STGCNBlock(in_channels=num_features, out_channels=args.n_hidden,
                                 spatial_channels=16, num_nodes=num_nodes)
        self.block2 = STGCNBlock(in_channels=args.n_hidden, out_channels=args.n_hidden,
                                 spatial_channels=16, num_nodes=num_nodes)
        self.last_temporal = TimeBlock(in_channels=args.n_hidden, out_channels=args.n_hidden)
        self.fully = nn.Linear((num_timesteps_input - 2 * 5) * args.n_hidden,
                               num_timesteps_output)               
        self.adj = data.adj
        
    def forward(self, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        """
 
        X = X.permute(0,2,1).contiguous()
        X = X.unsqueeze(-1)
        
        out1 = self.block1(X, self.adj)
        out2 = self.block2(out1, self.adj)
        out3 = self.last_temporal(out2)

        out4 = self.fully(out3.reshape((out3.shape[0], out3.shape[1], -1)))
       
        return out4[:, :, -1], None

class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)

class gcn(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len=3,order=2):
        super(gcn,self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):
        out = [x]
        for a in support:
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


# *************************************************************************
# 融合后的 HybridGNN 模型
# *************************************************************************
class HybridGNN(nn.Module):
    def __init__(self, args, data):
        super().__init__()
        # =====================================================================
        # 0. 通用参数和数据初始化
        # =====================================================================
        self.x_h = 1 # 输入特征维度，例如病例数
        self.m = data.m # 区域数量
        self.w = args.window # 回溯窗口大小
        if data.add_his_day:
            self.w += 1 # 如果添加了历史天数特征，则窗口加 1
        self.droprate = args.dropout # Dropout 率
        self.dropout_layer = nn.Dropout(self.droprate) # Dropout 层

        # EpiGNN 特有参数
        self.hidR = args.hidR # EpiGNN 特征嵌入维度
        self.hidA = args.hidA # Attention 隐藏维度
        self.n = args.n # GCN 层数 
        self.res = args.res # 是否使用残差连接 
       
        # ColaGNN 特有参数
        self.n_hidden = args.n_hidden # RNN 隐藏层维度
        self.bi_rnn = args.bi if hasattr(args, 'bi') else False # RNN 是否双向
       
        # 计算双向RNN的实际输出维度
        self.rnn_output_hidden_size = self.n_hidden * (2 if self.bi_rnn else 1)
        
        # 邻接矩阵
        self.adj_orig = data.orig_adj 
        self.adj_geo_normalized = data.adj
        self.degree = data.degree_adj
       
        # =====================================================================
        # 1. 时间特征提取 (h_S^C) - 使用 AdvancedTimeConv 替代
        # =====================================================================
        self.k = args.k # kernel 数量
        
        # 使用 AdvancedTimeConv 类
        self.time_conv = AdvancedTimeConv(args, data)
        
        # 获取 AdvancedTimeConv 的输出维度
        # AdvancedTimeConv 输出维度是 2k (因为融合层输出是 k*2)
        self.h_SC_dim = self.k * 2
       
        # =====================================================================
        # 2. 传输风险编码 (h_L, h_G) - 沿用 EpiGNN 的逻辑
        # =====================================================================
        # 全球传输风险 (GTR) 编码
        self.WQ = nn.Linear(self.h_SC_dim, self.hidA)
        self.WK = nn.Linear(self.h_SC_dim, self.hidA)
        self.t_enc = nn.Linear(1, self.hidR)

        # 本地传输风险 (LTR) 编码
        self.s_enc = nn.Linear(1, self.hidR)

        # =====================================================================
        # 3. 动态图构建 (A) - 使用 ColaGNN 的 RNN 和注意力机制
        # =====================================================================
        self.n_layer_rnn = args.n_layer if hasattr(args, 'n_layer') else 1
        
        if args.rnn_model == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=self.x_h, 
                hidden_size=self.n_hidden, 
                num_layers=self.n_layer_rnn, 
                dropout=self.droprate, 
                batch_first=True, 
                bidirectional=self.bi_rnn
            )
        elif args.rnn_model == 'GRU':
            self.rnn = nn.GRU(
                input_size=self.x_h, 
                hidden_size=self.n_hidden, 
                num_layers=self.n_layer_rnn, 
                dropout=self.droprate, 
                batch_first=True, 
                bidirectional=self.bi_rnn
            )
        elif args.rnn_model == 'RNN':
            self.rnn = nn.RNN(
                input_size=self.x_h, 
                hidden_size=self.n_hidden, 
                num_layers=self.n_layer_rnn, 
                dropout=self.droprate, 
                batch_first=True, 
                bidirectional=self.bi_rnn
            )
        else:
            raise LookupError('Only support LSTM, GRU and RNN for dynamic graph generation')
       
        # ColaGNN 的注意力机制参数
        self.half_hid_rnn = self.rnn_output_hidden_size // 2
        
        # 注意力参数维度
        self.V = Parameter(torch.Tensor(self.half_hid_rnn))
        self.bv = Parameter(torch.Tensor(1))
        self.W1 = Parameter(torch.Tensor(self.half_hid_rnn, self.rnn_output_hidden_size))
        self.b1 = Parameter(torch.Tensor(self.half_hid_rnn))
        self.W2 = Parameter(torch.Tensor(self.half_hid_rnn, self.rnn_output_hidden_size))
        self.act = F.elu

        # ColaGNN 的门控机制参数
        self.Wb = Parameter(torch.Tensor(self.m, self.m))
        self.wb = Parameter(torch.Tensor(1))
        
        # EpiGNN 的额外信息融合
        self.extra = args.extra if hasattr(args, 'extra') else False
        if self.extra:
            self.external = data.external
            self.external_parameter = nn.Parameter(torch.FloatTensor(self.m, self.m), requires_grad=True)
        
        # EpiGNN 的 degree gate
        self.d_gate = nn.Parameter(torch.FloatTensor(self.m, self.m), requires_grad=True)

        # =====================================================================
        # 4. 特征融合与图消息传递
        # =====================================================================
        # 初始节点特征维度
        self.initial_feature_for_gnn_dim = self.h_SC_dim + 2 * self.hidR
       
        # GNN 层
        self.GNNBlocks = nn.ModuleList([
            GraphConvLayer(in_features=self.initial_feature_for_gnn_dim,
                           out_features=self.initial_feature_for_gnn_dim)
            for _ in range(self.n)
        ])

        # =====================================================================
        # 5. 预测层 (Y_hat)
        # =====================================================================
        # GNN 最终输出的维度
        gnn_final_output_dim = self.initial_feature_for_gnn_dim * (self.n + 1) if self.res == 1 else self.initial_feature_for_gnn_dim
       
        # 最终预测层的输入维度
        self.output_layer_input_dim = gnn_final_output_dim + self.rnn_output_hidden_size
        self.output = nn.Linear(self.output_layer_input_dim, 1)

        # 残差连接
        self.residual_window = args.residual_window if hasattr(args, 'residual_window') else 0
        self.ratio = args.ratio if hasattr(args, 'ratio') else 1.0
        if self.residual_window > 0:
            self.residual_window = min(self.residual_window, args.window)
            self.residual = nn.Linear(self.residual_window, 1)

        self.init_weights()
     
    def init_weights(self):
        with torch.no_grad():
            for p in self.parameters():
                if p.data.ndimension() >= 2:
                    nn.init.xavier_uniform_(p.data) # best
                else:
                    stdv = 1. / math.sqrt(p.size(0))
                    p.data.uniform_(-stdv, stdv)
   
    def forward(self, x, isEval=False):
        # x: (batch_size, time_step=window, num_regions=m)
        batch_size, window_len, num_regions = x.size()
        orig_x = x

        # =========================================================================
        # 1. 时间特征提取 (h_S^C) - 使用 AdvancedTimeConv
        # =========================================================================
        #  直接调用 AdvancedTimeConv
        h_S_C = self.time_conv(x)  # 输出: (batch, m, 2k)
        
        # 应用 dropout（AdvancedTimeConv 内部已经有dropout，这里可以再加一层）
        h_S_C = self.dropout_layer(h_S_C)

        # =========================================================================
        # 2. 传输风险编码 (h_L, h_G)
        # =========================================================================
        # 全球传输风险 (GTR) 编码 - h_G
        query = self.WQ(h_S_C) # (batch, N, hidA)
        key = self.WK(h_S_C)   # (batch, N, hidA)
        attn_gtr = torch.bmm(query, key.transpose(1, 2)) # (batch, N, N)
        attn_gtr = F.normalize(attn_gtr, dim=-1, p=2, eps=1e-12)
        attn_sum_per_node = torch.sum(attn_gtr, dim=-1).unsqueeze(2) # (batch, N, 1)
        h_G = self.t_enc(attn_sum_per_node) # (batch, N, hidR)
        h_G = self.dropout_layer(h_G)

        # 本地传输风险 (LTR) 编码 - h_L
        d_degree = self.degree.squeeze(0).expand(batch_size, -1, 1).to(x.device) # (batch, N, 1)
        h_L = self.s_enc(d_degree) # (batch, N, hidR)
        h_L = self.dropout_layer(h_L)

        # =========================================================================
        # 3. 动态图构建 (A)
        # =========================================================================
        # RNN 隐藏状态 for 动态图 attention
        x_rnn_input_for_attention = orig_x.permute(0, 2, 1).contiguous().view(-1, window_len, self.x_h)
       
        # r_out_rnn: (batch*m, window, rnn_output_hidden_size)
        r_out_rnn, _ = self.rnn(x_rnn_input_for_attention, None)
        
        # 处理双向RNN的最后一个时间步
        if self.bi_rnn:
            forward_last = r_out_rnn[:, -1, :self.n_hidden]
            backward_first = r_out_rnn[:, 0, self.n_hidden:]
            last_hid_rnn_flat = torch.cat([forward_last, backward_first], dim=-1)
        else:
            last_hid_rnn_flat = r_out_rnn[:, -1, :]
        
        # 重塑为 (batch, m, rnn_output_hidden_size)
        last_hid_rnn = last_hid_rnn_flat.view(batch_size, self.m, self.rnn_output_hidden_size)
       
        # out_temporal_rnn will be used in the final prediction layer
        out_temporal_rnn = last_hid_rnn

        # ColaGNN 动态注意力计算
        hid_rpt_m = last_hid_rnn.repeat(1, self.m, 1).view(
            batch_size, self.m, self.m, self.rnn_output_hidden_size
        )
        hid_rpt_w = last_hid_rnn.unsqueeze(1).repeat(1, self.m, 1, 1).view(
            batch_size, self.m, self.m, self.rnn_output_hidden_size
        )
       
        # 计算注意力矩阵
        term1 = torch.matmul(hid_rpt_m, self.W1.t())  # (batch, m, m, half_hid_rnn)
        term2 = torch.matmul(hid_rpt_w, self.W2.t())  # (batch, m, m, half_hid_rnn)
        
        # 激活并计算最终注意力
        a_mx = self.act(term1 + term2 + self.b1) @ self.V + self.bv # (batch, m, m)
        a_mx = F.normalize(a_mx, p=2, dim=1, eps=1e-12)

        # 结合地理邻接和学习到的注意力
        adjs_geo = self.adj_geo_normalized.repeat(batch_size, 1, 1)  # (batch, m, m)
        adj_orig = self.adj_orig.repeat(batch_size, 1, 1)
       
        # EpiGNN 的 degree gate (d_mat)
        d_mat = torch.bmm(d_degree, d_degree.permute(0, 2, 1))  # (batch, N, N)
        d_mat = torch.mul(self.d_gate, d_mat)  # (N, N) * (batch, N, N) -> (batch, N, N)
        d_mat = torch.sigmoid(d_mat)
        spatial_adj_from_epig = torch.mul(d_mat, adj_orig)

        # ColaGNN 的门控机制
        c_gate = torch.sigmoid(a_mx @ self.Wb + self.wb)  # (batch, m, m)
       
        # 融合 EpiGNN 的地理邻接门控和 ColaGNN 的动态注意力
        adj_dynamic = (adjs_geo * c_gate + a_mx * (1 - c_gate))  # ColaGNN 风格的动态图
        adj_final_input_to_laplace = adj_dynamic + spatial_adj_from_epig  # 融合两种图

        # 获取拉普拉斯邻接矩阵
        laplace_adj = getLaplaceMat(batch_size, self.m, adj_final_input_to_laplace)
       
        # =========================================================================
        # 4. 特征融合与图消息传递
        # =========================================================================
        # 初始节点特征 h_0^(0) = Concat(h_S^C, h_L, h_G)
        node_state = torch.cat([h_S_C, h_L, h_G], dim=-1)  # (batch, m, initial_feature_for_gnn_dim)
       
        # GNN 传播
        node_state_list = [node_state]  # 用于残差连接 (res == 1)
        for layer in self.GNNBlocks:
            node_state = layer(node_state, laplace_adj)
            node_state = self.dropout_layer(node_state)
            node_state_list.append(node_state)
       
        # 最终 GNN 输出
        if self.res == 1:
            node_state_final_gnn = torch.cat(node_state_list, dim=-1)
        else:
            node_state_final_gnn = node_state

        # =========================================================================
        # 5. 预测层 (Y_hat)
        # =========================================================================
        # 拼接 GNN 最终输出和 ColaGNN RNN 最终隐藏状态
        final_prediction_input = torch.cat((node_state_final_gnn, out_temporal_rnn), dim=-1)
        res = self.output(final_prediction_input).squeeze(2)  # (batch, m)

        # 残差连接
        if self.residual_window > 0:
            z = orig_x[:, -self.residual_window:, :]  # (batch, res_window, m)
            z = z.permute(0, 2, 1).contiguous().view(-1, self.residual_window)  # (batch*m, res_window)
            z = self.residual(z)  # (batch*m, 1)
            z = z.view(-1, self.m)  # (batch, m)
            res = res * self.ratio + z

        # 如果是评估模式，返回一些中间结果
        if isEval:
            imd = (adj_dynamic, attn_gtr, laplace_adj)
        else:
            imd = None

        return res, imd




"""
# *************************************************************************
# 用了nn.GroupNorm 的 HybridGNN 模型
# *************************************************************************
class HybridGNN(nn.Module):
    def __init__(self, args, data):
        super().__init__()
        # =====================================================================
        # 0. 通用参数和数据初始化
        # =====================================================================
        self.x_h = 1 # 输入特征维度，例如病例数
        self.m = data.m # 区域数量
        self.w = args.window # 回溯窗口大小
        if data.add_his_day:
            self.w += 1 # 如果添加了历史天数特征，则窗口加 1
        self.droprate = args.dropout # Dropout 率
        self.dropout_layer = nn.Dropout(self.droprate) # Dropout 层

        # EpiGNN 特有参数
        self.hidR = args.hidR # EpiGNN 特征嵌入维度
        self.hidA = args.hidA # Attention 隐藏维度
        self.n = args.n # GCN 层数 
        self.res = args.res # 是否使用残差连接 
       
        # ColaGNN 特有参数
        self.n_hidden = args.n_hidden # RNN 隐藏层维度
        self.bi_rnn = args.bi if hasattr(args, 'bi') else False # RNN 是否双向
       
        # 计算双向RNN的实际输出维度
        self.rnn_output_hidden_size = self.n_hidden * (2 if self.bi_rnn else 1)
        
        # 地理邻接矩阵和度矩阵
        self.adj_orig = data.orig_adj 
        self.adj_geo_normalized = data.adj
        self.degree = data.degree_adj # 区域度数矩阵
       
        # =====================================================================
        # 1. 时间特征提取 (h_S^C) - 改进的多尺度卷积
        # =====================================================================
        self.k = args.k # kernel 数量
        
        # 改进1：添加BatchNorm和Pooling的卷积模块
        # 短期卷积分支
        self.conv_short = nn.Conv1d(self.x_h, self.k, self.w) 
        self.gn_short = nn.GroupNorm(num_groups=self.k, num_channels=self.k) 

        # 长期卷积分支
        long_kernel = self.w // 2
        long_output_len = self.w - 2 * (long_kernel - 1)  # 计算输出长度
        
        self.conv_long = nn.Conv1d(self.x_h, self.k, long_kernel, dilation=2)
        self.gn_long = nn.GroupNorm(num_groups=self.k, num_channels=self.k)
        
        # 自适应池化（统一输出维度）
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # 计算 h_S^C 的维度（添加池化后，每个通道输出长度为1）
        self.h_SC_dim = 2 * self.k  # 短期和长期各k维，拼接后为2k
        
        # =====================================================================
        # 2. 传输风险编码 (h_L, h_G) - 沿用 EpiGNN 的逻辑
        # =====================================================================
        # 全球传输风险 (GTR) 编码
        self.WQ = nn.Linear(self.h_SC_dim, self.hidA)
        self.WK = nn.Linear(self.h_SC_dim, self.hidA)
        self.t_enc = nn.Linear(1, self.hidR)

        # 本地传输风险 (LTR) 编码
        self.s_enc = nn.Linear(1, self.hidR)

        # =====================================================================
        # 3. 动态图构建 (A) - 使用 ColaGNN 的 RNN 和注意力机制
        # =====================================================================
        # ColaGNN 的 RNN 模块
        self.n_layer_rnn = args.n_layer if hasattr(args, 'n_layer') else 1
        
        if args.rnn_model == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=self.x_h, 
                hidden_size=self.n_hidden, 
                num_layers=self.n_layer_rnn, 
                dropout=self.droprate, 
                batch_first=True, 
                bidirectional=self.bi_rnn
            )
        elif args.rnn_model == 'GRU':
            self.rnn = nn.GRU(
                input_size=self.x_h, 
                hidden_size=self.n_hidden, 
                num_layers=self.n_layer_rnn, 
                dropout=self.droprate, 
                batch_first=True, 
                bidirectional=self.bi_rnn
            )
        elif args.rnn_model == 'RNN':
            self.rnn = nn.RNN(
                input_size=self.x_h, 
                hidden_size=self.n_hidden, 
                num_layers=self.n_layer_rnn, 
                dropout=self.droprate, 
                batch_first=True, 
                bidirectional=self.bi_rnn
            )
        else:
            raise LookupError('Only support LSTM, GRU and RNN for dynamic graph generation')
       
        # ColaGNN 的注意力机制参数
        self.half_hid_rnn = self.rnn_output_hidden_size // 2
        
        # 注意力参数维度
        self.V = Parameter(torch.Tensor(self.half_hid_rnn))
        self.bv = Parameter(torch.Tensor(1))
        self.W1 = Parameter(torch.Tensor(self.half_hid_rnn, self.rnn_output_hidden_size))
        self.b1 = Parameter(torch.Tensor(self.half_hid_rnn))
        self.W2 = Parameter(torch.Tensor(self.half_hid_rnn, self.rnn_output_hidden_size))
        self.act = F.elu # ELU 激活函数

        # ColaGNN 的门控机制参数
        self.Wb = Parameter(torch.Tensor(self.m, self.m))
        self.wb = Parameter(torch.Tensor(1))
        
        # EpiGNN 的额外信息融合
        self.extra = args.extra if hasattr(args, 'extra') else False
        if self.extra:
            self.external = data.external
            self.external_parameter = nn.Parameter(torch.FloatTensor(self.m, self.m), requires_grad=True)
        
        # EpiGNN 的 degree gate
        self.d_gate = nn.Parameter(torch.FloatTensor(self.m, self.m), requires_grad=True)

        # =====================================================================
        # 4. 特征融合与图消息传递 - 改进的GCN层
        # =====================================================================
        # 初始节点特征维度
        self.initial_feature_for_gnn_dim = self.h_SC_dim + 2 * self.hidR
       
        # 改进2：创建带激活函数的GCN层
        self.GNNBlocks = nn.ModuleList([
            self._create_gnn_layer() for _ in range(self.n)
        ])

        # =====================================================================
        # 5. 预测层 (Y_hat)
        # =====================================================================
        # GNN 最终输出的维度
        gnn_final_output_dim = self.initial_feature_for_gnn_dim * (self.n + 1) if self.res == 1 else self.initial_feature_for_gnn_dim
       
        # 最终预测层的输入维度
        self.output_layer_input_dim = gnn_final_output_dim + self.rnn_output_hidden_size
        self.output = nn.Linear(self.output_layer_input_dim, 1)

        # 残差连接
        self.residual_window = args.residual_window if hasattr(args, 'residual_window') else 0
        self.ratio = args.ratio if hasattr(args, 'ratio') else 1.0
        if self.residual_window > 0:
            self.residual_window = min(self.residual_window, args.window)
            self.residual = nn.Linear(self.residual_window, 1)

        self.init_weights()
    
    def _create_gnn_layer(self):
        
        return nn.Sequential(
            GraphConvLayer(self.initial_feature_for_gnn_dim, self.initial_feature_for_gnn_dim),
            nn.ReLU(),
            nn.Dropout(self.droprate)
        )
     
    def init_weights(self):
        with torch.no_grad():
            for p in self.parameters():
                if p.data.ndimension() >= 2:
                    nn.init.xavier_uniform_(p.data)
                else:
                    stdv = 1. / math.sqrt(p.size(0))
                    p.data.uniform_(-stdv, stdv)
   
    def forward(self, x, isEval=False):
        # x: (batch_size, time_step=window, num_regions=m)
        batch_size, window_len, num_regions = x.size()
        orig_x = x

        # =========================================================================
        # 1. 时间特征提取 (h_S^C) - 改进版
        # =========================================================================

        # 在 __init__ 中，你需要定义支持并行计算的层
        # 假设 k 是卷积核数量，num_regions 是区域总数
        # self.bn_short 和 self.bn_long 替换为 GroupNorm 或 LayerNorm

        # --- forward 内部修改 ---
        batch_size, window_len, num_regions = x.size()
        orig_x = x 

        # 1. 维度准备：将 Batch 和 Regions 合并到一起
        # (batch, window, num_regions) -> (batch, num_regions, window) -> (batch * num_regions, 1, window)
        x_reshaped = orig_x.permute(0, 2, 1).contiguous().view(-1, 1, window_len)

        # 2. 短期卷积分支
        short = self.conv_short(x_reshaped)  # 输出 (batch * num_regions, k, 1)
        # 使用 GroupNorm 替代 BatchNorm：num_groups 设为 num_channels 即可实现 Instance/Layer 级别的独立归一化
        short = self.gn_short(short) 
        short = F.relu(short)
        short = self.pool(short)  # (batch * num_regions, k, 1)

        # 3. 长期卷积分支
        long = self.conv_long(x_reshaped)   # 输出 (batch * num_regions, k, long_out_len)
        long = self.gn_long(long)
        long = F.relu(long)
        long = self.pool(long)   # (batch * num_regions, k, 1)

        # 4. 特征拼接与维度还原
        # (batch * num_regions, 2*k, 1)
        combined = torch.cat([short, long], dim=1)

        # 还原回 (batch, num_regions, 2*k)
        h_S_C = combined.view(batch_size, num_regions, -1)

        # 5. 最后应用 Dropout
        h_S_C = self.dropout_layer(h_S_C)

        # =========================================================================
        # 2. 传输风险编码 (h_L, h_G)
        # =========================================================================
        # 全球传输风险 (GTR) 编码
        query = self.WQ(h_S_C)  # (batch, N, hidA)
        key = self.WK(h_S_C)    # (batch, N, hidA)
        attn_gtr = torch.bmm(query, key.transpose(1, 2))  # (batch, N, N)
        attn_gtr = F.normalize(attn_gtr, dim=-1, p=2, eps=1e-12)
        attn_sum_per_node = torch.sum(attn_gtr, dim=-1).unsqueeze(2)  # (batch, N, 1)
        h_G = self.t_enc(attn_sum_per_node)  # (batch, N, hidR)
        h_G = self.dropout_layer(h_G)

        # 本地传输风险 (LTR) 编码
        d_degree = self.degree.squeeze(0).expand(batch_size, -1, 1).to(x.device)  # (batch, N, 1)
        h_L = self.s_enc(d_degree)  # (batch, N, hidR)
        h_L = self.dropout_layer(h_L)

        # =========================================================================
        # 3. 动态图构建 (A)
        # =========================================================================
        # RNN处理
        x_rnn_input = orig_x.permute(0, 2, 1).contiguous().view(-1, window_len, self.x_h)
        r_out_rnn, _ = self.rnn(x_rnn_input, None)
        
        # 处理双向RNN输出
        if self.bi_rnn:
            forward_last = r_out_rnn[:, -1, :self.n_hidden]
            backward_first = r_out_rnn[:, 0, self.n_hidden:]
            last_hid_rnn_flat = torch.cat([forward_last, backward_first], dim=-1)
        else:
            last_hid_rnn_flat = r_out_rnn[:, -1, :]
        
        last_hid_rnn = last_hid_rnn_flat.view(batch_size, self.m, self.rnn_output_hidden_size)
        out_temporal_rnn = last_hid_rnn

        # ColaGNN动态注意力
        hid_rpt_m = last_hid_rnn.repeat(1, self.m, 1).view(
            batch_size, self.m, self.m, self.rnn_output_hidden_size
        )
        hid_rpt_w = last_hid_rnn.unsqueeze(1).repeat(1, self.m, 1, 1).view(
            batch_size, self.m, self.m, self.rnn_output_hidden_size
        )
       
        term1 = torch.matmul(hid_rpt_m, self.W1.t())  # (batch, m, m, half_hid_rnn)
        term2 = torch.matmul(hid_rpt_w, self.W2.t())  # (batch, m, m, half_hid_rnn)
        
        a_mx = self.act(term1 + term2 + self.b1) @ self.V + self.bv  # (batch, m, m)
        a_mx = F.normalize(a_mx, p=2, dim=1, eps=1e-12)

        # 结合地理邻接
        adjs_geo = self.adj_geo_normalized.repeat(batch_size, 1, 1)
        adj_orig = self.adj_orig.repeat(batch_size, 1, 1)
       
        # EpiGNN的degree gate
        d_mat = torch.bmm(d_degree, d_degree.permute(0, 2, 1))
        d_mat = torch.mul(self.d_gate, d_mat)
        d_mat = torch.sigmoid(d_mat)
        spatial_adj_from_epig = torch.mul(d_mat, adj_orig)

        # ColaGNN门控机制
        c_gate = torch.sigmoid(a_mx @ self.Wb + self.wb)
       
        # 融合动态图和地理图
        adj_dynamic = adjs_geo * c_gate + a_mx * (1 - c_gate)
        adj_final = adj_dynamic + spatial_adj_from_epig

        # 改进3：使用带自环的拉普拉斯矩阵计算
        laplace_adj = getLaplaceMat(batch_size, self.m, adj_final)

        # =========================================================================
        # 4. 特征融合与图消息传递
        # =========================================================================
        node_state = torch.cat([h_S_C, h_L, h_G], dim=-1)  # (batch, m, initial_feature_for_gnn_dim)
       
        # GNN传播
        node_state_list = [node_state]
        for gnn_block in self.GNNBlocks:
            # GNN块已包含激活和dropout
            node_state = gnn_block[0](node_state, laplace_adj)  # GraphConvLayer
            node_state = gnn_block[1](node_state)  # ReLU
            node_state = gnn_block[2](node_state)  # Dropout
            node_state_list.append(node_state)
       
        # 最终GNN输出
        if self.res == 1:
            node_state_final_gnn = torch.cat(node_state_list, dim=-1)
        else:
            node_state_final_gnn = node_state

        # =========================================================================
        # 5. 预测层
        # =========================================================================
        final_prediction_input = torch.cat((node_state_final_gnn, out_temporal_rnn), dim=-1)
        res = self.output(final_prediction_input).squeeze(2)  # (batch, m)

        # 残差连接
        if self.residual_window > 0:
            z = orig_x[:, -self.residual_window:, :]
            z = z.permute(0, 2, 1).contiguous().view(-1, self.residual_window)
            z = self.residual(z)
            z = z.view(-1, self.m)
            res = res * self.ratio + z

        if isEval:
            imd = (adj_dynamic, attn_gtr, laplace_adj)
        else:
            imd = None

        return res, imd
"""
