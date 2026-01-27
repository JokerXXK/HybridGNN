import os, random, argparse, time # 系统、随机数、命令行参数、时间
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score,explained_variance_score # 评估指标
from math import sqrt
from scipy.stats import pearsonr # 统计相关性计算
from models import *
from data import *
import torch  # PyTorch深度学习框架
import torch.nn.functional as F # PyTorch函数操作
from dcrnn_model import *
import logging # 日志记录
#点击launch后定位到tensorboard文件夹  可以在vscode中打开tensorboard查看训练过程
#tensorboard --logdir=./tensorboard --port=6007   #启动tensorboard服务  端口6007  可以在浏览器中访问localhost:6007查看
from tensorboardX import SummaryWriter  # 可视化工具   
import shutil # 用于处理文件夹的整体复制、移动和递归删除
import warnings
warnings.filterwarnings("ignore", category=UserWarning)  #解决RNN层数为1的时候的warning问题

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器  用于输出信息
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s') # 输出格式include timestamp

# Training settings
ap = argparse.ArgumentParser() # argparse允许用户通过命令行传递参数，方便实验配置   创建参数解析器对象
ap.add_argument('--dataset', type=str, default='japan', help="Dataset string")
ap.add_argument('--sim_mat', type=str, default='japan-adj', help="adjacency matrix filename (*-adj.txt)")
ap.add_argument('--sparse', action='store_true', default=True, help="Whether to use sparse tensor")
ap.add_argument('--output_fun', type=str, default=None, choices=[None, 'sigmoid', 'tanh'], help="Output function")
ap.add_argument('--n_layer', type=int, default=2, help="number of RNN layers (default 1)") 
ap.add_argument('--n_hidden', type=int, default=20, help="rnn hidden states (could be set as any value)") 
ap.add_argument('--seed', type=int, default=42, help='random seed')
ap.add_argument('--epochs', type=int, default=1500, help='number of epochs to train')
ap.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
ap.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay (L2 loss on parameters).')
ap.add_argument('--dropout', type=float, default=0.2, help='dropout rate usually 0.2-0.5.')
ap.add_argument('--batch', type=int, default=128, help="batch size")
ap.add_argument('--shuffle', action='store_true', default=False, help="not used, default false")
ap.add_argument('--train', type=float, default=.7, help="Training ratio (0, 1)")
ap.add_argument('--val', type=float, default=.15, help="Validation ratio (0, 1)")
ap.add_argument('--test', type=float, default=.15, help="Testing ratio (0, 1)")
ap.add_argument('--model', default='cola_gnn', choices=['HybridGNN','cola_gnn','CNNRNN_Res','RNN','AR','ARMA','VAR','GAR','SelfAttnRNN','lstnet','stgcn','dcrnn'], help='')
ap.add_argument('--rnn_model', default='RNN', choices=['LSTM','RNN','GRU'], help='')
ap.add_argument('--mylog', action='store_false', default=True,  help='save tensorboad log')
ap.add_argument('--cuda', action='store_true', default=True,  help='')
ap.add_argument('--window', type=int, default=20, help='')
ap.add_argument('--horizon', type=int, default=32, help='leadtime default 1')
ap.add_argument('--save_dir', type=str,  default='save',help='dir path to save the final model')
ap.add_argument('--gpu', type=int, default=0,  help='choose gpu 0-10')
ap.add_argument('--lamda', type=float, default=0.01,  help='regularize params similarities of states')
ap.add_argument('--bi', action='store_true', default=False,  help='bidirectional default false')
ap.add_argument('--patience', type=int, default=100, help='patience default 100')
ap.add_argument('--k', type=int, default=8,  help='kernels')

ap.add_argument('--check_point', type=int, default=1, help="check point")   # not used
ap.add_argument('--hidsp', type=int, default=15,  help='spatial dim')       # not used

ap.add_argument('--hidR', type=int, default=64,  help='hidden dim')
ap.add_argument('--hidA', type=int, default=64,  help='hidden dim of attention layer')
ap.add_argument('--n', type=int, default=2, help='layer number of GCN')
ap.add_argument('--res', type=int, default=0, help='0 means no residual link while 1 means need residual link')

args = ap.parse_args() 

#print('--------Parameters--------')
#print(args)
#print('--------------------------')

os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)  #设置GPU环境变量  控制哪些GPU对PyTorch可见

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

args.cuda = args.cuda and torch.cuda.is_available() #  如果args.cuda为True且系统有CUDA设备，则使用GPU
logger.info('cuda %s', args.cuda)   # 输出是否使用cuda

# 创建模型名称标识符，包含模型名、数据集、窗口大小、预测步长、RNN类型  存在tensorboard目录下
log_token = '%s.%s.w-%s.h-%s.%s' % (args.model, args.dataset, args.window, args.horizon, args.rnn_model)

if args.mylog:
    tensorboard_log_dir = 'tensorboard/%s' % (log_token)  # 日志保存目录 比如tensorboard/HybridGNN.japan.w-20.h-19.RNN
    
    # 第一步：先清理旧垃圾
    if os.path.exists(tensorboard_log_dir): #判断 tensorboard/模型名.数据集... 这个文件夹是否已经存在。如果这是你第二次运行同一个实验（参数没变），这个文件夹肯定存在。
        shutil.rmtree(tensorboard_log_dir) # 先删除旧的
        logger.info('Cleaned up old logs in %s', tensorboard_log_dir)

    # 第二步：确保目录存在（rmtree 删掉后这里需要重建）
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    
    # 第三步：最后开启写入器
    writer = SummaryWriter(tensorboard_log_dir) #实例化 TensorBoard 写入对象  SummaryWriter 会在指定的路径下创建一个以 events.out.tfevents... 开头的二进制文件。
    logger.info('tensorboard logging to %s', tensorboard_log_dir)
    print("*"*84)

data_loader = DataBasicLoader(args)


if args.model == 'CNNRNN_Res':
    model = CNNRNN_Res(args, data_loader)  
elif args.model == 'RNN':
    model = RNN(args, data_loader)
elif args.model == 'AR':
    model = AR(args, data_loader)
elif args.model == 'ARMA':
    model = ARMA(args, data_loader)
elif args.model == 'VAR':
    model = VAR(args, data_loader)
elif args.model == 'GAR':
    model = GAR(args, data_loader)
elif args.model == 'SelfAttnRNN':
    model = SelfAttnRNN(args, data_loader)
elif args.model == 'lstnet':
    model = LSTNet(args, data_loader)      
elif args.model == 'stgcn':
    model = model = STGCN(args, data_loader, data_loader.m, 1, data_loader.train[0].shape[1], args.horizon ) 
elif args.model == 'dcrnn':
    model = DCRNNModel(args, data_loader)   
elif args.model == 'cola_gnn':
    model = cola_gnn(args, data_loader)  
elif args.model == 'HybridGNN':
    model = HybridGNN(args, data_loader)        
else: 
    raise LookupError('can not find the model')
 
logger.info('model %s', model)
if args.cuda:
    model.cuda()     # 将模型移动到GPU上进行计算
# Adam优化器，过滤掉不需要梯度的参数，设置学习率和权重衰减  # 只优化需要梯度的参数
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)  #计算模型参数量（用于了解模型复杂度）
print("*"*84)
print('#params:',pytorch_total_params)
print("*"*84)

def evaluate(data_loader, data, tag='val'):
    
    model.eval()  # 评估模式  禁用 dropout、固定 batch norm 的统计量、不计算梯度
    n_samples = 0.
    total_loss = 0.   
    batch_size = args.batch

    y_pred_mx = [] 
    y_true_mx = [] 

    for inputs in data_loader.get_batches(data, batch_size, False):
        X, Y = inputs[0], inputs[1]
        output,_  = model(X)

        loss_train = F.l1_loss(output, Y) 
        #全局拍平：它把 (Batch, m) 看作一个拥有 Batch * m 个元素的超长向量。
        #算绝对误差：计算每个位置的 $|y_{pred} - y_{true}|$。
        #求总平均：将所有误差加起来，除以 (Batch * m)

        total_loss += loss_train.item()*(output.size(0) * data_loader.m)
        n_samples += (output.size(0) * data_loader.m); # 细化到节点的n_samples，区别于batch size       

        y_true_mx.append(Y.data.cpu())  #.data 获取张量的数据部分
        y_pred_mx.append(output.data.cpu())

    y_pred_mx = torch.cat(y_pred_mx) # 将列表中所有的预测张量沿着第0维（batch维度）拼接成一个大的张量
    y_true_mx = torch.cat(y_true_mx) # (所有 Batch 大小的总和, 节点数)

    #反归一化：将标准化后的数据还原为原始尺度    公式: 原始值 = 标准化值 × (max - min) + min
    #维度为(所有 Batch 大小的总和, 节点数)
    y_true_states = y_true_mx.numpy() * (data_loader.max - data_loader.min ) * 1.0 + data_loader.min   
    y_pred_states = y_pred_mx.numpy() * (data_loader.max - data_loader.min ) * 1.0 + data_loader.min  

    # 计算按节点的评估指标
    # 按节点平均的 RMSE
    rmse_states = np.mean(np.sqrt(mean_squared_error(y_true_states, y_pred_states, multioutput='raw_values'))) 

    """
    1. mean_squared_error(..., multioutput='raw_values')
    通常 sklearn 的 MSE 函数会返回一个单一的平均值。
    但因为设置了 multioutput='raw_values'，它的行为发生了变化：
    输入维度：y_true_states 和 y_pred_states 的形状都是 (N, m)。
    计算逻辑：它会针对 $m$ 列中的每一列（即每一个节点）独立计算 MSE。
    输出结果：一个形状为 (m,) 的数组，其中每个元素代表该节点的均方误差。
    2. np.sqrt(...)操作：对上一步得到的 (m,) 数组中的每一个元素开平方。
    物理意义：将单位从“平方单位”还原回“原始单位”。
    输出结果：一个形状为 (m,) 的数组，代表每个节点的独立 RMSE。
    比如：[节点1的RMSE, 节点2的RMSE, ..., 节点m的RMSE]
    3. np.mean(...)操作：对这 $m$ 个 RMSE 值求算术平均。
    物理意义：得到一个标量，代表整个图网络在当前测试集上的综合表现。
    """

    raw_mae = mean_absolute_error(y_true_states, y_pred_states, multioutput='raw_values')  # (N, m) -> (m,)
    std_mae = np.std(raw_mae) # Standard deviation of MAEs for all states/places     (m,)-> scalar


    pcc_tmp = []
    for k in range(data_loader.m):
        true_k = y_true_states[:,k]  #(N, m) 降维为 (N,)
        pred_k = y_pred_states[:,k]
        
        # 跳过常数列或包含NaN的列
        if (np.all(true_k == true_k[0]) or   # 条件1：真实值都是常数
            np.all(pred_k == pred_k[0]) or   # 条件2：预测值都是常数
            np.isnan(true_k).any() or        # 条件3：真实值包含NaN
            np.isnan(pred_k).any()):         # 条件4：预测值包含NaN
            continue  # 跳过该列
            
        with warnings.catch_warnings():  # 临时忽略警告
            warnings.simplefilter("ignore")
            pcc_tmp.append(pearsonr(true_k, pred_k)[0])
            #长度为 m 的 Python 列表。其中每个元素都是一个节点独立计算出的相关性评分
    
    # 计算均值时处理可能为空的情况
    pcc_states = np.mean(pcc_tmp) if len(pcc_tmp) > 0 else 0.0 

    
    r2_states = np.mean(r2_score(y_true_states, y_pred_states, multioutput='raw_values'))
    var_states = np.mean(explained_variance_score(y_true_states, y_pred_states, multioutput='raw_values'))



    #计算全局指标
    # convert y_true & y_pred to real data
    y_true = np.reshape(y_true_states,(-1))  # 展平为一维数组
    y_pred = np.reshape(y_pred_states,(-1))

    #rmse (全局 RMSE) 这是将所有时间点、所有节点的数据全部“拍平”成一个超长向量后的结果。
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    pcc = pearsonr(y_true,y_pred)[0]
    r2 = r2_score(y_true, y_pred,multioutput='uniform_average') #variance_weighted 
    var = explained_variance_score(y_true, y_pred, multioutput='uniform_average')
    peak_mae = peak_error(y_true_states.copy(), y_pred_states.copy(), data_loader.peak_thold)

    global y_true_t  #全局变量  用于外部访问
    global y_pred_t
    y_true_t = y_true_states
    y_pred_t = y_pred_states
    return float(total_loss / n_samples), mae,std_mae, rmse, rmse_states, pcc, pcc_states, r2, r2_states, var, var_states, peak_mae

def train(data_loader, data):
    model.train()
    # model.train() 是 PyTorch 中用来设置模型为训练模式的方法。
    # 启用 dropout、batch norm 等层的训练行为
    total_loss = 0.
    n_samples = 0.
    batch_size = args.batch

    for inputs in data_loader.get_batches(data, batch_size, True):
        X, Y = inputs[0], inputs[1]
        #    X: 输入特征，形状通常是 [batch_size, , ]
        #    Y: 目标标签/值，形状通常是 [batch_size, ...]

        optimizer.zero_grad()
        # 清空梯度缓存 防止梯度在多个batch间累积（PyTorch默认会累加梯度）
        output,_  = model(X) 
        
        loss_train = F.l1_loss(output, Y) 
        # $$Loss = \frac{1}{N} \sum_{i=1}^{N} |output_i - Y_i|$$这里的 $N$ 是张量中所有元素的总数（即 $batch\_size \times num\_regions$）
        # 当你调用 F.l1_loss(output, Y) 时，PyTorch 内部并不关心它是哪一维，它会把这两个张量看作两个包含 $32 \times 20 = 640$ 个数字的列表。
        #    F.l1_loss: 平均绝对误差损失 (MAE, Mean Absolute Error)
        #    先逐个元素求差值，再求绝对值，最后取平均
        #    loss = mean(|output - Y|)
        #    如果要换为MSE损失，用 F.mse_loss()

        current_batch_elements = output.size(0) * data_loader.m  # 当前 Batch 的总点数
        total_loss += loss_train.item() * current_batch_elements # 只乘当前 Batch 的规模
        n_samples += current_batch_elements                     # 累加总点数
        loss_train.backward()   #反向传播 计算损失相对于模型参数的梯度
        optimizer.step()  # 根据梯度更新模型权重

    return float(total_loss / n_samples)
 
bad_counter = 0  # bad_counter == args.patience -> early stop
best_epoch = 0
best_val = 1e+20;
try:
    print('begin training: ');
    if not os.path.exists(args.save_dir):  # 创建save文件夹
        os.makedirs(args.save_dir)
    
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train_loss = train(data_loader, data_loader.train)
        val_loss, mae,std_mae, rmse, rmse_states, pcc, pcc_states, r2, r2_states, var, var_states, peak_mae = evaluate(data_loader, data_loader.val)
        #print('Epoch {:3d}|time:{:5.2f}s|train_loss {:5.4f}|val_loss {:5.4f}'.format(epoch, (time.time() - epoch_start_time), train_loss, val_loss))

        # 如果启用日志记录，将指标写入TensorBoard
        if args.mylog:
            # 添加各种指标到TensorBoard
            writer.add_scalars('data/loss', {'train': train_loss}, epoch )
            writer.add_scalars('data/loss', {'val': val_loss}, epoch)
            writer.add_scalars('data/mae', {'val': mae}, epoch)
            writer.add_scalars('data/rmse', {'val': rmse_states}, epoch)
            writer.add_scalars('data/rmse_states', {'val': rmse_states}, epoch)
            writer.add_scalars('data/pcc', {'val': pcc}, epoch)
            writer.add_scalars('data/pcc_states', {'val': pcc_states}, epoch)
            writer.add_scalars('data/R2', {'val': r2}, epoch)
            writer.add_scalars('data/R2_states', {'val': r2_states}, epoch)
            writer.add_scalars('data/var', {'val': var}, epoch)
            writer.add_scalars('data/var_states', {'val': var_states}, epoch)
            writer.add_scalars('data/peak_mae', {'val': peak_mae}, epoch)
       
        # Save the model if the validation loss is the best we've seen so far.
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            bad_counter = 0
            model_path = '%s/%s.pt' % (args.save_dir, log_token) # 模型保存路径
            with open(model_path, 'wb') as f:
                torch.save(model.state_dict(), f)    # 只保存模型参数  覆盖之前的最佳模型
            test_loss, mae,std_mae, rmse, rmse_states, pcc, pcc_states, r2, r2_states, var, var_states, peak_mae  = evaluate(data_loader, data_loader.test,tag='test')
            # 打印 Best validation epoch 的测试集结果. B_Epo: Best Epoch    T_Loss: Test Loss
            print(f"B_Epo: {epoch:03d} | T_Loss: {test_loss:.3f} | " 
                f"MAE/s: {mae:.0f}/{std_mae:.0f} | "
                f"RMSE/s: {rmse:.0f}/{rmse_states:.0f} | "
                f"PCC/s: {pcc:.3f}/{pcc_states:.3f} | "
                f"R2/s: {r2:.3f}/{r2_states:.3f} | "
                f"Var/s: {var:.3f}/{var_states:.3f} | "
                f"Peak: {peak_mae:.0f}")
        else:
            bad_counter += 1

        if bad_counter == args.patience:
            break

except KeyboardInterrupt:   # 捕获键盘中断（Ctrl+C）以便优雅退出
    print('-' * 89)
    print('Exiting from training early, epoch',epoch)

# Load the best saved model.
model_path = '%s/%s.pt' % (args.save_dir, log_token)
with open(model_path, 'rb') as f:
    model.load_state_dict(torch.load(f));   ## 加载模型参数
test_loss, mae,std_mae, rmse, rmse_states, pcc, pcc_states, r2, r2_states, var, var_states, peak_mae  = evaluate(data_loader, data_loader.test,tag='test')
print("*"*84)
print('Final evaluation')
print(f"T_Loss: {test_loss:.3f} | "
                f"MAE/s: {mae:.0f}/{std_mae:.0f} | "
                f"RMSE/s: {rmse:.0f}/{rmse_states:.0f} | "
                f"PCC/s: {pcc:.3f}/{pcc_states:.3f} | "
                f"R2/s: {r2:.3f}/{r2_states:.3f} | "
                f"Var/s: {var:.3f}/{var_states:.3f} | "
                f"Peak: {peak_mae:.0f}")