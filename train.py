import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 指定使用第0块GPU（仅1块可用）
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import models # 导入TorchVision预训练模型
from torch.autograd import Variable # 自动求导（PyTorch旧版本需要，新版本可省略）

# 导入自定义模块（需根据实际路径确认）
from net import vgg16, vgg16_bn # 自定义VGG网络结构
from resnet_yolo import resnet50, resnet18 # 自定义ResNet网络结构（适配YOLO）
from yoloLoss import yoloLoss # 自定义YOLO损失函数
from dataset import yoloDataset  # 自定义YOLO数据集类

from visualize import Visualizer # 自定义可视化工具（如Visdom）
import numpy as np  # 数值计算库


def train():
    use_gpu = torch.cuda.is_available()

    file_root = './VOCdevkit/VOC2012/JPEGImages/'
    learning_rate = 0.001 # 初始学习率
    num_epochs = 50  # 总训练轮数
    batch_size = 24  # 批次大小（根据GPU内存调整）
    use_resnet = True # 设为True使用ResNet50，否则使用VGG16 BN
    if use_resnet:
        net = resnet50() # 自定义ResNet50（适配YOLO输出）
    else:
        net = vgg16_bn() # 自定义VGG16 BN（适配YOLO输出）
    # net.classifier = nn.Sequential(
    #             nn.Linear(512 * 7 * 7, 4096),
    #             nn.ReLU(True),
    #             nn.Dropout(),
    #             #nn.Linear(4096, 4096),
    #             #nn.ReLU(True),
    #             #nn.Dropout(),
    #             nn.Linear(4096, 1470),
    #         )
    #net = resnet18(pretrained=True)
    #net.fc = nn.Linear(512,1470)
    # initial Linear
    # for m in net.modules():
    #     if isinstance(m, nn.Linear):
    #         m.weight.data.normal_(0, 0.01)
    #         m.bias.data.zero_()
    print(net)
    #net.load_state_dict(torch.load('yolo.pth'))
    print('load pre-trined model')
    if use_resnet:
        resnet = models.resnet50(pretrained=True) # 加载官方预训练ResNet50
        new_state_dict = resnet.state_dict() # 预训练模型参数
        dd = net.state_dict() # 当前模型参数
        # 复制预训练参数（跳过全连接层，仅加载特征提取层）
        for k in new_state_dict.keys():
            print(k)
            if k in dd.keys() and not k.startswith('fc'):
                print('yes')
                dd[k] = new_state_dict[k]
        net.load_state_dict(dd)
    else:
        vgg = models.vgg16_bn(pretrained=True)
        new_state_dict = vgg.state_dict()
        dd = net.state_dict()
        for k in new_state_dict.keys():
            print(k)
            if k in dd.keys() and k.startswith('features'):
                print('yes')
                dd[k] = new_state_dict[k]
        net.load_state_dict(dd)
    if False:
        net.load_state_dict(torch.load('best.pth'))
    print('cuda', torch.cuda.current_device(), torch.cuda.device_count())

    # 定义YOLO损失函数（参数：网格数7，锚框数2，类别数5？需根据实际调整）
    criterion = yoloLoss(7,2,5,0.5) # 假设参数：S=7（网格数），B=2（锚框数），C=5（类别数？可能需修正）
    if use_gpu:
        net.cuda()

    net.train()  # 开启训练模式（影响BatchNorm、Dropout等层）
    # different learning rate
    params=[]
    params_dict = dict(net.named_parameters())  # 获取模型所有参数（名称: 参数张量）
    for key,value in params_dict.items():
        if key.startswith('features'):  # 特征提取层（如卷积层）
            params += [{'params':[value],'lr':learning_rate*1}] # 学习率×1（较低）
        else:
            params += [{'params':[value],'lr':learning_rate}]  # 学习率×1（较高）
    # 定义优化器（SGD，带动量和权重衰减）
    optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    # optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate,weight_decay=1e-4)

    # train_dataset = yoloDataset(root=file_root,list_file=['voc12_trainval.txt','voc07_trainval.txt'],train=True,transform = [transforms.ToTensor()] )
    # 定义数据转换（仅ToTensor，可能需补充数据增强）
    # 转换为Tensor（通道优先，归一化到[0,1]）
    train_dataset = yoloDataset(root=file_root,list_file="voc2012.txt",train=True,transform = [transforms.ToTensor()] )
    # 训练集（启用数据增强，shuffle=True打乱顺序）
    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=4)
    # test_dataset = yoloDataset(root=file_root,list_file='voc07_test.txt',train=False,transform = [transforms.ToTensor()] )
    # 验证集（不启用数据增强，shuffle=False保持顺序）
    test_dataset = yoloDataset(root=file_root,list_file='voc2007test.txt',train=False,transform = [transforms.ToTensor()] )
    test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False,num_workers=4)
    print('the dataset has %d images' % (len(train_dataset)))
    print('the batch_size is %d' % (batch_size))
    logfile = open('log.txt', 'w')  # 保存训练日志（轮次、验证损失等）

    num_iter = 0
    vis = Visualizer(env='xiong') # 初始化可视化工具（如Visdom）
    best_test_loss = np.inf  # 记录最佳验证损失（初始为无穷大）

    # 主训练循环（遍历每个epoch）
    for epoch in range(num_epochs):
        net.train() # 开启训练模式（重要！影响BatchNorm等层）
        # if epoch == 1:
        #     learning_rate = 0.0005
        # if epoch == 2:
        #     learning_rate = 0.00075
        # if epoch == 3:
        #     learning_rate = 0.001

        # 动态调整学习率（第30、40轮降低学习率）
        if epoch == 30:
            learning_rate=0.0001 # 第30轮后学习率降为1e-4
        if epoch == 40:
            learning_rate=0.00001 # 第40轮后学习率降为1e-5
        # optimizer = torch.optim.SGD(net.parameters(),lr=learning_rate*0.1,momentum=0.9,weight_decay=1e-4)
        # 更新优化器的学习率（所有参数组）
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate
         # 打印epoch信息
        print('\n\nStarting epoch %d / %d' % (epoch + 1, num_epochs))
        print('Learning Rate for this epoch: {}'.format(learning_rate))
        
        total_loss = 0.
         # 遍历训练集（每个batch）
        for i,(images,target) in enumerate(train_loader):
            # 转换为Tensor并移动到GPU（若可用）
            images = Variable(images)
            target = Variable(target)
            if use_gpu:
                images,target = images.cuda(),target.cuda()

            # 前向传播：模型预测
            pred = net(images)
            loss = criterion(pred,target)
            print("loss shape:", loss.shape)       # 输出 torch.Size([])（0维）
            print("loss ndim:", loss.ndim)         # 输出 0（0维）
            print("loss:",loss)
            total_loss += loss

             # 反向传播与参数更新
            optimizer.zero_grad()  # 清空梯度
            loss.backward() # 反向传播计算梯度
            optimizer.step() # 优化器更新参数
            # 每5个batch打印一次训练进度
            if (i+1) % 5 == 0:
                print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, average_loss: %.4f' 
                %(epoch+1, num_epochs, i+1, len(train_loader), loss.data, total_loss / (i+1))) # 当前epoch的平均损失
                num_iter += 1
                vis.plot_train_val(loss_train=total_loss/(i+1)) # 实时绘制训练损失曲线

        # ---------------------- 验证阶段 ----------------------
        validation_loss = 0.0  # 累计验证损失
        net.eval() # 开启评估模式（关闭Dropout、BatchNorm的随机更新）
        for i,(images,target) in enumerate(test_loader):
            images = Variable(images,volatile=True)
            target = Variable(target,volatile=True)
            if use_gpu:
                images,target = images.cuda(),target.cuda()
            
            pred = net(images)  # 模型预测
            loss = criterion(pred,target)  # 计算验证损失
            validation_loss += loss.data  # 累计损失
        # 计算平均验证损失    
        validation_loss /= len(test_loader)  # 总损失 / 批次数量
        vis.plot_train_val(loss_val=validation_loss) # 实时绘制验证损失曲线

        # 保存最佳模型（验证损失下降时）
        if best_test_loss > validation_loss:
            best_test_loss = validation_loss
            print('get best test loss %.5f' % best_test_loss)
            torch.save(net.state_dict(),'best.pth')
        logfile.writelines(str(epoch) + '\t' + str(validation_loss) + '\n')  
        logfile.flush()      
        torch.save(net.state_dict(),'yolo.pth')
if __name__ == '__main__':
    import torch.multiprocessing as mp
    mp.freeze_support()  # 添加此句（仅打包时需要）
    train()