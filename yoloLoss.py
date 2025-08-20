#encoding:utf-8
#
#created by xiongzihua 2017.12.26
#
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class yoloLoss(nn.Module):
    def __init__(self,S,B,l_coord,l_noobj):
        """
        YOLO目标检测损失函数
        
        参数：
            S (int): 特征图网格大小(S*S)
            B (int): 每个网格预测的候选框数量(B个)
            l_coord (float): 坐标损失的权重系数
            l_noobj (float): 无目标置信度损失的权重系数
        """
        super(yoloLoss,self).__init__()
        self.S = S  # 网格尺寸（如YOLOv1中S=14）
        self.B = B # 每个网格的候选框数（如YOLOv1中B=2）
        self.l_coord = l_coord # 坐标损失权重（论文中设为5）
        self.l_noobj = l_noobj  # 无目标置信度损失权重（论文中设为0.5）

    def compute_iou(self, box1, box2):
        """
        计算两组边界框的交并比(IoU)
        
        参数：
            box1 (Tensor): 形状[N, 4]，格式为[x1, y1, x2, y2]（绝对坐标）
            box2 (Tensor): 形状[M, 4]，格式同上
            
        返回：
            Tensor: 形状[N, M],每对框的IoU值
        """
        N = box1.size(0) # 第一组框数量
        M = box2.size(0) # 第二组框数量
        
        # 计算交集区域的左上角坐标（lt）和右下角坐标（rb）
        # 扩展维度以广播计算：box1[:,:,:2]与box2[:,:,:2]逐元素比较取最大
        lt = torch.max(
            box1[:,:2].unsqueeze(1).expand(N,M,2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:,:2].unsqueeze(0).expand(N,M,2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        rb = torch.min(
            box1[:,2:].unsqueeze(1).expand(N,M,2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:,2:].unsqueeze(0).expand(N,M,2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        # 计算交集区域的宽高（若为负则置0，即无交集）
        wh = rb - lt  # [N,M,2]
        wh[wh<0] = 0   # 交集宽高不能为负
        inter = wh[:,:,0] * wh[:,:,1]  # 交集面积 [N,M]

        # 计算两组框各自的面积
        area1 = (box1[:,2]-box1[:,0]) * (box1[:,3]-box1[:,1])  # [N,]
        area2 = (box2[:,2]-box2[:,0]) * (box2[:,3]-box2[:,1])  # [M,]
        # 扩展面积维度以广播计算
        area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
        area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]
        # IoU = 交集面积 / (并集面积) = 交集 / (area1 + area2 - 交集)
        iou = inter / (area1 + area2 - inter)
        return iou
    def forward(self,pred_tensor,target_tensor):
        """
        计算YOLO的总损失
        
        参数：
            pred_tensor (Tensor): 预测张量，形状[batch_size, S, S, B*5+20]
                其中B*5为每个候选框的5个参数:[x, y, w, h, c](x,y,w,h归一化(c为置信度)
                20为类别概率(如VOC数据集的20类,one-hot编码)
            target_tensor (Tensor): 真实标签张量,形状与pred_tensor相同
            
        返回：
            Tensor: 总损失（标量）
        """        
        N = pred_tensor.size()[0]  # 批量大小
        # --------------------------
        # 步骤1：区分有目标与无目标的网格
        # --------------------------
        # 有目标的掩码：目标张量中第4列（置信度）>0（表示该网格存在目标）
        coo_mask = target_tensor[:,:,:,4] > 0  # 形状[batch_size,S,S]?
        print(coo_mask.shape)
        # 无目标的掩码：目标张量中第4列==0（表示该网格无目标）
        noo_mask = target_tensor[:,:,:,4] == 0 
        coo_mask = coo_mask.unsqueeze(-1).expand_as(target_tensor)   # [S,S] → [S,S,30]
        print(coo_mask.shape)
        noo_mask = noo_mask.unsqueeze(-1).expand_as(target_tensor)   # [S,S] → [S,S,30]

        # 有目标的真实标签数据：同样展平为[-1, 30]
        # --------------------------
        # 步骤2：提取有目标/无目标的预测与目标数据
        # --------------------------
        # 有目标的预测数据：通过掩码筛选，展平为[-1, 30]
        coo_pred = pred_tensor[coo_mask].view(-1,30)
        box_pred = coo_pred[:,:10].contiguous().view(-1,5) #box[x1,y1,w1,h1,c1]
        class_pred = coo_pred[:,10:]                       #[x2,y2,w2,h2,c2]
        
        coo_target = target_tensor[coo_mask].view(-1,30)
        box_target = coo_target[:,:10].contiguous().view(-1,5)
        class_target = coo_target[:,10:]

        # compute not contain obj loss
        # --------------------------
        # 步骤3：计算无目标置信度损失（nooobj_loss）
        # --------------------------
        # 无目标的预测数据展平为[-1, 30]
        noo_pred = pred_tensor[noo_mask].view(-1,30)
        noo_target = target_tensor[noo_mask].view(-1,30)
          # 无目标的候选框中，仅需关注置信度（第4和第9列，共2个框）
        noo_pred_mask = torch.cuda.ByteTensor(noo_pred.size())
        noo_pred_mask.zero_()
        noo_pred_mask[:,4]=1;noo_pred_mask[:,9]=1
        # 提取无目标候选框的置信度（形状[-1]）
        noo_pred_c = noo_pred[noo_pred_mask] #noo pred只需要计算 c 的损失 size[-1,2]
        noo_target_c = noo_target[noo_pred_mask]
        # 计算置信度的均方误差（MSE）损失（无目标时置信度应接近0）
        nooobj_loss = F.mse_loss(noo_pred_c,noo_target_c,size_average=False)

        # --------------------------
        # 步骤4：计算有目标的置信度与定位损失（contain_loss + loc_loss）
        # --------------------------
        # 初始化响应/非响应掩码（用于区分哪个候选框负责预测目标）
        coo_response_mask = torch.cuda.ByteTensor(box_target.size()) # [2×K,5]
        coo_response_mask.zero_()
        coo_not_response_mask = torch.cuda.ByteTensor(box_target.size())
        coo_not_response_mask.zero_()
        # 初始化目标置信度（用于存储真实IOU值）
        box_target_iou = torch.zeros(box_target.size()).cuda()  # [2×K,5]
        for i in range(0,box_target.size()[0],2): #choose the best iou box
            # 当前网格的2个候选框预测（box_pred[i]和box_pred[i+1]）
            box1 = box_pred[i:i+2] # [2,5]
            # 将预测框的归一化坐标转换为绝对坐标（x,y,w,h → x1,y1,x2,y2）
            box1_xyxy = Variable(torch.FloatTensor(box1.size()))
            box1_xyxy[:,:2] = box1[:,:2]/14. -0.5*box1[:,2:4] # x1 = (x/S) - 0.5w
            box1_xyxy[:,2:4] = box1[:,:2]/14. +0.5*box1[:,2:4]  # x2 = (x/S) + 0.5w
            # 同理处理真实框（假设target_tensor中的坐标是归一化的）
            box2 = box_target[i].view(-1,5) # 真实框参数 [5] → [1,5]
            box2_xyxy = Variable(torch.FloatTensor(box2.size()))
            box2_xyxy[:,:2] = box2[:,:2]/14. -0.5*box2[:,2:4]  # x1
            box2_xyxy[:,2:4] = box2[:,:2]/14. +0.5*box2[:,2:4]  # x2
             # 计算两个候选框与真实框的IoU（形状[2,1]）
            iou = self.compute_iou(box1_xyxy[:,:4],box2_xyxy[:,:4]) #[2,1]
            max_iou,max_index = iou.max(0) # 找到IoU最大的候选框索引（0或1）
            max_index = max_index.data.cuda()   # 转换为CUDA张量

            # 标记响应框（IoU大的候选框）和非响应框（IoU小的候选框）
            coo_response_mask[i+max_index]=1  # 响应框位置置1
            coo_not_response_mask[i+1-max_index]=1 # 非响应框位置置1


            #####
            # we want the confidence score to equal the
            # intersection over union (IOU) between the predicted box
            # and the ground truth
            #####
            # 真实置信度应等于该候选框与真实框的IoU（仅响应框需要学习此值）
            box_target_iou[i+max_index,torch.LongTensor([4]).cuda()] = (max_iou).data.cuda()
        # 转换为可训练的Tensor（若使用PyTorch新版本，可省略Variable）
        box_target_iou = Variable(box_target_iou).cuda()
        #1.response loss
        # --------------------------
        # 子步骤4.1：响应框的置信度损失（contain_loss）
        # --------------------------
        # 提取响应框的预测置信度和真实置信度（IoU值）
        box_pred_response = box_pred[coo_response_mask].view(-1,5) # [K,5]（K为有目标的候选框数）
        box_target_response_iou = box_target_iou[coo_response_mask].view(-1,5)  # [K,5]
        box_target_response = box_target[coo_response_mask].view(-1,5) # [K,5]（真实框参数）

        # 置信度损失：预测置信度 vs 真实IoU（MSE）
        contain_loss = F.mse_loss(box_pred_response[:,4],box_target_response_iou[:,4],size_average=False)
        # --------------------------
        # 子步骤4.2：定位损失（loc_loss）
        # --------------------------
        # 定位损失包含两部分：
        # 1. 中心坐标(x,y)的MSE损失
        # 2. 宽高(w,h)的平方根MSE损失（平衡大框与小框的误差）
        loc_loss = F.mse_loss(box_pred_response[:,:2],box_target_response[:,:2],size_average=False) + F.mse_loss(torch.sqrt(box_pred_response[:,2:4]),torch.sqrt(box_target_response[:,2:4]),size_average=False)
        #2.not response loss

        box_pred_not_response = box_pred[coo_not_response_mask].view(-1,5)  # [K',5]（K'为非响应框数）
        box_target_not_response = box_target[coo_not_response_mask].view(-1,5)  # [K',5]
        # 非响应框的真实置信度强制设为0
        box_target_not_response[:,4]= 0
        #not_contain_loss = F.mse_loss(box_pred_response[:,4],box_target_response[:,4],size_average=False)
        

        not_contain_loss = F.mse_loss(box_pred_not_response[:,4], box_target_not_response[:,4],size_average=False)
        #I believe this bug is simply a typo
        # --------------------------
        # 步骤6：计算类别损失（class_loss）
        # --------------------------
        # 类别损失：预测类别概率 vs 真实类别概率（one-hot编码，MSE）
        #3.class loss
        class_loss = F.mse_loss(class_pred,class_target,size_average=False)

        # --------------------------
        # 步骤7：总损失（加权求和后取平均）
        # --------------------------
        return (self.l_coord*loc_loss + 2*contain_loss + not_contain_loss + self.l_noobj*nooobj_loss + class_loss)/N




