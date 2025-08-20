#encoding:utf-8
#
#created by xiongzihua
#
'''
txt描述文件 image_name.jpg x y w h c x y w h c 这样就是说一张图片中有两个目标
'''
import os
import sys
import os.path

import random
import numpy as np

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import cv2
import matplotlib.pyplot as plt

class yoloDataset(data.Dataset):
    """自定义Yolo目标检测数据集,继承自pytorch的Dataset"""
    image_size = 448 # 输入·模型的图像尺寸（正方形）
    def __init__(self,root,list_file,train,transform):
        """
        初始化数据集
        :param root: 图像文件存储的根目录
        :param list_file: 标注列表文件路径（每行格式：图片名 标注1 标注2 ...)
        :param train: 是否为训练模式（决定是否应用数据增强）
        :param transform: 自定义的图像转换操作(如ToTensor)
        """
        print('data init')
        self.root=root # 图像的根目录
        self.train = train # 训练模式的标志位
        self.list_file = list_file
        self.transform=transform # 外部转换操作
        self.fnames = []  # 存储所有图像文件名（如['000001.jpg', '000002.jpg', ...]）
        self.boxes = []   # 存储每张图的所有边界框（格式：[x1,y1,x2,y2]的Tensor列表）
        self.labels = []  # 存储每张图的所有类别标签（LongTensor列表）
        self.mean = (123,117,104) # BGR三通道均值（用于归一化）

        # # 处理多列表文件的情况（如合并VOC07和VOC12的验证集）
        # if isinstance(list_file, list):
        #     # Cat multiple list files together.
        #     # This is especially useful for voc07/voc12 combination.
        #     tmp_file ='listfile.txt'
        #     os.system('cat %s > %s' % (' '.join(list_file), tmp_file))
        #     list_file = tmp_file # # 将list_file指向合并后的临时文件
        
        with open(list_file) as f:
            lines  = f.readlines() # 按行读取

        for line in lines:
            splited = line.strip().split() # 按空格分割每行内容
            self.fnames.append(splited[0])  # 提取图像文件名（如'000001.jpg'）
             # 计算边界框数量：总元素数-1（文件名）后除以5（每个框5个值）
            num_boxes = (len(splited) - 1) // 5
            box=[]   # 存储当前图的所有边界框（格式：[x1,y1,x2,y2]）
            label=[] # 存储当前图的所有类别标签（原始索引+1，因YOLO通常保留0为背景）
            for i in range(num_boxes):
                x = float(splited[1+5*i]) # 中心点x（归一化）
                y = float(splited[2+5*i]) # 中心点y（归一化）
                x2 = float(splited[3+5*i]) # 右下角x（归一化）
                y2 = float(splited[4+5*i]) # 右下角y（归一化）
                c = splited[5+5*i] # 类别索引（从0开始） //坐标并没有归一化
                # 转换为绝对坐标（后续归一化到图像宽高）
                # 注意：原代码此处可能存在问题！原标注的x,y,w,h如果是归一化的，
                # 则x2=x+w，y2=y+h也应是归一化的。但直接存储[x,y,x2,y2]作为绝对坐标是错误的。
                # 正确做法应为：x_abs = x * img_w; y_abs = y * img_h; w_abs = w * img_w; h_abs = h * img_h
                # 然后x1 = x_abs - w_abs/2; y1 = y_abs - h_abs/2; x2 = x_abs + w_abs/2; y2 = y_abs + h_abs/2
                # 原代码可能假设输入的x,y,w,h是绝对坐标的左上和右下？需要根据实际标注文件确认。
                box.append([x, y, x2, y2])  # 这里可能隐含假设输入是绝对坐标的左上和右下？
                label.append(int(c)+1)  # YOLO通常将类别索引+1（保留0给背景）
            self.boxes.append(torch.Tensor(box))
            self.labels.append(torch.LongTensor(label))
        self.num_samples = len(self.boxes)

        """
        获取单个样本（图像和对应的目标张量）
        :param idx: 样本索引
        :return: 图像(Tensor)和目标张量(7x7x30)
        """
    def __getitem__(self,idx):
        fname = self.fnames[idx]  # 当前图像文件名
        # 读取图像（注意：root可能需要以/结尾，否则拼接路径可能出错，建议改为os.path.join(root, fname)）
        img = cv2.imread(os.path.join(self.root+fname)) 
        boxes = self.boxes[idx].clone()  # 当前图的边界框（深拷贝避免修改原数据）
        labels = self.labels[idx].clone()  # 当前图的类别标签（深拷贝）

        if self.train:
            #img = self.random_bright(img)
            img, boxes = self.random_flip(img, boxes) # 随机水平翻转
            img,boxes = self.randomScale(img,boxes) # 随机缩放
            img = self.randomBlur(img) # 随机模糊
            img = self.RandomBrightness(img) # 随机亮度调整（另一种实现）
            img = self.RandomHue(img) # 随机色调调整
            img = self.RandomSaturation(img) # 随机饱和度调整
            img,boxes,labels = self.randomShift(img,boxes,labels) # 随机平移
            img,boxes,labels = self.randomCrop(img,boxes,labels) # 随机裁剪
        # #debug
        # box_show = boxes.numpy().reshape(-1)
        # print(box_show)
        # img_show = self.BGR2RGB(img)
        # pt1=(int(box_show[0]),int(box_show[1])); pt2=(int(box_show[2]),int(box_show[3]))
        # cv2.rectangle(img_show,pt1=pt1,pt2=pt2,color=(0,255,0),thickness=1)
        # plt.figure()
        
        # # cv2.rectangle(img,pt1=(10,10),pt2=(100,100),color=(0,255,0),thickness=1)
        # plt.imshow(img_show)
        # plt.show()
        # #debug


        # 获取图像尺寸（h:高度，w:宽度）
        h,w,_ = img.shape
        # 归一化边界框坐标（将绝对坐标转为相对于图像宽高的比例）
        # 例如：原坐标是绝对像素值，除以w得到x方向的归一化值，除以h得到y方向的归一化值
        boxes /= torch.Tensor([w,h,w,h]).expand_as(boxes)  # shape: (num_boxes, 4)
        """
        假设图像宽度 w=800,高度 h=600,有一个边界框的绝对坐标为 (100, 150, 300, 450)（即 x_min=100, y_min=150, x_max=300, y_max=450。
        归一化过程如下：

        x_min/w = 100/800 = 0.125
        y_min/h = 150/600 = 0.25
        x_max/w = 300/800 = 0.375
        y_max/h = 450/600 = 0.75
        归一化后的坐标为 (0.125, 0.25, 0.375, 0.75)，所有值均在 [0, 1] 范围内。
        """
         # BGR转RGB（PyTorch预训练模型通常使用RGB输入）
        img = self.BGR2RGB(img) #because pytorch pretrained model use RGB
        img = self.subMean(img,self.mean) # 减去均值（归一化）
         # 调整图像尺寸到模型输入大小（448x448）
        img = cv2.resize(img,(self.image_size,self.image_size))
        # 将边界框和标签编码为YOLO目标张量（7x7x30）
        target = self.encoder(boxes,labels)# 7x7x30
         # 应用用户指定的转换操作（如ToTensor）
        for t in self.transform:
            img = t(img)

        return img,target
    def __len__(self):
        return self.num_samples

    def encoder(self,boxes,labels):
        """
        将边界框和标签编码为YOLO的目标张量(7x7x30)
        YOLOv1的输出特征图是7x7(每个网格预测1个边界框),每个网格包含：
        - 边界框坐标(x,y,w,h)4个值
        - 置信度(objectness score);1个值
        - 类别概率(假设20类):20个值
        总计:7x7x(4+1+20)=7x7x30
        :param boxes: 边界框张量(shape: (num_boxes, 4)，格式：[x1,y1,x2,y2],归一化到0-1)
        :param labels: 类别标签张量(shape: (num_boxes,),类别索引从1开始)
        :return: 目标张量(shape: (7,7,30))
        """
        grid_num = 14 # 注意：这里可能有误！7x7网格的话grid_num应为7，14可能是笔误？
        target = torch.zeros((grid_num,grid_num,30))  # 初始化目标张量（全0）
        cell_size = 1./grid_num # 每个网格的宽度和高度（归一化到特征图）

        # 计算边界框的中心点坐标（cx,cy）和宽高（w,h）
        wh = boxes[:,2:]-boxes[:,:2]
        cxcy = (boxes[:,2:]+boxes[:,:2])/2
        for i in range(cxcy.size()[0]):
            cxcy_sample = cxcy[i] # 当前框的中心点坐标（归一化到原图）
            # 计算该中心点落在特征图的哪个网格中（ij是网格的行列索引，从0开始）
            # 公式：网格行号 = floor(cxcy_sample[1] / cell_size)，列号 = floor(cxcy_sample[0] / cell_size)
            # 原代码中使用ceil()-1，效果与floor相同（例如：0.3/0.25=1.2 → ceil(1.2)=2 → 2-1=1，即第1行）
            ij = (cxcy_sample/cell_size).ceil()-1 # ij形状：(2,)（i是行，j是列）

            # 设置网格中该框的置信度为1（表示该网格包含目标）
            target[int(ij[1]),int(ij[0]),4] = 1 # 置信度（第一个框）
            target[int(ij[1]),int(ij[0]),9] = 1 # 置信度（第二个框，若存在）
            # 设置类别概率（假设类别索引从1开始，对应20类时索引1-20）
            # 目标张量中类别概率的位置是[9+类别索引]（例如类别1对应位置10）
            target[int(ij[1]),int(ij[0]),int(labels[i])+9] = 1

            # 计算边界框相对于所在网格的偏移量（delta_xy）
            # 网格左上角的绝对坐标（归一化到原图）：xy = ij * cell_size
            xy = ij*cell_size #匹配到的网格的左上角相对坐标,# xy形状：(2,)
            delta_xy = (cxcy_sample -xy)/cell_size  # 偏移量（归一化到0-1)
            # 设置边界框的宽高（wh）和偏移量（delta_xy）到目标张量
            target[int(ij[1]),int(ij[0]),2:4] = wh[i] # 第一个框的宽高
            target[int(ij[1]),int(ij[0]),:2] = delta_xy # 第一个框的偏移量
            target[int(ij[1]),int(ij[0]),7:9] = wh[i]  # 第二个框的宽高（若存在）
            target[int(ij[1]),int(ij[0]),5:7] = delta_xy # 第二个框的偏移量（若存在）
        return target
    def BGR2RGB(self,img):
        return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    def BGR2HSV(self,img):
        return cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    def HSV2BGR(self,img):
        return cv2.cvtColor(img,cv2.COLOR_HSV2BGR)
    
    def RandomBrightness(self,bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h,s,v = cv2.split(hsv)
            adjust = random.choice([0.5,1.5])
            v = v*adjust
            v = np.clip(v, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h,s,v))
            bgr = self.HSV2BGR(hsv)
        return bgr
    def RandomSaturation(self,bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h,s,v = cv2.split(hsv)
            adjust = random.choice([0.5,1.5])
            s = s*adjust
            s = np.clip(s, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h,s,v))
            bgr = self.HSV2BGR(hsv)
        return bgr
    def RandomHue(self,bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h,s,v = cv2.split(hsv)
            adjust = random.choice([0.5,1.5])
            h = h*adjust
            h = np.clip(h, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h,s,v))
            bgr = self.HSV2BGR(hsv)
        return bgr

    def randomBlur(self,bgr):
        if random.random()<0.5:
            bgr = cv2.blur(bgr,(5,5))
        return bgr

    def randomShift(self,bgr,boxes,labels):
        #平移变换
        center = (boxes[:,2:]+boxes[:,:2])/2
        if random.random() <0.5:
            height,width,c = bgr.shape
            after_shfit_image = np.zeros((height,width,c),dtype=bgr.dtype)
            after_shfit_image[:,:,:] = (104,117,123) #bgr
            shift_x = random.uniform(-width*0.2,width*0.2)
            shift_y = random.uniform(-height*0.2,height*0.2)
            #print(bgr.shape,shift_x,shift_y)
            #原图像的平移
            if shift_x>=0 and shift_y>=0:
                after_shfit_image[int(shift_y):,int(shift_x):,:] = bgr[:height-int(shift_y),:width-int(shift_x),:]
            elif shift_x>=0 and shift_y<0:
                after_shfit_image[:height+int(shift_y),int(shift_x):,:] = bgr[-int(shift_y):,:width-int(shift_x),:]
            elif shift_x <0 and shift_y >=0:
                after_shfit_image[int(shift_y):,:width+int(shift_x),:] = bgr[:height-int(shift_y),-int(shift_x):,:]
            elif shift_x<0 and shift_y<0:
                after_shfit_image[:height+int(shift_y),:width+int(shift_x),:] = bgr[-int(shift_y):,-int(shift_x):,:]

            shift_xy = torch.FloatTensor([[int(shift_x),int(shift_y)]]).expand_as(center)
            center = center + shift_xy
            mask1 = (center[:,0] >0) & (center[:,0] < width)
            mask2 = (center[:,1] >0) & (center[:,1] < height)
            mask = (mask1 & mask2).view(-1,1)
            boxes_in = boxes[mask.expand_as(boxes)].view(-1,4)
            if len(boxes_in) == 0:
                return bgr,boxes,labels
            box_shift = torch.FloatTensor([[int(shift_x),int(shift_y),int(shift_x),int(shift_y)]]).expand_as(boxes_in)
            boxes_in = boxes_in+box_shift
            labels_in = labels[mask.view(-1)]
            return after_shfit_image,boxes_in,labels_in
        return bgr,boxes,labels

    def randomScale(self,bgr,boxes):
        #固定住高度，以0.8-1.2伸缩宽度，做图像形变
        if random.random() < 0.5:
            scale = random.uniform(0.8,1.2)
            height,width,c = bgr.shape
            bgr = cv2.resize(bgr,(int(width*scale),height))
            scale_tensor = torch.FloatTensor([[scale,1,scale,1]]).expand_as(boxes)
            boxes = boxes * scale_tensor
            return bgr,boxes
        return bgr,boxes

    def randomCrop(self,bgr,boxes,labels):
        if random.random() < 0.5:
            center = (boxes[:,2:]+boxes[:,:2])/2
            height,width,c = bgr.shape
            h = random.uniform(0.6*height,height)
            w = random.uniform(0.6*width,width)
            x = random.uniform(0,width-w)
            y = random.uniform(0,height-h)
            x,y,h,w = int(x),int(y),int(h),int(w)

            center = center - torch.FloatTensor([[x,y]]).expand_as(center)
            mask1 = (center[:,0]>0) & (center[:,0]<w)
            mask2 = (center[:,1]>0) & (center[:,1]<h)
            mask = (mask1 & mask2).view(-1,1)

            boxes_in = boxes[mask.expand_as(boxes)].view(-1,4)
            if(len(boxes_in)==0):
                return bgr,boxes,labels
            box_shift = torch.FloatTensor([[x,y,x,y]]).expand_as(boxes_in)

            boxes_in = boxes_in - box_shift
            boxes_in[:,0]=boxes_in[:,0].clamp_(min=0,max=w)
            boxes_in[:,2]=boxes_in[:,2].clamp_(min=0,max=w)
            boxes_in[:,1]=boxes_in[:,1].clamp_(min=0,max=h)
            boxes_in[:,3]=boxes_in[:,3].clamp_(min=0,max=h)

            labels_in = labels[mask.view(-1)]
            img_croped = bgr[y:y+h,x:x+w,:]
            return img_croped,boxes_in,labels_in
        return bgr,boxes,labels




    def subMean(self,bgr,mean):
        mean = np.array(mean, dtype=np.float32)
        bgr = bgr - mean
        return bgr

    def random_flip(self, im, boxes):
        if random.random() < 0.5:
            im_lr = np.fliplr(im).copy()
            h,w,_ = im.shape
            xmin = w - boxes[:,2]
            xmax = w - boxes[:,0]
            boxes[:,0] = xmin
            boxes[:,2] = xmax
            return im_lr, boxes
        return im, boxes
    def random_bright(self, im, delta=16):
        alpha = random.random()
        if alpha > 0.3:
            im = im * alpha + random.randrange(-delta,delta)
            im = im.clip(min=0,max=255).astype(np.uint8)
        return im

def main():
    from torch.utils.data import DataLoader
    import torchvision.transforms as transforms
    file_root = './VOCdevkit/VOC2012/JPEGImages/'
    train_dataset = yoloDataset(root=file_root,list_file='voc2012.txt',train=True,transform = [transforms.ToTensor()] )
    train_loader = DataLoader(train_dataset,batch_size=1,shuffle=False,num_workers=0)
    train_iter = iter(train_loader)
    for i in range(100):
        img,target = next(train_iter)
        print(img,target)


if __name__ == '__main__':
    main()


