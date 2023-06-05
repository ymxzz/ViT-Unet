import torch
import torch.nn as nn

from Unet3d_vit import Unet3d

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from glob import glob
from torch import optim
from myDataset import dataset
from myLossFunc import DiceLoss





if __name__ == "__main__":
    Epoch = 14
    random_seed = 3407 # 233
    torch.manual_seed(random_seed) # 设定随机数种子,固定每次开始程序训练的初始权重
    
    #imgList = glob(r"D:\lits_train\scan\*.nii")
    #maskList = glob(r"D:\lits_train\label\*.nii")
    imgList = glob(r".\img\*.nii.gz")
    maskList = glob(r".\mask\*.nii.gz")
    # 412,28
    dataset_train = dataset(imgList[:412],maskList[:412])
    dataloader_train = DataLoader(dataset=dataset_train,
                                  batch_size=2,
                                  shuffle=True)
    
    dataset_val = dataset(imgList[412:463],maskList[412:463])
    print(len(dataset_train),len(dataset_val))
    dataloader_val = DataLoader(dataset=dataset_val,
                                  batch_size=2,
                                  shuffle=True)
    
    model = Unet3d()
    #model = Unet()
    #model = SwinUNETR(img_size=(32, 96, 96),in_channels=1,out_channels=2,feature_size=24)
    #model = UNETR()
    model.load_state_dict(torch.load("./Data/net.pth"))  # 1e-5 # 1e-4,1e-5
    optimizer = optim.Adam(params=model.parameters(),lr=1e-4)
    #exp_lr_scheduler = lr.StepLR(optimizer, step_size=50, gamma=0.25)
    #loss_func = nn.CrossEntropyLoss()
    loss_func = DiceLoss()
    #loss_func = DiceCELoss()


    IoU_list = []
    loss_list = []

    val_IoU_list = []
    val_loss_list = []
    
    Dice_train_list = []
    Dice_val_list = []
    '''
    # 读取训练数据并转为list
    IoU_list = torch.load("./Data/IoU_list.pt").tolist()
    loss_list = torch.load("./Data/loss_list.pt").tolist()

    val_IoU_list = torch.load("./Data/val_IoU_list.pt").tolist()
    val_loss_list = torch.load("./Data/val_loss_list.pt").tolist()
    
    Dice_train_list = torch.load("./Data/Dice_train_list.pt").tolist()
    Dice_val_list = torch.load("./Data/Dice_val_list.pt").tolist()
    '''

    
    #torch.backends.cudnn.enable = True

    model.cuda() #模型放到GPU
    # 开始训练
    for epoch in range(Epoch):

        train_loss = 0.0
        train_IoU = 0.0

        val_loss = 0.0
        val_IoU = 0.0
        
        TP_ = 0.
        FP_ = 0.
        FN_ = 0.
        TN_ = 0.
        
        TP = 0.
        FP = 0.
        FN = 0.
        TN = 0.
        
        
        print(f"epoch {epoch+1}:")
        # 训练模式
        model.train()
        for x,y in dataloader_train:
            x,y = x.to("cuda"), y.to("cuda") # 数据需要放到GPU上进行计算
            
            y_pred = model(x)
            
            # 计算损失,实际输出与标准输出之间的混乱程度
            # y_pred:[batch_size, 2, 32,160,160] dtype=torch.float32
            # y:     [batch_size, 1,32,160,160] dtype=torch.long
            loss = loss_func(y_pred,y)
            # backward
            optimizer.zero_grad() # 梯度清零
            loss.backward() # 损失反向传播
            optimizer.step() # 保留梯度
            # 统计训练数据
            with torch.no_grad():
                y_pred = torch.argmax(y_pred,dim=1) #(1, 32, 160, 160)
                y= y.squeeze(1)
                
                TP_ += (y_pred*y).sum().item()
                FP_ += (y_pred*(1-y)).sum().item()
                FN_ += ((1-y_pred)*y).sum().item()
                TN_ += ((1-y_pred)*(1-y)).sum().item()

                #train_acc += ((y_pred == y).sum().item())/(160*160*32) # 本次epoch 单个训练正确率
                
                train_loss += loss.item() # 本次epoch 单个样本损失
        # 验证模式
        model.eval()
        with torch.no_grad():
            for x_val, y_val in dataloader_val:
                x_val, y_val = x_val.to("cuda"), y_val.to("cuda")
                y_val_pred = model(x_val)
                loss_ = loss_func(y_val_pred,y_val)
                # 统计验证数据
                y_val_pred = torch.argmax(y_val_pred,dim=1)
                y_val=y_val.squeeze(1)
                # val_acc += ((y_val_pred == y_val).sum().item())/(160*160*32) # 本次epoch 单个训练正确率
                val_loss += loss_.item() # 本次epoch 单个样本损失 
                # TP FP FN TN
                TP += (y_val_pred*y_val).sum().item()
                FP += (y_val_pred*(1-y_val)).sum().item()
                FN += ((1-y_val_pred)*y_val).sum().item()
                TN += ((1-y_val_pred)*(1-y_val)).sum().item()
        
        train_IoU = TP_/(TP_+FN_+FP_)
        val_IoU = TP/(TP+FN+FP)
        Dice_train = 2*TP_/(2*TP_+FN_+FP_)
        Dice_val = 2*TP/(2*TP+FN+FP)    
        # 统计后存放在列表里
        
        Dice_train_list.append(Dice_train)
        Dice_val_list.append(Dice_val)
        
        IoU_list.append(train_IoU)
        loss_list.append(train_loss/len(dataloader_train.dataset))
        
        val_IoU_list.append(val_IoU)
        val_loss_list.append(val_loss/len(dataloader_val.dataset)) 
        
        print(f"TP TP_ FP FP_{TP,TP_,FP,FP_}")
        print(f"train_loss:{train_loss/len(dataloader_train.dataset):.3f},\
            val_loss:{val_loss/len(dataloader_val.dataset):.4f},\
            train_IoU:{train_IoU:.4f},\
            val_IoU:{val_IoU:.4f}")
        
        '''print(f"TP:{TP}\
                FP:{FP}\
                FN:{FN}\
                TN:{TN}\
                IoU:{TP/(TP+FN+FP):.4f}")'''
        print(f"Dice_train:{Dice_train:.4f}\
                Dice_val:{Dice_val:.4f}")
        torch.save(model.state_dict(),f"./Data/net{str(epoch+1)}.pth") #保存模型数据
        #torch.cuda.empty_cache()
    # 保存下训练数据&模型，下次接着训练   
    torch.save(model.state_dict(),"./Data/net.pth") #保存模型数据
    # iou
    torch.save(torch.tensor(Dice_train_list),"./Data/Dice_train_list.pt") 
    torch.save(torch.tensor(Dice_val_list),"./Data/Dice_val_list.pt")
    # acc
    torch.save(torch.tensor(IoU_list),"./Data/IoU_list.pt") 
    torch.save(torch.tensor(val_IoU_list),"./Data/val_IoU_list.pt")
    # loss
    torch.save(torch.tensor(loss_list),"./Data/loss_list.pt") 
    torch.save(torch.tensor(val_loss_list),"./Data/val_loss_list.pt")
    
    # 绘制训练数据
    x_plot = range(1,int(len(IoU_list))+1)
    plt.figure("训练数据和验证数据")
    
    plt.subplot(1,2,1)
    plt.plot(x_plot,IoU_list,color='red',label='train_IoU')
    plt.plot(x_plot,val_IoU_list,color='green',label='val_IoU')
    plt.plot(x_plot,loss_list,color='blue',label='train_loss')
    plt.plot(x_plot,val_loss_list,color='pink',label='val_loss')
    plt.xlabel("epoch")
    plt.legend(loc='center right') # 标注的位置
    
    plt.subplot(1,2,2)
    plt.plot(x_plot,Dice_train_list,color='red',label='Dice_train')
    plt.plot(x_plot,Dice_val_list,color='green',label='Dice_val')
    plt.plot(x_plot,loss_list,color='blue',label='train_loss')
    plt.plot(x_plot,val_loss_list,color='pink',label='val_loss')

    plt.xlabel("epoch")
    plt.legend(loc='center right') # 标注的位置
    plt.show()
    

    