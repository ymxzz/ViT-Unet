import torch
from glob import glob
import nibabel as nib
from monai.transforms import Resize,LoadImage,NormalizeIntensity
from torch.utils.data import Dataset, DataLoader
from nibabel.imageglobals import LoggingOutputSuppressor
import logging
logging.disable(logging.WARNING)


resize = Resize(spatial_size=(96,96,32))
class dataset(Dataset):

    def __init__(self,img_paths,mask_paths):

        self.imgs = img_paths  #原图路径
        self.masks = mask_paths #mask路径

    def __getitem__(self,index):

        img = self.imgs[index]
        mask = self.masks[index]
        
        # 读取图片转成tensor
        #img_tensor,_ = LoadImage()(img)
        #mask_tensor,_ = LoadImage()(mask)
        
       
        img_tensor= torch.from_numpy(nib.load(img).get_fdata()).type(torch.float32)
        mask_tensor= torch.from_numpy(nib.load(mask).get_fdata())
        
        #img_tensor = NormalizeIntensity()(img_tensor)  
        #mask_tensor = NormalizeIntensity()(mask_tensor) 
        # 维度和尺寸转化
        img_tensor=resize(img_tensor.unsqueeze(0)).permute(0,3,1,2) # [c,d,h,w]
        # [c,d,h,w]
        mask_tensor=resize(mask_tensor.unsqueeze(0)).permute(0,3,1,2).type(torch.long)
        mask_tensor[mask_tensor > 0] = 1 # 图片里像素值大于0的都转为1
        mask_tensor[mask_tensor < 0] = 0
        
        return img_tensor,mask_tensor

    def __len__(self):

        return len(self.imgs) 
    
if __name__ == "__main__":
    imgList = glob(r".\img\*.nii.gz")
    maskList = glob(r".\mask\*.nii.gz")
    train_dataset = dataset(imgList,maskList)
    x=train_dataset.__getitem__(0)[0]
    print(x.shape)
