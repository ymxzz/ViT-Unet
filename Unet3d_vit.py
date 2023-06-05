import torch
from torch import nn
from torch.nn import functional as F
from torch import nn, einsum
from einops import rearrange
import copy


'''Embed'''
class Embedding(nn.Module):
    def __init__(self, input_dim, embed_dim, cube_size, patch_size, dropout):
        super().__init__()
        # 32768
        self.n_patches = (cube_size[0]//patch_size) * (cube_size[1]//patch_size) * (cube_size[2]//patch_size)
        self.patch_size = patch_size # 4
        self.embed_dim = embed_dim # 32      
        self.patch_embeddings = nn.Conv3d(in_channels=input_dim, out_channels=embed_dim,
                                          kernel_size=patch_size, stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.n_patches, embed_dim)) # (1,8192, 32)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x): # torch.Size([1, 4, 128, 128, 128])
        #_,_,d,h,w = x.shape
        #d,h,w = int(d/self.patch_size),int(h/self.patch_size),int(w/self.patch_size)
        #Conv3d(4, 768, kernel_size=(16, 16, 16), stride=(16, 16, 16))
        x = self.patch_embeddings(x) # torch.Size([1, 768, 8, 8, 8])
        x = x.flatten(2).transpose(-1, -2) # torch.Size([1, 768, 512])
        #x = x.transpose(-1, -2) # torch.Size([1, 512, 768])
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        #embeddings = embeddings.transpose(-1,-2).view(-1,self.embed_dim,d,h,w)
        return embeddings
    
'''MLP'''
class MLP(nn.Module):
    def __init__(self, dim, dim_=2048, dropout=0.1):
        super().__init__()
        self.fc_1 = nn.Linear(dim, dim_)
        self.fc_2 = nn.Linear(dim_, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.fc_2(self.dropout(F.gelu(self.fc_1(x)))) 
    
class Transformer_encoder_block(nn.Module):
    def __init__(self, head_nums,in_channel,drop_p = 0.1):
        super().__init__()
        self.head_num = head_nums
        self.embed_dim = in_channel
        self.scale = (in_channel/head_nums) ** (-0.5)
        self.soft_max = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(in_channel,3*in_channel)

        self.norm = nn.LayerNorm(self.embed_dim)
        self.mlp = MLP(in_channel)
        self.dropout1 = nn.Dropout(drop_p)
        self.dropout2 = nn.Dropout(drop_p)
        
        
    def forward(self,x): # torch.Size([1, 64, 16, 64, 64])
        #print(x.shape)
        #b, c, d,h,w= x.shape
        # [1, 128, 8, 32, 32]
        #x1 = x.flatten(2).transpose(-1,-2)
        out = self.norm(x) 
        # [1, 128, 8*32*32]
        QKV = self.to_qkv(out).chunk(3,dim=-1)  
        # map(func,iterable)
        Q, K, V = map(lambda t: rearrange(t,"b n (heads head_dim) -> b heads n head_dim", heads=self.head_num), QKV)
        QK = einsum("b h i d, b h j d -> b h i j", Q, K)*self.scale 
        QK_ = self.dropout1(self.soft_max(QK))
        att = einsum("b h i j, b h j d -> b h i d", QK_, V) 
        out = rearrange(att, "b heads n head_dim -> b n (heads head_dim)")
        
        x2 = x+out
        out = self.norm(x2)
        out = self.dropout2(self.mlp(out))
        out = x2 + out
        
        return out
    
class Down_layer(nn.Module):
    def __init__(self, in_channels,
                 out_channels) -> None:
        super().__init__()
        
        self.double_conv = nn.Sequential(nn.Conv3d(in_channels = in_channels,
                                            out_channels = in_channels,
                                            kernel_size = (3,3,3),
                                            stride = (1,1,1),
                                            padding = 1
                                            ),
                                  nn.BatchNorm3d(in_channels),
                                  nn.GELU(),
                                  
                                  nn.Conv3d(in_channels = in_channels,
                                            out_channels = out_channels,
                                            kernel_size = (3,3,3),
                                            stride = (1,1,1),
                                            padding = 1
                                            ),
                                  nn.BatchNorm3d(out_channels),
                                  nn.GELU())
        
        self.max_pool = nn.MaxPool3d(2)
        
    def forward(self,x):
        
        x= self.max_pool(x)
        
        x = self.double_conv(x)
        
        return x
    
class Up_layer(nn.Module):
    def __init__(self, in_channels,
                 out_channels) -> None:
        super().__init__()
        
        self.double_conv = nn.Sequential(nn.Conv3d(in_channels = in_channels,
                                            out_channels = out_channels,
                                            kernel_size = (3,3,3),
                                            stride = (1,1,1),
                                            padding = 1
                                            ),
                                  nn.BatchNorm3d(out_channels),
                                  nn.GELU(),
                                  
                                  nn.Conv3d(in_channels = out_channels,
                                            out_channels = out_channels,
                                            kernel_size = (3,3,3),
                                            stride = (1,1,1),
                                            padding = 1
                                            ),
                                  nn.BatchNorm3d(out_channels),
                                  nn.GELU())
            
    def forward(self,x):

        x = self.double_conv(x)
        
        return x
    
# 反卷积做上采样,恢复分辨率
class Up_conv(nn.Module):
    def __init__(self,in_channels) -> None:
        super().__init__()
        
        self.up_conv = nn.Sequential(nn.ConvTranspose3d(in_channels = in_channels,
                                          out_channels = in_channels,
                                          kernel_size=(3,3,3),
                                          stride=(2,2,2),
                                          padding=1,
                                          output_padding=1
                                          ),
                                     nn.BatchNorm3d(in_channels),
                                     nn.GELU() 
        )   
    def forward(self,x):
        
        x = self.up_conv(x)
        
        return x
    
class Unet3d(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.down1 = nn.Sequential(nn.Conv3d(in_channels = 1,
                                            out_channels = 32,
                                            kernel_size = (3,3,3),
                                            stride = (1,1,1),
                                            padding = 1
                                            ),
                                  nn.BatchNorm3d(32),
                                  nn.GELU(),
                                  
                                  nn.Conv3d(in_channels = 32,
                                            out_channels = 64,
                                            kernel_size = (3,3,3),
                                            stride = (1,1,1),
                                            padding = 1
                                            ),
                                  nn.BatchNorm3d(64),
                                  nn.GELU(),
                                  )
        
        
        self.down2 = Down_layer(64,128)
        
        self.down3 = Down_layer(128,256)
        
        self.bottom = Down_layer(256,512)
        
        self.up_conv4 = Up_conv(768)
        self.up4 = Up_layer(768+512,512)
        
        self.up_conv3 = Up_conv(512)
        self.up3 = Up_layer(512+256,256)
        
        self.up_conv2 = Up_conv(256)
        self.up2 = Up_layer(256+128,128)
        
        self.up_conv1 = Up_conv(128)
        self.up1 = Up_layer(128+64,64)
        
        self.out = nn.Conv3d(64,2,kernel_size=1)
    
        
        # ViT
        self.embed = Embedding(1,768,(32,96,96),16,0.1)
        self.vit = nn.ModuleList()
        for _ in range(6):
            layer = Transformer_encoder_block(8,768)
            self.vit.append(copy.deepcopy(layer))
        self.vit_conv = nn.Conv3d(768,768,kernel_size=3,stride=1,padding=1)
            
        
        
       
    def forward(self,x): # 传入 [b,1,32,96,96] [b,c,d,h,w]
        
        # [b,1,32,96,96] -> [b,32,32,96,96] -> [b,64,32,96,96]
        x1 = self.down1(x) 
        
        # [b,64,32,96,96] -> [b,64,16,48,48] -> [b,64,16,48,48] -> [b,128,16,48,48]
        x2 = self.down2(x1) 
        
        # [b,128,16,48,48] -> [b,128,8,24,24] -> [b,128,8,24,24]-> [b,256,8,24,24]
        x3 = self.down3(x2) 
        
        # [b,256,8,8,24,24] -> [b,256,4,12,12] -> [b,256,4,12,12] -> [b,512,4,12,12]->[b,576,512,]
        #xx = F.max_pool3d(self.bottom(x3),2).flatten(2).transpose(-1,-2)
        xx = self.bottom(x3)
        
        z = self.embed(x)
        #z = self.bottom_layer1(z)
        for block in self.vit:
            z = block(z)
             
        z = z.transpose(-1,-2).view(-1,768,2,6,6)
        z = self.vit_conv(z)
        
        z = self.up_conv4(z) # [1, 768, 4, 12, 12]
        
        #xx = self.up_conv4(xx.transpose(-1,-2).view(-1,512,2,6,6))
        
        xx = self.up4(torch.cat([xx,z],dim =1))
        
        # [b,512,4,12,12] -> [b,512,8,32,32]
        xx3 = self.up_conv3(xx) 
        
        # [b,256+512,8,32,32] -> [b,256,8,32,32] -> [b,256,8,32,32]
        xx3 =self.up3(torch.cat([x3,xx3],dim=1)) 
        
        # [b,256,8,32,32] -> [b,256,16,64,64]
        xx2 = self.up_conv2(xx3) 
        
        # [b,128+256,16,64,64] -> [b,128,16,64,64] -> [b,128,16,64,64]
        xx2 =self.up2(torch.cat([x2,xx2],dim=1)) 
        
        # [b,128,16,64,64] -> [b,128,32,128,128]
        xx1 = self.up_conv1(xx2)
        
        # [b,64+128,32,128,128] -> [b,64,32,128,128] -> [b,64,32,128,128]
        xx1 =self.up1(torch.cat([x1,xx1],dim=1)) 
        
        # [b,64,32,128,128] -> [b,2,32,128,128]
        out = self.out(xx1)       
        
        return  out
    
       
if __name__ == "__main__":
    
    x = torch.ones(1,1,32,96,96)
    module = Unet3d()
    y = module.forward(x) 
    print(y.shape)