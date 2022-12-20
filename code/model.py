import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Util(nn.Module):
    def __init__(self):
        super().__init__()
        self.channel = 60
        self.layer = 8
        self.wsize = 8

    def _isMultiple(self, n, digit):
        if n%digit == 0:
            return 0
        else:
            return 1

    def _padWithReflect(self, img):
        *_, H, W = img.shape
        padding_bottom = self._isMultiple(H,self.wsize)*(self.wsize-H%self.wsize)
        padding_right = self._isMultiple(W,self.wsize)*(self.wsize-W%self.wsize)
        return F.pad(img, (0, padding_right, 0, padding_bottom), mode='reflect')

    def _crop(self, img, h, w):
        return img[:,:,:h,:w]

class SwinCIR(Util): # SwinIR with Connection of Multi Resolutions
    def __init__(self):
        super().__init__()
        self.shallow_feature_extractor = ShallowFE(self.channel)
        self.deep_fusion_feature_extractor = DeepFusionFE(self.channel, self.layer, self.wsize)
        self.hq_image_reconstructor = HQImgRecon(self.channel)

    def forward(self, img):
        *_, H, W = img.shape
        img = self._padWithReflect(img)
        lf = self.shallow_feature_extractor(img)
        #hf, i1, i2, i4 = self.deep_fusion_feature_extractor(lf)
        hf = self.deep_fusion_feature_extractor(lf)
        z = hf+lf
        out = self.hq_image_reconstructor(z)
        #return self._crop(out, 4*H, 4*W), i1, i2, i4
        return self._crop(out, 4*H, 4*W)

class ShallowFE(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.inner = nn.Conv2d(3, channel, 3, padding=1)
    
    def forward(self, img):
        return self.inner(img)

class DeepFusionFE(Util):
    def __init__(self, channel, n, wsize):
        super().__init__()
        """
        Stage 1
        """
        self.block_x4_s1 = CRSTB(channel, n, wsize, fuse=False, fuse_num=None)
        """
        Stage 2
        """
        self.patch_merge_x4_s2 = nn.Conv2d(channel, 2*channel, 4, padding=1, stride=2)
        self.block_x4_s2 = CRSTB(channel, n, wsize, fuse=False, fuse_num=None)
        self.block_x2_s2 = CRSTB(2*channel, n, wsize, fuse=False, fuse_num=None)
        """
        Stage 3
        """
        self.patch_merge_x4_s3_1 = nn.Conv2d(channel, 2*channel, 4, padding=1, stride=2)
        self.patch_merge_x4_s3_2 = nn.Conv2d(2*channel, 4*channel, 4, padding=1, stride=2)
        self.patch_merge_x2_s3 = nn.Conv2d(2*channel, 4*channel, 4, padding=1, stride=2)
        self.patch_expand_x2_s3 = nn.Sequential(
            nn.Conv2d(2*channel, 4*channel, 3, padding=1),
            nn.PixelShuffle(2),
            nn.Conv2d(channel, channel, 3, padding=1)
        )
        self.block_x4_s3 = CRSTB(channel, n, wsize, fuse=True, fuse_num=1)
        self.block_x2_s3 = CRSTB(2*channel, n, wsize, fuse=True, fuse_num=1)
        self.block_x1_s3 = CRSTB(4*channel, n, wsize, fuse=True, fuse_num=1)
        """
        Stage 4
        """
        self.patch_merge_x4_s4_1 = nn.Conv2d(channel, 2*channel, 4, padding=1, stride=2)
        self.patch_merge_x4_s4_2 = nn.Conv2d(2*channel, 4*channel, 4, padding=1, stride=2)
        self.patch_merge_x2_s4 = nn.Conv2d(2*channel, 4*channel, 4, padding=1, stride=2)
        self.patch_expand_x2_s4 = nn.Sequential(
            nn.Conv2d(2*channel, 4*channel, 3, padding=1),
            nn.PixelShuffle(2),
            nn.Conv2d(channel, channel, 3, padding=1)
        )
        self.patch_expand_x1_s4_1 = nn.Sequential(
            nn.Conv2d(4*channel, 8*channel, 3, padding=1),
            nn.PixelShuffle(2),
            nn.Conv2d(2*channel, 2*channel, 3, padding=1)
        )
        self.patch_expand_x1_s4_2 = nn.Sequential(
            nn.Conv2d(2*channel, 4*channel, 3, padding=1),
            nn.PixelShuffle(2),
            nn.Conv2d(channel, channel, 3, padding=1)
        )
        self.block_x4_s4 = CRSTB(channel, n, wsize, fuse=True, fuse_num=2)
        self.block_x2_s4 = CRSTB(2*channel, n, wsize, fuse=True, fuse_num=2)
        self.block_x1_s4 = CRSTB(4*channel, n, wsize, fuse=True, fuse_num=2)

        """
        Stage 5
        """
        self.patch_expand_x2_s5 = nn.Sequential(
            nn.Conv2d(2*channel, 4*channel, 3, padding=1),
            nn.PixelShuffle(2),
            nn.Conv2d(channel, channel, 3, padding=1)
        )
        self.patch_expand_x1_s5_1 = nn.Sequential(
            nn.Conv2d(4*channel, 8*channel, 3, padding=1),
            nn.PixelShuffle(2),
            nn.Conv2d(2*channel, 2*channel, 3, padding=1)
        )
        self.patch_expand_x1_s5_2 = nn.Sequential(
            nn.Conv2d(2*channel, 4*channel, 3, padding=1),
            nn.PixelShuffle(2),
            nn.Conv2d(channel, channel, 3, padding=1)
        )

        self.conv = nn.Conv2d(3*channel, channel, 3, padding=1)

    def forward(self, img):
        """
        Stage 1
        """
        x4 = self.block_x4_s1(img)

        """
        Stage 2
        """
        z2 = self.patch_merge_x4_s2(x4)
        *_, h_x2, w_x2 = z2.shape
        z2 = self._padWithReflect(z2)
        x4 = self.block_x4_s2(x4)
        x2 = self.block_x2_s2(z2)

        """
        Stage 3
        """
        z2 = self.patch_merge_x4_s3_1(x4)
        z2 = self._padWithReflect(z2)
        z1 = self.patch_merge_x4_s3_2(z2)
        *_, h_x1, w_x1 = z1.shape
        z1 = self._padWithReflect(z1)
        y1 = self.patch_merge_x2_s3(x2)
        y1 = self._padWithReflect(y1)
        y4 = self.patch_expand_x2_s3(self._crop(x2, h_x2, w_x2))

        x4 = x4 + y4
        x2 = x2 + z2
        x1 = z1 + y1

        x4 = self.block_x4_s3(x4)
        x2 = self.block_x2_s3(x2)
        x1 = self.block_x1_s3(x1)

        """
        Stage 4
        """
        z2 = self.patch_merge_x4_s4_1(x4)
        z2 = self._padWithReflect(z2)
        z1 = self.patch_merge_x4_s4_2(z2)
        z1 = self._padWithReflect(z1)
        y1 = self.patch_merge_x2_s4(x2)
        y1 = self._padWithReflect(y1)
        y4 = self.patch_expand_x2_s4(self._crop(x2, h_x2, w_x2))
        w2 = self.patch_expand_x1_s4_1(self._crop(x1, h_x1, w_x1))
        w4 = self.patch_expand_x1_s4_2(self._crop(w2, h_x2, w_x2))

        x4 = x4 + y4 + w4
        x2 = x2 + z2 + w2
        x1 = x1 + z1 + y1

        x4 = self.block_x4_s4(x4)
        x2 = self.block_x2_s4(x2)
        x1 = self.block_x1_s4(x1)

        """
        Stage 5
        """
        
        y4 = self.patch_expand_x2_s5(self._crop(x2, h_x2, w_x2))
        w2 = self.patch_expand_x1_s5_1(self._crop(x1, h_x1, w_x1))
        w4 = self.patch_expand_x1_s5_2(self._crop(w2, h_x2, w_x2))

        out = torch.cat((x4, y4, w4), dim=1)
        out = self.conv(out)

        """
        zeros = torch.zeros(x4.shape).cuda()

        i4 = torch.cat((x4, zeros, zeros), dim=1)
        i4 = self.conv(i4)
        i4 = i4[0,:,:,:]


        i2 = torch.cat((zeros, y4, zeros), dim=1)
        i2 = self.conv(i2)
        i2 = i2[0,:,:,:]


        i1 = torch.cat((zeros, zeros, w4), dim=1)
        i1 = self.conv(i1)
        i1 = i1[0,:,:,:]
        
        
        return out, i4, i2, i1
        """
        return out

class HQImgRecon(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.inner = nn.Sequential(
            nn.Conv2d(channel, channel//3, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(channel//3, (channel//3)*4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.Conv2d(channel//3, (channel//3)*4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.Conv2d(channel//3, 3, 3, padding=1),
        )
    
    def forward(self, img):
        return self.inner(img)

class CRSTB(nn.Module): # Residual N STL Block
    def __init__(self, channel, n, wsize, fuse, fuse_num):
        super().__init__()
        self.fuse = fuse
        self.fuse_num = fuse_num
        if fuse:
            self.adjMtrxes = nn.ParameterList([nn.Parameter(torch.randn((wsize*wsize, channel))) for _ in range(fuse_num)])
        self.conv1 = nn.Conv2d(channel, channel, 3, padding=1)
        self.inner = nn.Sequential(*[STL(i, channel, wsize) for i in range(n)])
        self.conv2 = nn.Conv2d(channel, channel, 3, padding=1)
        self.wsize = wsize

    def forward(self, img):
        if self.fuse:
            B, C, H, W = img.shape
            cvrt_img = img.permute(0, 2, 3, 1) # (B,C,H,W) -> (B,H,W,C)
            cvrt_img = cvrt_img.view(B, H//self.wsize, self.wsize, W//self.wsize, self.wsize, C).transpose(2, 3).reshape(-1, self.wsize*self.wsize, C)
            for idx in range(self.fuse_num):
                adjust_matrix = self.adjMtrxes[idx]
                cvrt_img += adjust_matrix.unsqueeze(dim=0)
            img = cvrt_img.view(B, H//self.wsize, W//self.wsize, self.wsize, self.wsize, C).transpose(2, 3).reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        
        raw = self.conv1(img)
        cvrt_img = raw.permute(0, 2, 3, 1) # (B,C,H,W) -> (B,H,W,C)
        cvrt_img = self.inner(cvrt_img)
        raw = cvrt_img.permute(0, 3, 1, 2)# (B,H,W,C) -> (B,C,H,W)
        out = self.conv2(raw)
        return out

class STL(nn.Module):
    def __init__(self, order, channel, wsize):
        super().__init__()
        cycShft = (order%2 != 0)
        self.inner1 = nn.Sequential(
            nn.LayerNorm(channel),
            MSA(cycShft, channel, wsize)
        )
        self.inner2 = nn.Sequential(
            nn.LayerNorm(channel),
            MLP(channel)
        )

    def forward(self, cvrt_img):
        x = cvrt_img
        x = self.inner1(x)
        z = x + cvrt_img

        r = z
        z = self.inner2(z)
        out = z + r

        return out

class MSA(nn.Module):
    def __init__(self, cycShft, channel, wsize):
        super().__init__()
        self.cyc_shft_wndw_partition = CycShftWndwPartition(wsize, cycShft)
        self.self_attention = SelfAttention(channel, wsize)
        self.un_cyc_shft_wndw_partition = UnCycShftWndwPartition(wsize, cycShft)
    
    def forward(self, cvrt_img):
        windows, mask, shape = self.cyc_shft_wndw_partition(cvrt_img)
        windows = self.self_attention(windows, mask)
        new_cvrt_img = self.un_cyc_shft_wndw_partition(windows, shape)
        return new_cvrt_img #(B, H, W, C)


class CycShftWndwPartition(nn.Module):
    def __init__(self, window_size, cycShft):
        super().__init__()
        self.wsize = window_size
        self.cycShft = cycShft
        self.h_slices = [slice(0,-window_size), slice(-window_size,-window_size//2), slice(-window_size//2,None)]
        self.w_slices = [slice(0,-window_size), slice(-window_size,-window_size//2), slice(-window_size//2,None)]
    
    def _mask(self, H, W):
        att_partition = torch.zeros((1,H,W))
        attention_idx = 0
        for h_slice in self.h_slices:
            for w_slice in self.w_slices:
                att_partition[:,h_slice,w_slice] = attention_idx
                attention_idx += 1
        att_partition = att_partition.view(1, H//self.wsize, self.wsize, W//self.wsize, self.wsize)
        att_partition = att_partition.transpose(2, 3).reshape(-1, self.wsize*self.wsize)
        mask = att_partition.unsqueeze(1) - att_partition.unsqueeze(2) #(i,j): 0 if "i" is in same window with "j"
        mask = mask.masked_fill(mask==0, 0.0)
        mask = mask.masked_fill(mask!=0, -100.0)
        return mask # (H/w*W/w, N, N)

    def forward(self, cvrt_img):
        B, H, W, C = cvrt_img.shape
        if self.cycShft:
            x = torch.roll(cvrt_img, shifts=(-self.wsize//2,-self.wsize//2), dims=(1,2))
            mask = self._mask(H,W).to(x.device)
        else:
            x = cvrt_img
            mask = torch.zeros((H*W//(self.wsize*self.wsize),self.wsize*self.wsize,self.wsize*self.wsize)).to(x.device)
        x = x.view(B, H//self.wsize, self.wsize, W//self.wsize, self.wsize, C)
        windows = x.transpose(2, 3).reshape(-1, self.wsize*self.wsize, C) #(B=B*H/w*W/w, N=w*w, C)

        return windows, mask, (B, H, W, C)

class UnCycShftWndwPartition(nn.Module):
    def __init__(self, window_size, cycShft):
        super().__init__()
        self.wsize = window_size
        self.cycShft = cycShft
    
    def forward(self, windows, shape):
        B, H, W, C = shape
        x = windows.view(B, H//self.wsize, W//self.wsize, self.wsize, self.wsize, C)
        x = x.transpose(2, 3).reshape(B, H, W, C)
        if self.cycShft:
            cvrt_img = torch.roll(x, shifts=(self.wsize//2,self.wsize//2), dims=(1,2))
        else:
            cvrt_img = x
        return cvrt_img

class SelfAttention(nn.Module):
    def __init__(self, channel, wsize):
        super().__init__()
        self.wsize = wsize
        self.qkv = nn.Linear(channel, 3*channel, bias=False)
        self.biasMatrix = nn.Parameter(torch.zeros((2*wsize-1)**2, 6))
        self.relativeIndex = self._getRelativeIndex()
        self.softmax = nn.Softmax(dim=-1)
        self.proj = nn.Linear(channel, channel)

    def _getRelativeIndex(self):
        h_cord = torch.arange(self.wsize)
        w_cord = torch.arange(self.wsize)
        h_grid, w_grid = torch.meshgrid([h_cord, w_cord], indexing="ij") # (8,8), (8,8)
        x = torch.stack((h_grid, w_grid)) # (2,8,8)
        x = torch.flatten(x, 1) # (2,64)
        x = x.unsqueeze(dim=2) - x.unsqueeze(dim=1) #(2,64,64), (i,j): distance from i to j
        x[0,:,:] += (self.wsize-1)
        x[0,:,:] *= (2*self.wsize - 1)
        x[1,:,:] += (self.wsize-1)
        relative_index_matrix = x[0,:,:] + x[1,:,:] # (64,64)
        return relative_index_matrix.reshape(-1)

    def forward(self, windows, mask):
        B, N, C = windows.shape
        WNum, *_ = mask.shape #(windownum, N, N)

        qkv = self.qkv(windows).view(B, N, 3, 6, C//6).permute(2,0,3,1,4) #(3,B,headnum,N,dimension)
        q,k,v = qkv[0], qkv[1], qkv[2]

        x = torch.matmul(q, k.transpose(-2,-1)) / ((C//6)**0.5) #(B,headnum,N,N)
        relative_pos_bias = self.biasMatrix[self.relativeIndex].view((self.wsize*self.wsize),(self.wsize*self.wsize),6).permute(2,0,1) #(headnum,64,64)
        x = x+relative_pos_bias.unsqueeze(dim=0) #(B,headnum,N=w*w=64,N)
        x = x.view(B//WNum, WNum, 6, N, N).transpose(1, 2) + mask.view(1, 1, WNum, N, N)
        x = x.transpose(1,2).reshape(-1, 6, N, N)
        attention = self.softmax(x)
        self_attention = torch.matmul(attention, v) #(B,headnum,N,dimension)
        z = self_attention.transpose(1,2).reshape(B, N, C)
        new_windows = self.proj(z)
        return new_windows #(B, w*w, C)

class MLP(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.inner = nn.Sequential(
            nn.Linear(channel,2*channel),
            nn.GELU(),
            nn.Linear(2*channel,channel)
        )
    
    def forward(self, cvrt_img):
        return self.inner(cvrt_img)

