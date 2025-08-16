import torch.nn as nn
from torch.autograd import Function
import torch

class RankBasedLinearDropout(nn.Module):
    def __init__(self, pmin=0.1, pmax=0.3):
        super().__init__()
        self.pmin = pmin
        self.pmax = pmax

    def forward(self, x):
        if not self.training:
            return x

        sorted_x, indices = torch.sort(x, dim=-1, descending=False) # 沿最后一个维度排序
        N = x.size(-1) # 生成线性概率 (0.2~0.5)
        ranks = torch.linspace(self.pmin, self.pmin, N, device=x.device)
        inv_indices = torch.argsort(indices, dim=-1) # 根据原位置恢复概率分布
        probs = torch.gather(ranks.expand_as(x), -1, inv_indices)
        mask = (torch.rand_like(x) > probs).float() # 生成掩码并保持期望

        return x * mask / (1 - probs.mean(dim=-1, keepdim=True))


class RandomKSelectionFunction(Function):
    @staticmethod
    def forward(ctx, x, k):
        # 保存输入形状和随机索引用于反向传播
        ctx.save_for_backward(x)
        ctx.k = k
        batch_size, channels, height, width = x.shape
        total_elements = height * width

        x_flat = x.view(batch_size, channels, -1)
        rand_weights = torch.rand(batch_size, channels, total_elements, device=x.device) # 生成随机索引（无重复）
        _, indices = torch.topk(rand_weights, k, dim=-1)
        selected_values = torch.gather(x_flat, -1, indices)
        sorted_values, sort_indices = torch.sort(selected_values, dim=-1)

        ctx.indices = indices
        ctx.sort_indices = sort_indices

        return sorted_values

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        batch_size, channels, height, width = x.shape
        grad_input = torch.zeros_like(x)
        grad_input_flat = grad_input.view(batch_size, channels, -1)

        unsorted_grad = torch.gather(
            grad_output,
            -1,
            ctx.sort_indices.argsort(-1)
        )
        # 将梯度散射回原始位置
        grad_input_flat.scatter_(
            dim=-1,
            index=ctx.indices,
            src=unsorted_grad
        )

        return grad_input, None

class RandomKSelection(nn.Module):
    def __init__(self, k):
        super(RandomKSelection, self).__init__()
        self.k = k  # 要选择的值的数量

    def forward(self, x):
        return RandomKSelectionFunction.apply(x, self.k)

class TopKPooling(nn.Module):
    def __init__(self, k=1, descending=True, dropout=False):
        super(TopKPooling, self).__init__()
        self.k = k
        self.descending = descending
        self.dropout = dropout
        self.dropout_layer = RankBasedLinearDropout() if dropout else None

    def forward(self, x):
        batch, channels, height, width = x.shape
        x_flat = x.view(batch, channels, -1)  # (batch, channels, height*width)
        topk_values, topk_indices = torch.topk(
            x_flat,
            k=self.k,
            dim=-1,
            largest=self.descending,
            sorted=True  # 保持排序顺序
        )
        self.original_shape = x.shape
        self.topk_indices = topk_indices
        
        if self.dropout > 0 and self.training:
            topk_values = self.dropout_layer(topk_values)

        return topk_values

    def backward(self, grad_output):
        batch, channels, height, width = self.original_shape
        grad_input = torch.zeros(self.original_shape,
                                 device=grad_output.device,
                                 dtype=grad_output.dtype)
        # 将梯度散射回原始位置
        grad_input = grad_input.view(batch, channels, -1)
        grad_input.scatter_(
            dim=-1,
            index=self.topk_indices,
            src=grad_output
        )

        return grad_input.view(batch, channels, height, width)

class SNet(nn.Module):
    def __init__(self):
        super(SNet, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer1 = self.creat_layer(3, 64)
        self.layer2 = self.creat_layer(64, 128)
        self.layer3 = self.creat_layer(128, 256)
        self.layer4 = self.creat_layer(256, 512)

    def creat_layer(self, in_c, out_c):
        layer = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=2, padding=0),
            nn.BatchNorm2d(out_c),
            nn.ReLU())

        return layer

    def forward(self, x):
        out = self.maxpool(self.layer1(x))
        out = self.maxpool(self.layer2(out))
        out = self.maxpool(self.layer3(out))
        out = self.layer4(out)

        return out

class AttentionPooling(nn.Module):
    def __init__(self, in_channels):
        super(AttentionPooling, self).__init__()
        # 用 1x1 卷积生成注意力权重（输出单通道）
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=1),  # 压缩通道数
            nn.ReLU(),
            nn.Conv2d(in_channels // 8, 1, kernel_size=1),  # 输出 [B, 1, H, W]
            nn.Softmax(dim=-1))  # 在空间维度 (H,W) 上做 Softmax

    def forward(self, x):
        B, C, H, W = x.shape
        # 生成注意力权重 [B, 1, H, W]
        attn_weights = self.attention(x)
        #print(attn_weights.shape)
        # [B, C, H, W] * [B, 1, H, W] -> sum over (H,W) -> [B, C]
        feature_vector = (x * attn_weights).sum(dim=(2, 3))

        return feature_vector

class LFMSN(nn.Module):
    def __init__(self, num_classes=1, factor=2, rbld=True, rks=False):
        super(LFMSN, self).__init__()
        self.factor = factor

        self.downsample_conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=2, stride=2, padding=0,bias=False)
        self.upsample_conv = nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=2, stride=2, padding=0, output_padding=0, bias=False)

        self.cnn = SNet()
        self.covn1_1 = nn.Conv2d(512, 64, kernel_size=1, padding=0, bias=False)

        self.k = 16
        self.rbld = rbld
        self.pool_adapter = TopKPooling(k=self.k, dropout=True) if self.rbld else TopKPooling(k=self.k, dropout=False)
        # self.pool_adapter = AttentionPooling(64)
        self.rsk = rks
        self.RSK_module = RandomKSelection(k=self.k) if self.rsk else None
        #self.pool_AAP = nn.AdaptiveAvgPool2d((int(self.k**0.5), int(self.k**0.5)))

        self.fc1 = nn.Linear(1024, num_classes)

        self.up_down_module_weight_init()
        self.freeze_and_unfreeze(up_down_module_grad=False,cnn_grad=True)

    def up_down_module_weight_init(self):
        down_value = torch.tensor([
            [[[1, 0], [0, 0]],
             [[0, 0], [0, 0]],
             [[0, 0], [0, 0]]],
            [[[0, 0], [0, 0]],
             [[1, 0], [0, 0]],
             [[0, 0], [0, 0]]],
            [[[0, 0], [0, 0]],
             [[0, 0], [0, 0]],
             [[1, 0], [0, 0]]]], dtype=torch.float32)
        up_value = torch.tensor([
            [[[1, 1], [1, 1]],
             [[0, 0], [0, 0]],
             [[0, 0], [0, 0]]],
            [[[0, 0], [0, 0]],
             [[1, 1], [1, 1]],
             [[0, 0], [0, 0]]],
            [[[0, 0], [0, 0]],
             [[0, 0], [0, 0]],
             [[1, 1], [1, 1]]]], dtype=torch.float32)
        self.downsample_conv.weight.data = down_value
        self.upsample_conv.weight.data = up_value

    def freeze_and_unfreeze(self, up_down_module_grad = False, cnn_grad = True):
        # freeze(冻结) or unfreeze 层
        for param in self.parameters():
            param.requires_grad = cnn_grad
        index=0
        for param in self.parameters():
            param.requires_grad = up_down_module_grad
            index+=1
            if index==2:break

    def load_pre_weights(self, model_path):
        self.load_state_dict(torch.load(model_path, weights_only=True, map_location='cpu'))

    def encoder(self, x):
        x_down = self.downsample_conv(x)
        x_up = self.upsample_conv(x_down)
        NPR  = x - x_up
        maps = self.cnn(abs(NPR))
        maps = self.covn1_1(maps)
        vector = self.pool_adapter(maps).reshape(self.pool_adapter(maps).size(0), -1)

        return vector

    def forward(self, x):
        x_down = self.downsample_conv(x)
        x_up = self.upsample_conv(x_down)
        NPR  = x - x_up
        maps = self.cnn(abs(NPR))
        maps = self.covn1_1(maps)
        vector = self.pool_adapter(maps).reshape(self.pool_adapter(maps).size(0), -1)
        out = self.fc1(vector)

        if self.rsk:
            vector_assist = self.RSK_module(maps).reshape(maps.size(0), -1)
            out_assist = self.fc1(vector_assist)
            return out, out_assist
            if not self.rsk:
                vector_assist = self.pool_AAP(maps).reshape(maps.size(0), -1)
                out_assist = self.fc1(vector_assist)
                return out, out_assist

        return out, out