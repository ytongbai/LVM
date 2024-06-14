import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops.layers.torch import Rearrange


def normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

def swish(x):
    return x*torch.sigmoid(x)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, activation_fn="relu"):
        super(ResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.norm1 = normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = normalize(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        if self.in_channels != self.out_channels:
            self.conv_out = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.activation_fn = activation_fn
        if activation_fn=="relu":
            self.actn = nn.ReLU()


    def forward(self, x_in):
        x = x_in
        x = self.norm1(x)
        if self.activation_fn=="relu":
            x = self.actn(x)
        elif self.activation_fn=="swish":
            x = swish(x)
        x = self.conv1(x)
        x = self.norm2(x)
        if self.activation_fn=="relu":
            x = self.actn(x)
        elif self.activation_fn=="swish":
            x = swish(x)
        x = self.conv2(x)
        if self.in_channels != self.out_channels:
            x_in = self.conv_out(x_in)

        return x + x_in

class Encoder(nn.Module):
    def __init__(self, ):
        super().__init__()

        self.filters = 128
        self.num_res_blocks =  2
        self.ch_mult = [1,1,2,2,4]
        self.in_ch_mult = (1,)+tuple(self.ch_mult)
        self.embedding_dim = 32
        self.conv_downsample =  False

        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1, bias=False)
        blocks = []
        for i in range(len(self.ch_mult)):
            block_in_ch = self.filters  * self.in_ch_mult[i]
            block_out_ch = self.filters  * self.ch_mult[i]
            for _ in range(self.num_res_blocks):
                blocks.append(ResBlock(block_in_ch, block_out_ch, activation_fn="swish"))
                block_in_ch = block_out_ch
        for _ in range(self.num_res_blocks):
            blocks.append(ResBlock(block_in_ch, block_out_ch, activation_fn="swish"))
        self.norm1 = normalize(block_in_ch)
        self.conv2 = nn.Conv2d(block_in_ch, self.embedding_dim, kernel_size=1, stride=1, padding=0)
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        x = self.conv1(x)
        for i in range(len(self.ch_mult)):
            for j in range(self.num_res_blocks):
                x = self.blocks[i*2+j](x)

            if i < len(self.ch_mult) -1:
                x = torch.nn.functional.avg_pool2d(x, (2,2),(2,2))

        x = self.blocks[-2](x)
        x = self.blocks[-1](x)

        x = self.norm1(x)
        x = swish(x)
        x = self.conv2(x)
        return x

class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size=8192, emb_dim=32, beta=None):
        super(VectorQuantizer, self).__init__()
        self.codebook_size = codebook_size  # number of embeddings
        self.emb_dim = emb_dim  # dimension of embedding
        self.embedding = nn.Embedding(self.codebook_size, self.emb_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.codebook_size, 1.0 / self.codebook_size)
        self.beta=0.0
        self.z_dim = emb_dim

    def forward(self, z):
        # preprocess

        b, c, h, w = z.size()
        flatten = z.permute(0, 2, 3, 1).reshape(-1, c)
        codebook = self.embedding.weight
        with torch.no_grad():
            tokens = torch.cdist(flatten, codebook).argmin(dim=1)
        quantized = F.embedding(tokens,
                                codebook).view(b, h, w, c).permute(0, 3, 1, 2)

        # compute loss
        codebook_loss = F.mse_loss(quantized, z.detach())
        commitment_loss = F.mse_loss(quantized.detach(), z)
        loss = codebook_loss + self.beta * commitment_loss

        # perplexity
        counts = F.one_hot(tokens, self.codebook_size).sum(dim=0).to(z.dtype)
        # dist.all_reduce(counts)
        p = counts / counts.sum()
        perplexity = torch.exp(-torch.sum(p * torch.log(p + 1e-10)))

        # postprocess
        tokens = tokens.view(b, h, w)
        quantized = z + (quantized - z).detach()

        # quantized_2 = self.get_codebook_feat(tokens, (b, h, w, c))

        return quantized, tokens, loss, perplexity


    def get_codebook_feat(self, indices, shape=None):
        # input indices: batch*token_num -> (batch*token_num)*1
        # shape: batch, height, width, channel
        indices = indices.view(-1,1)
        min_encodings = torch.zeros(indices.shape[0], self.codebook_size).to(indices)
        min_encodings.scatter_(1, indices, 1)
        # get quantized latent vectors
        z_q = torch.matmul(min_encodings.float(), self.embedding.weight)

        if shape is not None:  # reshape back to match original input shape
            z_q = z_q.view(shape).permute(0, 3, 1, 2).contiguous()

        return z_q


class Decoder(nn.Module):
    def __init__(self,):
        super().__init__()
        self.filters = 128
        self.num_res_blocks =  2
        self.ch_mult = [1,1,2,2,4]
        self.in_ch_mult = (1,)+tuple(self.ch_mult)
        self.embedding_dim =32
        self.out_channels = 3
        self.in_channels = self.embedding_dim
        self.conv_downsample =  False

        self.conv1 = nn.Conv2d(32, 512, kernel_size=3, stride=1, padding=1)
        blocks = []
        block_in_ch = self.filters * self.ch_mult[-1]
        block_out_ch = self.filters * self.ch_mult[-1]
        #blocks.append(nn.Conv2d(self.in_channels, block_in_ch, kernel_size=3, stride=1, padding=1))
        for _ in range(self.num_res_blocks):
            blocks.append(ResBlock(block_in_ch, block_out_ch, activation_fn="swish"))
        upsample_conv_layers = []
        for i in reversed(range(len(self.ch_mult))):
            block_out_ch = self.filters * self.ch_mult[i]
            for _ in range(self.num_res_blocks):
                blocks.append(ResBlock(block_in_ch, block_out_ch, activation_fn="swish"))
                block_in_ch = block_out_ch
            if i > 0:
                upsample_conv_layers.append(nn.Conv2d(block_in_ch, block_out_ch*4, kernel_size=3, stride=1, padding=1))

        self.upsample = Rearrange("b h w (h2 w2 c) -> b (h h2) (w w2) c", h2=2, w2=2)
        self.norm1 = normalize(block_in_ch)
        # self.act_fn
        self.conv6 = nn.Conv2d(block_in_ch, self.out_channels, kernel_size=3, stride=1, padding=1)
        self.blocks = nn.ModuleList(blocks)
        self.up_convs = nn.ModuleList(upsample_conv_layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.blocks[0](x)
        x = self.blocks[1](x)
        for i in range(len(self.ch_mult)):
            for j in range(self.num_res_blocks):
                x = self.blocks[2+i*2+j](x)
            if i < len(self.ch_mult)-1:
                x = self.up_convs[i](x)
                #print("pre: x.size()",x.size())
                x = x.permute(0,2,3,1)
                x = self.upsample(x)
                x = x.permute(0,3,1,2)
                #print("post: x.size()", x.size())
        x = self.norm1(x)
        x = swish(x)
        x = self.conv6(x)
        return x


class VQVAE(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.encoder = Encoder()
        self.quantizer = VectorQuantizer()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        quant,tokens, loss, perplexity = self.quantizer(x)
        x = self.decoder(quant)
        return x

    def tokenize(self, x):
        batch_shape = x.shape[:-3]
        x = x.reshape(-1, *x.shape[-3:])
        x = self.encoder(x)
        quant,tokens, loss, perplexity = self.quantizer(x)
        return tokens.reshape(*batch_shape, *tokens.shape[1:])

    def decode(self, tokens):
        tokens = einops.rearrange(tokens, 'b ... -> b (...)')
        b = tokens.shape[0]
        if tokens.shape[-1] == 256:
            hw = 16
        elif tokens.shape[-1] == 224:
            hw = 14
        else:
            raise ValueError("Invalid tokens shape")
        quant = self.quantizer.get_codebook_feat(tokens, (b, hw, hw, 32))
        x = self.decoder(quant)
        return x


class VAEDecoder(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.quantizer = VectorQuantizer()
        self.decoder = Decoder()

    def forward(self, x):
        quant = self.quantizer.get_codebook_feat(x,(1,14,14,32))
        x = self.decoder(quant)
        return x


def get_tokenizer():
    checkpoint_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "xh_ckpt.pth"
    )
    torch_state_dict = torch.load(checkpoint_path)
    net = VQVAE()
    net.load_state_dict(torch_state_dict)
    return net

