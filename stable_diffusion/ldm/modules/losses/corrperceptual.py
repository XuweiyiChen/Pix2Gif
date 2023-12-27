import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import vgg19
from torch.autograd import Function, Variable

# try:
#     import resample2d_cuda
#     import local_attn_reshape_cuda
#     import block_extractor_cuda
# except ImportError:
#     print('Warning! Import resample2d_cuda/local_attn_reshape_cuda/block_extractor_cuda, If you are training network, please install them firstly.')
#     print()


#################################################
# borrowed from from https://github.com/RenYurui/Global-Flow-Local-Attention
#################################################

class BlockExtractorFunction(Function):

    @staticmethod
    def forward(ctx, source, flow_field, kernel_size):
        assert source.is_contiguous()
        assert flow_field.is_contiguous()

        # TODO: check the shape of the inputs
        bs, ds, hs, ws = source.size()
        bf, df, hf, wf = flow_field.size()
        # assert bs==bf and hs==hf and ws==wf
        assert df==2

        ctx.save_for_backward(source, flow_field)
        ctx.kernel_size = kernel_size

        output = flow_field.new(bs, ds, kernel_size*hf, kernel_size*wf).zero_()

        if not source.is_cuda:
            raise NotImplementedError
        else:
            block_extractor_cuda.forward(source, flow_field, output, kernel_size)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        if not grad_output.is_contiguous():
            grad_output.contiguous()
        source, flow_field = ctx.saved_tensors
        grad_source = Variable(source.new(source.size()).zero_())
        grad_flow_field = Variable(flow_field.new(flow_field.size()).zero_())

        block_extractor_cuda.backward(source, flow_field, grad_output.data,
                                 grad_source.data, grad_flow_field.data,
                                 ctx.kernel_size)

        return grad_source, grad_flow_field, None

class BlockExtractor(nn.Module):
    def __init__(self, kernel_size=3):
        super(BlockExtractor, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, source, flow_field):
        source_c = source.contiguous()
        flow_field_c = flow_field.contiguous()
        return BlockExtractorFunction.apply(source_c, flow_field_c,
                                          self.kernel_size)

class LocalAttnReshapeFunction(Function):

    @staticmethod
    def forward(ctx, inputs, kernel_size):
        assert inputs.is_contiguous()

        # TODO: check the shape of the inputs
        bs, ds, hs, ws = inputs.size()
        assert ds == kernel_size*kernel_size

        ctx.save_for_backward(inputs)
        ctx.kernel_size = kernel_size

        output = inputs.new(bs, 1, kernel_size*hs, kernel_size*ws).zero_()

        if not inputs.is_cuda:
            raise NotImplementedError
        else:
            local_attn_reshape_cuda.forward(inputs, output, kernel_size)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        if not grad_output.is_contiguous():
            grad_output.contiguous()
        inputs, = ctx.saved_tensors
        grad_inputs = Variable(inputs.new(inputs.size()).zero_())

        local_attn_reshape_cuda.backward(inputs, grad_output.data,
                                 grad_inputs.data, ctx.kernel_size)

        return grad_inputs, None

class LocalAttnReshape(nn.Module):
    def __init__(self):
        super(LocalAttnReshape, self).__init__()

    def forward(self, inputs, kernel_size=3):
        inputs_c = inputs.contiguous()
        return LocalAttnReshapeFunction.apply(inputs_c, kernel_size)

class Resample2dFunction(Function):

    @staticmethod
    def forward(ctx, input1, input2, kernel_size=2, dilation=1):
        assert input1.is_contiguous()
        assert input2.is_contiguous()

        ctx.save_for_backward(input1, input2)
        ctx.kernel_size = kernel_size
        ctx.dilation = dilation

        _, d, _, _ = input1.size()
        b, _, h, w = input2.size()
        output = input1.new(b, d, h, w).zero_()

        resample2d_cuda.forward(input1, input2, output, kernel_size, dilation)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        if not grad_output.is_contiguous():
            grad_output.contiguous()

        input1, input2 = ctx.saved_tensors

        grad_input1 = Variable(input1.new(input1.size()).zero_())
        grad_input2 = Variable(input1.new(input2.size()).zero_())

        resample2d_cuda.backward(input1, input2, grad_output.data,
                                 grad_input1.data, grad_input2.data,
                                 ctx.kernel_size, ctx.dilation)

        return grad_input1, grad_input2, None, None

class Resample2d(nn.Module):

    def __init__(self, kernel_size=2, dilation=1, sigma=5 ):
        super(Resample2d, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.sigma = torch.tensor(sigma, dtype=torch.float)

    def forward(self, input1, input2):
        input1_c = input1.contiguous()
        sigma = self.sigma.expand(input2.size(0), 1, input2.size(2), input2.size(3)).type(input2.dtype)
        input2 = torch.cat((input2,sigma), 1)
        return Resample2dFunction.apply(input1_c, input2, self.kernel_size, self.dilation)

class PerceptualCorrectness(nn.Module):

    def __init__(self, layer=['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1']):
        super(PerceptualCorrectness, self).__init__()
        self.add_module('vgg', VGG19())
        self.layer = layer
        self.eps = 1e-8
        self.resample = Resample2d(4, 1, sigma=2)
        self.l1_loss = nn.L1Loss()

    def __call__(self, target, source, flow_list, used_layers, norm_mask=None, use_bilinear_sampling=True):
        used_layers = sorted(used_layers, reverse=True)
        self.target_vgg, self.source_vgg = self.vgg(target), self.vgg(source)
        loss = 0
        for i in range(len(flow_list)):
            loss += self.calculate_loss(flow_list[i], self.layer[used_layers[i]], norm_mask, use_bilinear_sampling)

        return loss

    def calculate_loss(self, flow, layer, norm_mask=None, use_bilinear_sampling=False):
        target_vgg = self.target_vgg[layer]
        source_vgg = self.source_vgg[layer]
        [b, c, h, w] = target_vgg.shape
        flow = F.interpolate(flow, [h, w])

        target_all = target_vgg.view(b, c, -1)  # [b C N2]
        source_all = source_vgg.view(b, c, -1).transpose(1, 2)  # [b N2 C]

        source_norm = source_all / (source_all.norm(dim=2, keepdim=True) + self.eps)
        target_norm = target_all / (target_all.norm(dim=1, keepdim=True) + self.eps)
        correction = torch.bmm(source_norm, target_norm)  # [b N2 N2]
        (correction_max, max_indices) = torch.max(correction, dim=1)

        # interple with bilinear sampling
        if use_bilinear_sampling:
            input_sample = self.bilinear_warp(source_vgg, flow).view(b, c, -1)
        else:
            input_sample = self.resample(source_vgg, flow).view(b, c, -1)

        correction_sample = F.cosine_similarity(input_sample, target_all)  # [b 1 N2]
        loss_map = torch.exp(-correction_sample / (correction_max + self.eps))
        if norm_mask is None:
            loss = torch.mean(loss_map) - torch.exp(torch.tensor(-1).type_as(loss_map))
        else:
            norm_mask = F.interpolate(norm_mask, size=(target_vgg.size(2), target_vgg.size(3)))
            norm_mask = norm_mask.view(-1, target_vgg.size(2) * target_vgg.size(3))
            loss = (torch.sum(norm_mask * loss_map) - torch.exp(torch.tensor(-1).type_as(loss_map))) / (
                        torch.sum(norm_mask) + self.eps)

        return loss

    def perceptual_loss(self, flow, layer, norm_mask=None, use_bilinear_sampling=False):
        target_vgg = self.target_vgg[layer]
        source_vgg = self.source_vgg[layer]
        [b, c, h, w] = target_vgg.shape
        flow = F.interpolate(flow, [h, w])

        if use_bilinear_sampling:
            input_sample = self.bilinear_warp(source_vgg, flow)
        else:
            input_sample = self.resample(source_vgg, flow).view(b, c, -1)

        if norm_mask is None:
            loss = self.l1_loss(input_sample, target_vgg)
        else:
            norm_mask = F.interpolate(norm_mask, size=(target_vgg.size(2), target_vgg.size(3)))
            loss = self.l1_loss(input_sample * norm_mask, target_vgg * norm_mask)

        return loss

    def bilinear_warp(self, source, flow, view=True):
        b, c = source.shape[:2]
        grid = flow.permute(0, 2, 3, 1)
        input_sample = F.grid_sample(source, grid, mode='bilinear')
        return input_sample

class VGG19(torch.nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        features = vgg19(pretrained=True).features
        self.relu1_1 = torch.nn.Sequential()
        self.relu1_2 = torch.nn.Sequential()

        self.relu2_1 = torch.nn.Sequential()
        self.relu2_2 = torch.nn.Sequential()

        self.relu3_1 = torch.nn.Sequential()
        self.relu3_2 = torch.nn.Sequential()
        self.relu3_3 = torch.nn.Sequential()
        self.relu3_4 = torch.nn.Sequential()

        self.relu4_1 = torch.nn.Sequential()
        self.relu4_2 = torch.nn.Sequential()
        self.relu4_3 = torch.nn.Sequential()
        self.relu4_4 = torch.nn.Sequential()

        self.relu5_1 = torch.nn.Sequential()
        self.relu5_2 = torch.nn.Sequential()
        self.relu5_3 = torch.nn.Sequential()
        self.relu5_4 = torch.nn.Sequential()

        for x in range(2):
            self.relu1_1.add_module(str(x), features[x])

        for x in range(2, 4):
            self.relu1_2.add_module(str(x), features[x])

        for x in range(4, 7):
            self.relu2_1.add_module(str(x), features[x])

        for x in range(7, 9):
            self.relu2_2.add_module(str(x), features[x])

        for x in range(9, 12):
            self.relu3_1.add_module(str(x), features[x])

        for x in range(12, 14):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(14, 16):
            self.relu3_3.add_module(str(x), features[x])

        for x in range(16, 18):
            self.relu3_4.add_module(str(x), features[x])

        for x in range(18, 21):
            self.relu4_1.add_module(str(x), features[x])

        for x in range(21, 23):
            self.relu4_2.add_module(str(x), features[x])

        for x in range(23, 25):
            self.relu4_3.add_module(str(x), features[x])

        for x in range(25, 27):
            self.relu4_4.add_module(str(x), features[x])

        for x in range(27, 30):
            self.relu5_1.add_module(str(x), features[x])

        for x in range(30, 32):
            self.relu5_2.add_module(str(x), features[x])

        for x in range(32, 34):
            self.relu5_3.add_module(str(x), features[x])

        for x in range(34, 36):
            self.relu5_4.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        relu1_1 = self.relu1_1(x)
        relu1_2 = self.relu1_2(relu1_1)

        relu2_1 = self.relu2_1(relu1_2)
        relu2_2 = self.relu2_2(relu2_1)

        relu3_1 = self.relu3_1(relu2_2)
        relu3_2 = self.relu3_2(relu3_1)
        relu3_3 = self.relu3_3(relu3_2)
        relu3_4 = self.relu3_4(relu3_3)

        relu4_1 = self.relu4_1(relu3_4)
        relu4_2 = self.relu4_2(relu4_1)
        relu4_3 = self.relu4_3(relu4_2)
        relu4_4 = self.relu4_4(relu4_3)

        relu5_1 = self.relu5_1(relu4_4)
        relu5_2 = self.relu5_2(relu5_1)
        relu5_3 = self.relu5_3(relu5_2)
        relu5_4 = self.relu5_4(relu5_3)

        out = {
            'relu1_1': relu1_1,
            'relu1_2': relu1_2,

            'relu2_1': relu2_1,
            'relu2_2': relu2_2,

            'relu3_1': relu3_1,
            'relu3_2': relu3_2,
            'relu3_3': relu3_3,
            'relu3_4': relu3_4,

            'relu4_1': relu4_1,
            'relu4_2': relu4_2,
            'relu4_3': relu4_3,
            'relu4_4': relu4_4,

            'relu5_1': relu5_1,
            'relu5_2': relu5_2,
            'relu5_3': relu5_3,
            'relu5_4': relu5_4,
        }
        return out    