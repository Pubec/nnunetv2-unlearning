from dynamic_network_architectures.building_blocks.simple_conv_blocks import ConvDropoutNormReLU

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class DomainPred_ConvReluNormLinear(nn.Module):
    def __init__(self, feature_map: int = 320, kernel_size: int = 4, fm_out: int = 3, index: int = 0, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.index = index;
        self.fm_in = feature_map
        self.ks_in = kernel_size
        blocks = []

        current_feature_map = feature_map
        current_kernel_size = kernel_size

        # using this loop, I will assemble neccessary conv blocks to reach same depth for all blocks in the basic unet
        while current_feature_map <= 320 and current_kernel_size > 4:

            next_feature_map = DomainPred_ConvReluNormLinear.next_power_of_two(current_feature_map)
            if next_feature_map > 320:
                next_feature_map = 320

            singleblock = ConvDropoutNormReLU(
                conv_op=nn.Conv3d,
                input_channels=current_feature_map,
                output_channels=next_feature_map,
                kernel_size=2,
                stride=2,
                conv_bias=False,
                norm_op=nn.InstanceNorm3d,
                nonlin=nn.LeakyReLU,
                nonlin_kwargs={'inplace':True}
                )
            blocks.append(singleblock)

            current_feature_map = next_feature_map

            current_kernel_size /= 2

        singleblock = ConvDropoutNormReLU(
            conv_op=nn.Conv3d,
            input_channels=320,
            output_channels=256,
            kernel_size=1,
            stride=1,
            conv_bias=False,
            norm_op=nn.InstanceNorm3d,
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={'inplace':True}
        )
        blocks.append(singleblock)

        singleblock = ConvDropoutNormReLU(
            conv_op=nn.Conv3d,
            input_channels=256,
            output_channels=64,
            kernel_size=1,
            stride=1,
            conv_bias=False,
            norm_op=nn.InstanceNorm3d,
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={'inplace':True}
        )
        blocks.append(singleblock)

        blocks.append(nn.Flatten())

        blocks.append(nn.Linear(4096, fm_out))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        out = self.blocks(x)
        return out
    
    @staticmethod
    def largest_power_of_two(n):
        if n <= 1:
            return 1

        power_of_two = 1
        while power_of_two * 2 < n:
            power_of_two *= 2

        return power_of_two

    @staticmethod
    def next_power_of_two(n):
        power_of_two = 1
        while power_of_two <= n:
            power_of_two *= 2
        
        return power_of_two


class ConfusionLoss(nn.Module):

    def __init__(self, domains: int, batch_size: int, device: torch.device):
        super(ConfusionLoss, self).__init__()
        self.device = device
        self.target = torch.ones((batch_size, domains)) / domains
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input: Tensor) -> Tensor:
        target = self.target.to(self.device)
        loss = self.loss_fn(input, target)
        return loss


class KLDivergenceConfussionLoss(nn.Module):

    def __init__(self, domains: int, device: torch.device):
        super(KLDivergenceConfussionLoss, self).__init__()
        self.device = device
        self.domains = domains
        self.loss_fn = nn.KLDivLoss(reduction="batchmean")

    def forward(self, input: Tensor) -> Tensor:
        model_probabilities = F.softmax(input, dim=1) + 1e-8
        target_distribution = torch.full_like(model_probabilities, 1/self.domains, device=self.device)
        loss = self.loss_fn(model_probabilities.log() , target_distribution)
        return loss


class RefConfusionLossEpsilonV2(nn.Module):

    def __init__(self):
        super(RefConfusionLossEpsilon, self).__init__()
        self.prior = nn.Softmax(1)

    def forward(self, x):
        log_softmax = F.log_softmax(x + 1e-6, dim=1)
        log_sum = torch.sum(log_softmax, dim=1)
        normalised_log_sum = torch.div(log_sum,  x.size()[1])
        loss = torch.mul(torch.sum(normalised_log_sum, dim=0), -1)
        return loss


class RefConfusionLossEpsilon(nn.Module):

    def __init__(self):
        super(RefConfusionLossEpsilon, self).__init__()
        self.prior = nn.Softmax(1)
        self.eps = 1e-6

    def forward(self, x):
        soft = self.prior(x) + self.eps
        log = torch.log(soft)
        log_sum = torch.sum(log, dim=1)
        normalised_log_sum = torch.div(log_sum,  x.size()[1])
        loss = torch.mul(torch.sum(normalised_log_sum, dim=0), -1)
        return loss

class RefConfusionLoss(nn.Module):

    def __init__(self):
        super(RefConfusionLoss, self).__init__()
        self.prior = nn.Softmax(1)

    def forward(self, x):
        soft = self.prior(x)
        log = torch.log(soft)
        log_sum = torch.sum(log, dim=1)
        normalised_log_sum = torch.div(log_sum,  x.size()[1])
        loss = torch.mul(torch.sum(normalised_log_sum, dim=0), -1)
        return loss


class RefConfusionLossNoNeg(nn.Module):

    def __init__(self):
        super(RefConfusionLossNoNeg, self).__init__()
        self.prior = nn.Softmax(1)

    def forward(self, x):
        soft = self.prior(x)
        log = torch.log(soft)
        log_sum = torch.sum(log, dim=1)
        normalised_log_sum = torch.div(log_sum,  x.size()[1])
        loss = torch.sum(normalised_log_sum, dim=0)
        return loss
