from typing import List, Tuple, Type
from dynamic_network_architectures.architectures.unet import PlainConvUNet

from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd


class PlainConvUNetUpgraded(PlainConvUNet):
    def __init__(
        self,
        input_channels: int,
        n_stages: int,
        features_per_stage: int | List[int] | Tuple[int, ...],
        conv_op: Type[_ConvNd],
        kernel_sizes: int | List[int] | Tuple[int, ...],
        strides: int | List[int] | Tuple[int, ...],
        n_conv_per_stage: int | List[int] | Tuple[int, ...],
        num_classes: int,
        n_conv_per_stage_decoder: int | Tuple[int, ...] | List[int],
        conv_bias: bool = False,
        norm_op: Type[nn.Module] | None = None,
        norm_op_kwargs: dict = None,
        dropout_op: Type[_DropoutNd] | None = None,
        dropout_op_kwargs: dict = None,
        nonlin: Type[nn.Module] | None = None,
        nonlin_kwargs: dict = None,
        deep_supervision: bool = False,
        nonlin_first: bool = False,
    ):
        print("Building Upgraded PlainConvUNet Network")
        super().__init__(
            input_channels,
            n_stages,
            features_per_stage,
            conv_op,
            kernel_sizes,
            strides,
            n_conv_per_stage,
            num_classes,
            n_conv_per_stage_decoder,
            conv_bias,
            norm_op,
            norm_op_kwargs,
            dropout_op,
            dropout_op_kwargs,
            nonlin,
            nonlin_kwargs,
            deep_supervision,
            nonlin_first,
        )

    def forward(self, x, return_skips=False):
        skips = self.encoder(x)
        returns = self.decoder(skips)
        if return_skips:
            return returns, skips
        return returns
