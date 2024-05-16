
import os
import csv
from time import time, sleep
from typing import Union, Tuple, List
from datetime import datetime

import numpy as np
import torch
from torch import autocast, nn
from torch import distributed as dist
from torch.cuda.amp import GradScaler
from torch._dynamo import OptimizedModule
from torch.nn.parallel import DistributedDataParallel as DDP
from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm, convert_dim_to_conv_op

from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.color_transforms import  BrightnessMultiplicativeTransform, ContrastAugmentationTransform, GammaTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms.utility_transforms import RemoveLabelTransform, RenameTransform, NumpyToTensor
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.data_augmentation.custom_transforms.cascade_transforms import MoveSegAsOneHotToData, ApplyRandomBinaryOperatorTransform, RemoveRandomConnectedComponentFromOneHotEncodingTransform
from nnunetv2.training.data_augmentation.custom_transforms.deep_supervision_donwsampling import DownsampleSegForDSTransform2
from nnunetv2.training.data_augmentation.custom_transforms.masking import MaskTransform
from nnunetv2.training.data_augmentation.custom_transforms.region_based_training import ConvertSegmentationToRegionsTransform
from nnunetv2.training.data_augmentation.custom_transforms.transforms_for_dummy_2d import Convert2DTo3DTransform, Convert3DTo2DTransform
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.utilities.network_initialization import InitWeights_He
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from nnunetv2.training.nnUNetTrainer.variants.unlearning.models import DomainPred_ConvReluNormLinear, ConfusionLoss
from nnunetv2.training.logging.unlearn_logger import nnUNetUnlearnLogger
from nnunetv2.training.dataloading.data_loader_unlearn import nnUNetDataLoader3DUnlearn
from nnunetv2.models.models import PlainConvUNetUpgraded


class nnUNetTrainer_UnlearnBase(nnUNetTrainer):

    domainpred_stages = (
        (4, 320),
        (8, 320),
        (16, 256),
        (32, 128),
        (64, 64),
        (128, 32),
    )

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        experiment_identifier: str = "",
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
        dir_can_exist: bool = False
    ):
        super().__init__(
            plans, configuration, fold, dataset_json, experiment_identifier, unpack_dataset, device, dir_can_exist
        )
        # configuration file stuff
        self.domains = self.configuration_manager.configuration['domains']
        self.num_epochs = self.configuration_manager.configuration.get('num_epochs', 500)
        self.num_epochs_stage_0 = self.configuration_manager.configuration.get('num_epochs_stage_0', 50)
        self.confusion_loss_weight = self.configuration_manager.configuration.get('confusion_loss_weight', 0.01)

        self.logger = nnUNetUnlearnLogger(unlearn_plots=len(self.domainpred_stages))
        self.is_unlearn_step = False

        # unlearn indexes define which domain predictors have their confusion loss bacpropagated
        self.network_domainpredictors_unlearning_indexes = self.configuration_manager.configuration.get('network_domainpredictors_unlearning_indexes', [])
        if not self.network_domainpredictors_unlearning_indexes:
            self.network_domainpredictors_unlearning_indexes = list(range(len(self.domainpred_stages)))

        # these steps define what is bacpropagated in the unlearn step
        self.unlearn_step_network_bp = True
        self.unlearn_step_classifier_bp = True
        self.unlearn_step_unlearn_bp = True

        self.save_every = 1

        self.log_raw = self.configuration_manager.configuration.get('log_raw', False)
        if self.log_raw:
            timestamp = datetime.now()
            self.raw_file = join(self.output_folder, "training_raw_%d_%d_%d_%02.0d_%02.0d_%02.0d.csv" %
                            (timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute,
                            timestamp.second))
            self.print_to_raw_csv([["epoch", "batch_index", "domain_target", "domain_prediction", "raw_output", "domain_loss", "domain_accuracy"]])

    def print_unlearning_parameters(self):
        self.print_to_log_file(f"Using trainer: {self.__class__.__name__}")
        self.print_to_log_file(f"{self.domains = }")
        self.print_to_log_file(f"{self.num_epochs = }")
        self.print_to_log_file(f"{self.num_epochs_stage_0 = }")
        self.print_to_log_file(f"{self.confusion_loss_weight = }")
        self.print_to_log_file(f"{self.network_domainpredictors_unlearning_indexes = }")

    def initialize(self):
        if not self.was_initialized:
            self.num_input_channels = determine_num_input_channels(
                self.plans_manager, self.configuration_manager, self.dataset_json
            )

            self.network = self.build_network_architecture(
                self.plans_manager,
                self.dataset_json,
                self.configuration_manager,
                self.num_input_channels,
                enable_deep_supervision=True,
            ).to(self.device)

            # domain predictor
            self.network_domainpredictors = []

            index = 0
            for kernel_size, feature_size in self.domainpred_stages:
                domain_predictor = DomainPred_ConvReluNormLinear(
                    feature_size, kernel_size, self.domains, index
                ).to(self.device)
                self.network_domainpredictors.append(domain_predictor)
                index += 1

            # compile network for free speedup
            if ("nnUNet_compile" in os.environ.keys()) and (
                os.environ["nnUNet_compile"].lower() in ("true", "1", "t")
            ):
                self.print_to_log_file("Compiling network...")
                self.network = torch.compile(self.network)
                for i in range(len(self.network_domainpredictors)):
                    self.network_domainpredictors[i] = torch.compile(
                        self.network_domainpredictors[i]
                    )

            # stage 0 optimizers and parameters
            # optimizer and lr_scheduler have the same name to not mess up with all the other code in trainer
            (
                self.all_optimizers_stage_0,
                self.all_lr_schedulers_stage_0,
            ) = self.configure_optimizers_stage_0()

            self.optimizer_stage0_unet = self.all_optimizers_stage_0[0]
            self.lr_scheduler_stage0_unet = self.all_lr_schedulers_stage_0[0]

            self.optimizers_stage0_domainpred = self.all_optimizers_stage_0[1:]
            self.lr_schedulers_stage0_domainpred = self.all_lr_schedulers_stage_0[1:]
            self.grad_scalers_stage0_domainpred = [
                GradScaler() if self.device.type == "cuda" else None
                for _ in range(len(self.network_domainpredictors))
            ]

            # stage 1 optimizers and parameters
            (
                self.all_optimizers_stage_1,
                self.all_lr_schedulers_stage_1,
            ) = self.configure_optimizers_stage_1()
            self.optimizer_stage1_unet = self.all_optimizers_stage_1[0]
            self.lr_scheduler_stage1_unet = self.all_lr_schedulers_stage_1[0]

            self.optimizers_stage1_domainpred = self.all_optimizers_stage_1[
                1 : len(self.network_domainpredictors) + 1
            ]
            self.lr_schedulers_stage1_domainpred = self.all_lr_schedulers_stage_1[
                1 : len(self.network_domainpredictors) + 1
            ]
            self.grad_scalers_stage1_domainpred = [
                GradScaler() if self.device.type == "cuda" else None
                for _ in range(len(self.network_domainpredictors))
            ]

            self.optimizers_stage1_unlearn = self.all_optimizers_stage_1[
                len(self.network_domainpredictors) + 1 :
            ]
            self.lr_schedulers_stage1_unlearn = self.all_lr_schedulers_stage_1[
                len(self.network_domainpredictors) + 1 :
            ]
            self.grad_scalers_stage1_unlearn = [
                GradScaler() if self.device.type == "cuda" else None
                for _ in range(len(self.network_domainpredictors))
            ]

            # if ddp, wrap in DDP wrapper
            if self.is_ddp:
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
                    self.network
                )
                self.network = DDP(self.network, device_ids=[self.local_rank])

                for i in range(len(self.network_domainpredictors)):
                    self.network_domainpredictors[
                        i
                    ] = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
                        self.network_domainpredictors[i]
                    )
                    self.network_domainpredictors[i] = DDP(
                        self.network_domainpredictors[i], device_ids=[self.local_rank]
                    )

            self.loss = self._build_loss()
            self.loss_domain_predictors = self._build_domainpredictor_loss()
            self.loss_unlearn = self._build_unlearn_loss()

            for network in self.network_domainpredictors:
                self.logger.my_fantastic_logging[
                    f"train_losses_domain_{network.index}"
                ] = list()
                self.logger.my_fantastic_logging[
                    f"train_acc_domain_{network.index}"
                ] = list()
                self.logger.my_fantastic_logging[
                    f"ema_train_acc_domain_{network.index}"
                ] = list()
                self.logger.my_fantastic_logging[
                    f"val_losses_domain_{network.index}"
                ] = list()
                self.logger.my_fantastic_logging[
                    f"val_acc_domain_{network.index}"
                ] = list()
                self.logger.my_fantastic_logging[
                    f"ema_val_acc_domain_{network.index}"
                ] = list()
                self.logger.my_fantastic_logging[
                    f"train_losses_confusion_{network.index}"
                ] = list()
            self.was_initialized = True

            self.print_unlearning_parameters()

        else:
            raise RuntimeError(
                "You have called self.initialize even though the trainer was already initialized. "
                "That should not happen."
            )

    def _build_domainpredictor_loss(self):
        return [nn.CrossEntropyLoss() for _ in self.network_domainpredictors]

    def _build_unlearn_loss(self):
        return [
            ConfusionLoss(
                domains=self.domains,
                batch_size=self.batch_size,
                device=self.device,
            ).to(self.device)
            for _ in self.network_domainpredictors
        ]

    def configure_optimizers_stage_0(self):
        optimizers = []
        lr_schedulers = []
        # network
        optimizer = torch.optim.SGD(
            self.network.parameters(),
            self.initial_lr,
            weight_decay=self.weight_decay,
            momentum=0.99,
            nesterov=True,
        )
        lr_scheduler = PolyLRScheduler(
            optimizer,
            self.initial_lr,
            1000,
            # self.num_epochs,
        )
        optimizers.append(optimizer)
        lr_schedulers.append(lr_scheduler)

        # classifiers
        for i in range(len(self.network_domainpredictors)):
            optimizer = torch.optim.SGD(
                self.network_domainpredictors[i].parameters(),
                self.initial_lr,
                weight_decay=self.weight_decay,
                momentum=0.99,
                nesterov=True,
            )
            lr_scheduler = PolyLRScheduler(
                optimizer,
                self.initial_lr,
                1000,
                # self.num_epochs
            )

            optimizers.append(optimizer)
            lr_schedulers.append(lr_scheduler)

        return optimizers, lr_schedulers

    def configure_optimizers_stage_1(self):
        # optimizers = [
        # optimizer,
        # optimizer domain 1, optimizer domain 2 etc ... optimizer domain 6,
        # optimizer confusion 1, optimizer confusion 2 etc ... optimizer confusion 6
        # ]
        # optimizer unet: entire unet
        # optimizer domain: entire domain predictor
        # optimizer confusion: single unet block
        optimizers = []
        lr_schedulers = []
        # network
        optimizer = torch.optim.SGD(
            self.network.parameters(),
            self.initial_lr,
            weight_decay=self.weight_decay,
            momentum=0.99,
            nesterov=True,
        )
        lr_scheduler = PolyLRScheduler(
            optimizer,
            self.initial_lr,
            1000,
            # self.num_epochs - self.num_epochs_stage_0
        )
        optimizers.append(optimizer)
        lr_schedulers.append(lr_scheduler)

        # classifiers
        for i in range(len(self.network_domainpredictors)):
            optimizer = torch.optim.SGD(
                self.network_domainpredictors[i].parameters(),
                self.initial_lr,
                weight_decay=self.weight_decay,
                momentum=0.99,
                nesterov=True,
            )
            lr_scheduler = PolyLRScheduler(
                optimizer,
                self.initial_lr,
                1000,
                # self.num_epochs - self.num_epochs_stage_0 + 500,
            )

            optimizers.append(optimizer)
            lr_schedulers.append(lr_scheduler)

        # pairs that go together
        # self.network.encoder.stages[-1] <-> self.network_domainpredictors[0]
        # self.network.encoder.stages[-2] <-> self.network_domainpredictors[1]
        # self.network.encoder.stages[-3] <-> self.network_domainpredictors[2]
        # self.network.encoder.stages[-4] <-> self.network_domainpredictors[3]
        # self.network.encoder.stages[-5] <-> self.network_domainpredictors[4]
        # self.network.encoder.stages[-6] <-> self.network_domainpredictors[5]
        # ------
        # self.network.encoder.stages[0] <-> self.network_domainpredictors[-6]
        # self.network.encoder.stages[1] <-> self.network_domainpredictors[-5]
        # self.network.encoder.stages[2] <-> self.network_domainpredictors[-4]
        # self.network.encoder.stages[3] <-> self.network_domainpredictors[-3]
        # self.network.encoder.stages[4] <-> self.network_domainpredictors[-2]
        # self.network.encoder.stages[5] <-> self.network_domainpredictors[-1]

        # domain_predictors[5] is the first one, meaning it covers first stage only [0 / -6]
        # domain_predictors[0] is the last one (the bottleneck), meaning it covers stages from 0 to 5

        # 0 -> -1, -2, -3, -4, -5, -6
        # 1 -> -2, -3, -4, -5, -6
        # 2 -> -3, -4, -5, -6
        # 3 -> -4, -5, -6
        # 4 -> -5, -6
        # 5 -> -6

        self.network_paramaters_for_unlearn = []

        for i in range(len(self.network_domainpredictors)):
            # changed this from updating network to point from beggining, to just updating a single block
            stages_to_point = self.network.encoder.stages[-i - 1:-len(self.network_domainpredictors)-1:-1]
            # stages_to_point = self.network.encoder.stages[-i - 1]
            parameters = stages_to_point.parameters()
            optimizer = torch.optim.SGD(
                parameters,
                self.initial_lr,
                weight_decay=self.weight_decay,
                momentum=0.99,
                nesterov=True,
            )
            lr_scheduler = PolyLRScheduler(
                optimizer,
                self.initial_lr,
                1000,
                # self.num_epochs - self.num_epochs_stage_0,
            )

            optimizers.append(optimizer)
            lr_schedulers.append(lr_scheduler)
            self.network_paramaters_for_unlearn.append(parameters)

        return optimizers, lr_schedulers

    @staticmethod
    def build_network_architecture(
        plans_manager: PlansManager,
        dataset_json,
        configuration_manager: ConfigurationManager,
        num_input_channels,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        num_stages = len(configuration_manager.conv_kernel_sizes)

        dim = len(configuration_manager.conv_kernel_sizes[0])
        conv_op = convert_dim_to_conv_op(dim)

        label_manager = plans_manager.get_label_manager(dataset_json)

        segmentation_network_class_name = configuration_manager.UNet_class_name
        mapping = {
            "PlainConvUNet": PlainConvUNetUpgraded,
        }
        kwargs = {
            "PlainConvUNet": {
                "conv_bias": True,
                "norm_op": get_matching_instancenorm(conv_op),
                "norm_op_kwargs": {"eps": 1e-5, "affine": True},
                "dropout_op": None,
                "dropout_op_kwargs": None,
                "nonlin": nn.LeakyReLU,
                "nonlin_kwargs": {"inplace": True},
            },
        }
        assert (
            segmentation_network_class_name in mapping.keys()
        ), "The network architecture specified by the plans file"
        network_class = mapping[segmentation_network_class_name]

        conv_or_blocks_per_stage = {
            "n_conv_per_stage": configuration_manager.n_conv_per_stage_encoder,
            "n_conv_per_stage_decoder": configuration_manager.n_conv_per_stage_decoder,
        }
        model = network_class(
            input_channels=num_input_channels,
            n_stages=num_stages,
            features_per_stage=[
                min(
                    configuration_manager.UNet_base_num_features * 2**i,
                    configuration_manager.unet_max_num_features,
                )
                for i in range(num_stages)
            ],
            conv_op=conv_op,
            kernel_sizes=configuration_manager.conv_kernel_sizes,
            strides=configuration_manager.pool_op_kernel_sizes,
            num_classes=label_manager.num_segmentation_heads,
            deep_supervision=enable_deep_supervision,
            **conv_or_blocks_per_stage,
            **kwargs[segmentation_network_class_name],
        )
        model.apply(InitWeights_He(1e-2))
        return model

    def print_to_raw_csv(self, rows):
        if self.local_rank == 0:
            with open(self.raw_file, 'a+') as f:
                cw = csv.writer(f, delimiter=",")
                cw.writerows(rows)

    @staticmethod
    def get_training_transforms(
        patch_size: Union[np.ndarray, Tuple[int]],
        rotation_for_DA: dict,
        deep_supervision_scales: Union[List, Tuple],
        mirror_axes: Tuple[int, ...],
        do_dummy_2d_data_aug: bool,
        order_resampling_data: int = 3,
        order_resampling_seg: int = 1,
        border_val_seg: int = -1,
        use_mask_for_norm: List[bool] = None,
        is_cascaded: bool = False,
        foreground_labels: Union[Tuple[int, ...], List[int]] = None,
        regions: List[Union[List[int], Tuple[int, ...], int]] = None,
        ignore_label: int = None,
    ) -> AbstractTransform:
        tr_transforms = []
        if do_dummy_2d_data_aug:
            ignore_axes = (0,)
            tr_transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size
            ignore_axes = None

        tr_transforms.append(
            SpatialTransform(
                patch_size_spatial,
                patch_center_dist_from_border=None,
                do_elastic_deform=False,
                alpha=(0, 0),
                sigma=(0, 0),
                do_rotation=True,
                angle_x=rotation_for_DA["x"],
                angle_y=rotation_for_DA["y"],
                angle_z=rotation_for_DA["z"],
                p_rot_per_axis=1,  # todo experiment with this
                do_scale=True,
                scale=(0.7, 1.4),
                border_mode_data="constant",
                border_cval_data=0,
                order_data=order_resampling_data,
                border_mode_seg="constant",
                border_cval_seg=border_val_seg,
                order_seg=order_resampling_seg,
                random_crop=False,  # random cropping is part of our dataloaders
                p_el_per_sample=0,
                p_scale_per_sample=0.2,
                p_rot_per_sample=0.2,
                independent_scale_for_each_axis=False,  # todo experiment with this
            )
        )

        if do_dummy_2d_data_aug:
            tr_transforms.append(Convert2DTo3DTransform())

        tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.1))
        tr_transforms.append(
            GaussianBlurTransform(
                (0.5, 1.0),
                different_sigma_per_channel=True,
                p_per_sample=0.2,
                p_per_channel=0.5,
            )
        )
        tr_transforms.append(
            BrightnessMultiplicativeTransform(
                multiplier_range=(0.75, 1.25), p_per_sample=0.15
            )
        )
        tr_transforms.append(ContrastAugmentationTransform(p_per_sample=0.15))
        tr_transforms.append(
            SimulateLowResolutionTransform(
                zoom_range=(0.5, 1),
                per_channel=True,
                p_per_channel=0.5,
                order_downsample=0,
                order_upsample=3,
                p_per_sample=0.25,
                ignore_axes=ignore_axes,
            )
        )
        tr_transforms.append(
            GammaTransform((0.7, 1.5), True, True, retain_stats=True, p_per_sample=0.1)
        )
        tr_transforms.append(
            GammaTransform((0.7, 1.5), False, True, retain_stats=True, p_per_sample=0.3)
        )

        if mirror_axes is not None and len(mirror_axes) > 0:
            tr_transforms.append(MirrorTransform(mirror_axes))

        if use_mask_for_norm is not None and any(use_mask_for_norm):
            tr_transforms.append(
                MaskTransform(
                    [i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]],
                    mask_idx_in_seg=0,
                    set_outside_to=0,
                )
            )

        tr_transforms.append(RemoveLabelTransform(-1, 0))

        if is_cascaded:
            assert (
                foreground_labels is not None
            ), "We need foreground_labels for cascade augmentations"
            tr_transforms.append(
                MoveSegAsOneHotToData(1, foreground_labels, "seg", "data")
            )
            tr_transforms.append(
                ApplyRandomBinaryOperatorTransform(
                    channel_idx=list(range(-len(foreground_labels), 0)),
                    p_per_sample=0.4,
                    key="data",
                    strel_size=(1, 8),
                    p_per_label=1,
                )
            )
            tr_transforms.append(
                RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                    channel_idx=list(range(-len(foreground_labels), 0)),
                    key="data",
                    p_per_sample=0.2,
                    fill_with_other_class_p=0,
                    dont_do_if_covers_more_than_x_percent=0.15,
                )
            )

        tr_transforms.append(RenameTransform("seg", "target", True))

        if regions is not None:
            # the ignore label must also be converted
            tr_transforms.append(
                ConvertSegmentationToRegionsTransform(
                    list(regions) + [ignore_label]
                    if ignore_label is not None
                    else regions,
                    "target",
                    "target",
                )
            )

        if deep_supervision_scales is not None:
            tr_transforms.append(
                DownsampleSegForDSTransform2(
                    deep_supervision_scales, 0, input_key="target", output_key="target"
                )
            )
        tr_transforms.append(NumpyToTensor(["data", "target"], "float"))
        tr_transforms.append(NumpyToTensor(["domain"]))
        tr_transforms = Compose(tr_transforms)
        return tr_transforms

    @staticmethod
    def get_validation_transforms(
        deep_supervision_scales: Union[List, Tuple],
        is_cascaded: bool = False,
        foreground_labels: Union[Tuple[int, ...], List[int]] = None,
        regions: List[Union[List[int], Tuple[int, ...], int]] = None,
        ignore_label: int = None,
    ) -> AbstractTransform:
        val_transforms = []
        val_transforms.append(RemoveLabelTransform(-1, 0))

        if is_cascaded:
            val_transforms.append(
                MoveSegAsOneHotToData(1, foreground_labels, "seg", "data")
            )

        val_transforms.append(RenameTransform("seg", "target", True))

        if regions is not None:
            # the ignore label must also be converted
            val_transforms.append(
                ConvertSegmentationToRegionsTransform(
                    list(regions) + [ignore_label]
                    if ignore_label is not None
                    else regions,
                    "target",
                    "target",
                )
            )

        if deep_supervision_scales is not None:
            val_transforms.append(
                DownsampleSegForDSTransform2(
                    deep_supervision_scales, 0, input_key="target", output_key="target"
                )
            )

        val_transforms.append(NumpyToTensor(["data", "target"], "float"))
        val_transforms.append(NumpyToTensor(["domain"]))
        val_transforms = Compose(val_transforms)
        return val_transforms

    def get_plain_dataloaders(self, initial_patch_size: Tuple[int, ...], dim: int):
        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        dl_tr = nnUNetDataLoader3DUnlearn(dataset_tr, self.batch_size,
                                initial_patch_size,
                                self.configuration_manager.patch_size,
                                self.label_manager,
                                oversample_foreground_percent=self.oversample_foreground_percent,
                                sampling_probabilities=None, pad_sides=None, domain_constraint=True, domain_count=self.domains)
        dl_val = nnUNetDataLoader3DUnlearn(dataset_val, self.batch_size,
                                    self.configuration_manager.patch_size,
                                    self.configuration_manager.patch_size,
                                    self.label_manager,
                                    oversample_foreground_percent=self.oversample_foreground_percent,
                                    sampling_probabilities=None, pad_sides=None, domain_constraint=True, domain_count=self.domains)
        return dl_tr, dl_val

    def train_step(self, batch: dict) -> dict:
        data = batch["data"]
        target = batch["target"]
        domain = batch["domain"]

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        domain = domain.to(self.device, non_blocking=True)

        self.optimizer_stage0_unet.zero_grad()
        for optimizer in self.optimizers_stage0_domainpred:
            optimizer.zero_grad()

        # network loss backpropagation
        with autocast(
            self.device.type, enabled=True
        ) if self.device.type == "cuda" else dummy_context():
            output, encoder_outputs = self.network(data, return_skips=True)
            l = self.loss(output, target)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer_stage0_unet)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer_stage0_unet)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer_stage0_unet.step()

        # domainpred loss backpropagation
        domain_true = domain.flatten()
        domain_true_np = domain_true.detach().cpu().numpy()
        l_domain = []
        acc_domain = []

        raw_log_row = []

        with autocast(
            self.device.type, enabled=True
        ) if self.device.type == "cuda" else dummy_context():
            for i in range(len(self.network_domainpredictors)):
                domain_output = self.network_domainpredictors[i](
                    encoder_outputs[-1 - i].detach()
                )
                domain_loss = self.loss_domain_predictors[i](
                    domain_output, domain_true
                )
                l_domain.append(domain_loss)
                domain_pred = torch.argmax(domain_output, axis=1)
                domain_acc = (domain_pred == domain_true).sum() / len(domain)
                acc_domain.append(domain_acc)
                if self.log_raw and i == 0:
                    for i in range(self.batch_size):
                        raw_log_row.append([self.current_epoch, i, domain_true_np[i],  domain_pred[i].detach().cpu().numpy(), domain_output[i].detach().cpu().numpy(), domain_loss.detach().cpu().numpy(), domain_acc.detach().cpu().numpy()])

        if raw_log_row:
            self.print_to_raw_csv(raw_log_row)

        if self.grad_scalers_stage0_domainpred is not None:
            for i in range(len(self.network_domainpredictors)):
                self.grad_scalers_stage0_domainpred[i].scale(l_domain[i]).backward()
                self.grad_scalers_stage0_domainpred[i].unscale_(
                    self.optimizers_stage0_domainpred[i]
                )
                torch.nn.utils.clip_grad_norm_(
                    self.network_domainpredictors[i].parameters(), 12
                )
                self.grad_scalers_stage0_domainpred[i].step(
                    self.optimizers_stage0_domainpred[i]
                )
                self.grad_scalers_stage0_domainpred[i].update()
        else:
            for i in range(len(self.network_domainpredictors)):
                l_domain[i].backwards()
                torch.nn.utils.clip_grad_norm_(
                    self.network_domainpredictors[i].parameters(), 12
                )
                self.optimizers_stage0_domainpred[i].step()

        out = {"loss": l.detach().cpu().numpy()}
        out.update(
            {
                f"loss_domain_{enum}": x.detach().cpu().numpy()
                for enum, x in enumerate(l_domain)
            }
        )
        out.update(
            {
                f"acc_domain_{enum}": x.detach().cpu().numpy()
                for enum, x in enumerate(acc_domain)
            }
        )
        out.update({f"loss_confusion_{enum}": 0.0 for enum in range(len(l_domain))})
        return out

    def train_step_unlearn(self, batch: dict) -> dict:
        data = batch["data"]
        target = batch["target"]
        domain = batch["domain"]

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        domain = domain.to(self.device, non_blocking=True)

        self.optimizer_stage1_unet.zero_grad()
        for optimizer in self.optimizers_stage1_domainpred:
            optimizer.zero_grad()

        # ########################
        # network
        # ########################
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            output, encoder_outputs = self.network(data, return_skips=True)
            l = self.loss(output, target)

        # network loss backprogagation
        if self.unlearn_step_network_bp:
            if self.grad_scaler is not None:
                self.grad_scaler.scale(l).backward(retain_graph=True)
                self.grad_scaler.unscale_(self.optimizer_stage1_unet)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.grad_scaler.step(self.optimizer_stage1_unet)
                self.grad_scaler.update()
            else:
                l.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer_stage1_unet.step()

        # ########################
        # domainpred
        # ########################
        domain_true = domain.flatten()
        domain_true_np = domain_true.detach().cpu().numpy()
        raw_log_row = []

        l_domain = []
        acc_domain = []
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            for i in range(len(self.network_domainpredictors)):
                domain_output = self.network_domainpredictors[i](encoder_outputs[-1 - i].detach())
                domain_loss = self.loss_domain_predictors[i](domain_output, domain_true)
                l_domain.append(domain_loss)
                domain_pred = torch.argmax(domain_output, axis=1)
                domain_acc = (domain_pred == domain_true).sum() / len(domain)
                acc_domain.append(domain_acc)

                if self.log_raw and i == 0:
                    for i in range(self.batch_size):
                        raw_log_row.append([self.current_epoch, i, domain_true_np[i],  domain_pred[i].detach().cpu().numpy(), domain_output[i].detach().cpu().numpy(), domain_loss.detach().cpu().numpy(), domain_acc.detach().cpu().numpy()])

        if raw_log_row:
            self.print_to_raw_csv(raw_log_row)

        # domainpred loss backpropagation
        if self.unlearn_step_classifier_bp:
            if self.grad_scalers_stage1_domainpred is not None:
                for i in range(len(self.network_domainpredictors)):
                    self.grad_scalers_stage1_domainpred[i].scale(l_domain[i]).backward(retain_graph=False)
                    self.grad_scalers_stage1_domainpred[i].unscale_(self.optimizers_stage1_domainpred[i])
                    torch.nn.utils.clip_grad_norm_(self.network_domainpredictors[i].parameters(), 12)
                    self.grad_scalers_stage1_domainpred[i].step(self.optimizers_stage1_domainpred[i])
                    self.grad_scalers_stage1_domainpred[i].update()
            else:
                for i in range(len(self.network_domainpredictors)):
                    l_domain[i].backwards(retain_graph=False)
                    torch.nn.utils.clip_grad_norm_(self.network_domainpredictors[i].parameters(), 12)
                    self.optimizers_stage1_domainpred[i].step()

        # ########################
        # unlearn
        # ########################

        l_conf = []
        for i in range(len(self.network_domainpredictors)):
            with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
                _, encoder_outputs = self.network(data, return_skips=True)
                domain_output = self.network_domainpredictors[i](encoder_outputs[-1 - i])
                confusion_loss = self.confusion_loss_weight * self.loss_unlearn[i](domain_output)
                l_conf.append(confusion_loss)
            
            if self.unlearn_step_unlearn_bp:
                if self.grad_scalers_stage1_unlearn is not None:
                    if i in self.network_domainpredictors_unlearning_indexes:
                        confusion_loss = l_conf[i]
                        self.grad_scalers_stage1_unlearn[i].scale(confusion_loss).backward(retain_graph=False)
                        self.grad_scalers_stage1_unlearn[i].unscale_(self.optimizers_stage1_unlearn[i])
                        torch.nn.utils.clip_grad_norm_(self.network_paramaters_for_unlearn[i], 12)
                        self.grad_scalers_stage1_unlearn[i].step(self.optimizers_stage1_unlearn[i])
                        self.grad_scalers_stage1_unlearn[i].update()
                else:
                    if i in self.network_domainpredictors_unlearning_indexes:
                        confusion_loss = l_conf[i]
                        confusion_loss.backwards(retain_graph=False)
                        torch.nn.utils.clip_grad_norm_(self.network_paramaters_for_unlearn[i], 12)
                        self.optimizers_stage1_unlearn[i].step()

        out = {"loss": l.detach().cpu().numpy()}
        out.update(
            {
                f"loss_domain_{enum}": x.detach().cpu().numpy()
                for enum, x in enumerate(l_domain)
            }
        )
        out.update(
            {
                f"acc_domain_{enum}": x.detach().cpu().numpy()
                for enum, x in enumerate(acc_domain)
            }
        )
        out.update(
            {
                f"loss_confusion_{enum}": x.detach().cpu().numpy()
                for enum, x in enumerate(l_conf)
            }
        )
        return out

    def on_train_epoch_start(self):
        self.network.train()
        for network in self.network_domainpredictors:
            network.train()
        if not self.is_unlearn_step:
            self.lr_scheduler_stage0_unet.step(self.current_epoch)
            for scheduler in self.lr_schedulers_stage0_domainpred:
                scheduler.step(self.current_epoch)
        else:
            self.lr_scheduler_stage1_unet.step(
                self.current_epoch - self.num_epochs_stage_0
            )
            for scheduler in self.lr_schedulers_stage1_domainpred:
                scheduler.step(self.current_epoch - self.num_epochs_stage_0)

        optimizer = (
            self.optimizer_stage0_unet
            if not self.is_unlearn_step
            else self.optimizer_stage1_unet
        )

        self.print_to_log_file("")
        self.print_to_log_file(f"Epoch {self.current_epoch}")
        self.print_to_log_file(f"Unlearn step: {self.is_unlearn_step}")
        self.print_to_log_file(
            f"Current learning rate: {np.round(optimizer.param_groups[0]['lr'], decimals=5)}"
        )
        self.logger.log("lrs", optimizer.param_groups[0]["lr"], self.current_epoch)

    def on_train_epoch_end(self, train_outputs: List[dict]):
        outputs = collate_outputs(train_outputs)
        if self.is_ddp:
            losses_tr = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(losses_tr, outputs["loss"])
            loss_here = np.vstack(losses_tr).mean()
        else:
            loss_here = np.mean(outputs["loss"])

        self.logger.log("train_losses", loss_here, self.current_epoch)

        # domain losses
        keys = [x for x in outputs.keys() if x.startswith("loss_domain")]
        lossdomains_here = []
        if self.is_ddp:
            for key in keys:
                losses_tr_domain = [None for _ in range(dist.get_world_size())]
                dist.all_gather_object(losses_tr_domain, outputs[key])
                lossdomains_here.append(np.vstack(losses_tr_domain).mean())
        else:
            for key in keys:
                lossdomains_here.append(np.mean(outputs[key]))

        for i in range(len(self.network_domainpredictors)):
            self.logger.log(
                f"train_losses_domain_{i}", lossdomains_here[i], self.current_epoch
            )

        # domain accuracies
        keys = [x for x in outputs.keys() if x.startswith("acc_domain")]
        accdomains_here = []
        if self.is_ddp:
            for key in keys:
                acc_val_domain = [None for _ in range(dist.get_world_size())]
                dist.all_gather_object(acc_val_domain, outputs[key])
                accdomains_here.append(np.vstack(acc_val_domain).mean())
        else:
            for key in keys:
                accdomains_here.append(np.mean(outputs[key]))

        for i in range(len(self.network_domainpredictors)):
            self.logger.log(
                f"train_acc_domain_{i}", accdomains_here[i], self.current_epoch
            )

        # confusion losses
        keys = [x for x in outputs.keys() if x.startswith("loss_confusion")]
        lossconfusion_here = []
        if self.is_ddp:
            for key in keys:
                losses_confusion = [None for _ in range(dist.get_world_size())]
                dist.all_gather_object(losses_confusion, outputs[key])
                lossconfusion_here.append(np.vstack(losses_confusion).mean())
        else:
            for key in keys:
                lossconfusion_here.append(np.mean(outputs[key]))

        for i in range(len(self.network_domainpredictors)):
            self.logger.log(
                f"train_losses_confusion_{i}", lossconfusion_here[i], self.current_epoch
            )

    def on_validation_epoch_start(self):
        self.network.eval()
        for network in self.network_domainpredictors:
            network.eval()

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"]
        target = batch["target"]
        domain = batch["domain"]

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        domain = domain.to(self.device, non_blocking=True)

        l_domain = []
        acc_domain = []
        with autocast(
            self.device.type, enabled=True
        ) if self.device.type == "cuda" else dummy_context():
            output, encoder_outputs = self.network(data, return_skips=True)
            del data
            l = self.loss(output, target)
            for i in range(len(self.network_domainpredictors)):
                domain_output = self.network_domainpredictors[i](
                    encoder_outputs[-1 - i].detach()
                )
                l_domain.append(
                    self.loss_domain_predictors[i](domain_output, domain.flatten())
                )
                domain_pred = torch.argmax(domain_output, axis=1)
                domain_true = domain.flatten()
                domain_acc = (domain_pred == domain_true).sum() / len(domain)
                acc_domain.append(domain_acc)

        # we only need the output with the highest output resolution
        output = output[0]
        target = target[0]

        # the following is needed for online evaluation. Fake dice (green line)
        axes = [0] + list(range(2, len(output.shape)))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            # no need for softmax
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(
                output.shape, device=output.device, dtype=torch.float32
            )
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                # CAREFUL that you don't rely on target after this line!
                target[target == self.label_manager.ignore_label] = 0
            else:
                mask = 1 - target[:, -1:]
                # CAREFUL that you don't rely on target after this line!
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(
            predicted_segmentation_onehot, target, axes=axes, mask=mask
        )

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            # if we train with regions all segmentation heads predict some kind of foreground. In conventional
            # (softmax training) there needs tobe one output for the background. We are not interested in the
            # background Dice
            # [1:] in order to remove background
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        out = {
            "loss": l.detach().cpu().numpy(),
            "tp_hard": tp_hard,
            "fp_hard": fp_hard,
            "fn_hard": fn_hard,
        }
        out.update(
            {
                f"loss_domain_{enum}": x.detach().cpu().numpy()
                for enum, x in enumerate(l_domain)
            }
        )
        out.update(
            {
                f"acc_domain_{enum}": x.detach().cpu().numpy()
                for enum, x in enumerate(acc_domain)
            }
        )
        return out

    def on_validation_epoch_end(self, val_outputs: List[dict]):
        outputs_collated = collate_outputs(val_outputs)
        tp = np.sum(outputs_collated["tp_hard"], 0)
        fp = np.sum(outputs_collated["fp_hard"], 0)
        fn = np.sum(outputs_collated["fn_hard"], 0)

        if self.is_ddp:
            world_size = dist.get_world_size()

            tps = [None for _ in range(world_size)]
            dist.all_gather_object(tps, tp)
            tp = np.vstack([i[None] for i in tps]).sum(0)

            fps = [None for _ in range(world_size)]
            dist.all_gather_object(fps, fp)
            fp = np.vstack([i[None] for i in fps]).sum(0)

            fns = [None for _ in range(world_size)]
            dist.all_gather_object(fns, fn)
            fn = np.vstack([i[None] for i in fns]).sum(0)

            losses_val = [None for _ in range(world_size)]
            dist.all_gather_object(losses_val, outputs_collated["loss"])
            loss_here = np.vstack(losses_val).mean()
        else:
            loss_here = np.mean(outputs_collated["loss"])

        global_dc_per_class = [
            i for i in [2 * i / (2 * i + j + k) for i, j, k in zip(tp, fp, fn)]
        ]
        mean_fg_dice = np.nanmean(global_dc_per_class)
        self.logger.log("mean_fg_dice", mean_fg_dice, self.current_epoch)
        self.logger.log(
            "dice_per_class_or_region", global_dc_per_class, self.current_epoch
        )
        self.logger.log("val_losses", loss_here, self.current_epoch)

        # domain losses
        keys = [x for x in outputs_collated.keys() if x.startswith("loss_domain")]
        lossdomains_here = []
        if self.is_ddp:
            for key in keys:
                losses_val_domain = [None for _ in range(dist.get_world_size())]
                dist.all_gather_object(losses_val_domain, outputs_collated[key])
                lossdomains_here.append(np.vstack(losses_val_domain).mean())
        else:
            for key in keys:
                lossdomains_here.append(np.mean(outputs_collated[key]))

        for i in range(len(self.network_domainpredictors)):
            self.logger.log(
                f"val_losses_domain_{i}", lossdomains_here[i], self.current_epoch
            )

        # domain accuracies
        keys = [x for x in outputs_collated.keys() if x.startswith("acc_domain")]
        accdomains_here = []
        if self.is_ddp:
            for key in keys:
                acc_val_domain = [None for _ in range(dist.get_world_size())]
                dist.all_gather_object(acc_val_domain, outputs_collated[key])
                accdomains_here.append(np.vstack(acc_val_domain).mean())
        else:
            for key in keys:
                accdomains_here.append(np.mean(outputs_collated[key]))

        for i in range(len(self.network_domainpredictors)):
            self.logger.log(
                f"val_acc_domain_{i}", accdomains_here[i], self.current_epoch
            )

    def on_epoch_end(self):
        self.logger.log("epoch_end_timestamps", time(), self.current_epoch)

        # todo find a solution for this stupid shit
        self.print_to_log_file(
            "train_loss",
            np.round(self.logger.my_fantastic_logging["train_losses"][-1], decimals=4),
        )
        self.print_to_log_file(
            "val_loss",
            np.round(self.logger.my_fantastic_logging["val_losses"][-1], decimals=4),
        )
        for network in self.network_domainpredictors:
            self.print_to_log_file(
                f"train_loss_domain_{network.index}",
                np.round(
                    self.logger.my_fantastic_logging[
                        f"train_losses_domain_{network.index}"
                    ][-1],
                    decimals=4,
                ),
            )
            self.print_to_log_file(
                f"train_acc_domain_{network.index}",
                np.round(
                    self.logger.my_fantastic_logging[f"train_acc_domain_{network.index}"][
                        -1
                    ],
                    decimals=4,
                ),
            )
            self.print_to_log_file(
                f"validation_loss_domain_{network.index}",
                np.round(
                    self.logger.my_fantastic_logging[
                        f"val_losses_domain_{network.index}"
                    ][-1],
                    decimals=4,
                ),
            )
            self.print_to_log_file(
                f"validation_acc_domain_{network.index}",
                np.round(
                    self.logger.my_fantastic_logging[f"val_acc_domain_{network.index}"][
                        -1
                    ],
                    decimals=4,
                ),
            )
            if self.is_unlearn_step:
                self.print_to_log_file(
                    f"train_loss_confusion_{network.index}",
                    np.round(
                        self.logger.my_fantastic_logging[
                            f"train_losses_confusion_{network.index}"
                        ][-1],
                        decimals=4,
                    ),
                )
        self.print_to_log_file(
            "Pseudo dice",
            [
                np.round(i, decimals=4)
                for i in self.logger.my_fantastic_logging["dice_per_class_or_region"][
                    -1
                ]
            ],
        )
        self.print_to_log_file(
            f"Epoch time: {np.round(self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1], decimals=2)} s"
        )

        # handling periodic checkpointing
        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (
            self.num_epochs - 1
        ):
            self.save_checkpoint(join(self.output_folder, "checkpoint_latest.pth"))

        # handle 'best' checkpointing. ema_fg_dice is computed by the logger and can be accessed like this
        if (
            self._best_ema is None
            or self.logger.my_fantastic_logging["ema_fg_dice"][-1] > self._best_ema
        ):
            self._best_ema = self.logger.my_fantastic_logging["ema_fg_dice"][-1]
            self.print_to_log_file(
                f"Yayy! New best EMA pseudo Dice: {np.round(self._best_ema, decimals=4)}"
            )
            self.save_checkpoint(join(self.output_folder, "checkpoint_best.pth"))

        if self.local_rank == 0:
            self.logger.plot_progress_png(self.output_folder)

        self.current_epoch += 1

        # handle last checkpoint before unlearning
        if self.current_epoch == self.num_epochs_stage_0:
            self.save_checkpoint(join(self.output_folder, "checkpoint_before_unlearn.pth"))

    def run_training(self):
        self.on_train_start()

        for epoch in range(self.current_epoch, self.num_epochs):
            self.on_epoch_start()

            self.is_unlearn_step = self.current_epoch > self.num_epochs_stage_0

            self.on_train_epoch_start()
            train_outputs = []
            for batch_id in range(self.num_iterations_per_epoch):
                if not self.is_unlearn_step:
                    out = self.train_step(next(self.dataloader_train))
                else:
                    out = self.train_step_unlearn(next(self.dataloader_train))
                train_outputs.append(out)
            self.on_train_epoch_end(train_outputs)

            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []
                for batch_id in range(self.num_val_iterations_per_epoch):
                    val_outputs.append(self.validation_step(next(self.dataloader_val)))
                self.on_validation_epoch_end(val_outputs)

            self.on_epoch_end()

        self.on_train_end()

    def plot_network_architecture(self):
        super().plot_network_architecture()

        import hiddenlayer as hl

        for network in self.network_domainpredictors:
            g = hl.build_graph(
                network,
                torch.rand(
                    (1, network.fm_in, network.ks_in, network.ks_in, network.ks_in),
                    device=self.device,
                ),
                transforms=None,
            )
            g.save(
                join(
                    self.output_folder,
                    f"domain_predictor_architecture_{network.index}.pdf",
                )
            )

    def save_checkpoint(self, filename: str) -> None:
        # if unlearn step, then stage is 1 (not 0) so optimizers and networks are from stage 0 (by name)
        stage_1 = self.is_unlearn_step
        optimizer = self.optimizer_stage1_unet if stage_1 else self.optimizer_stage0_unet
        if self.local_rank == 0:
            if not self.disable_checkpointing:
                if self.is_ddp:
                    mod = self.network.module
                else:
                    mod = self.network
                if isinstance(mod, OptimizedModule):
                    mod = mod._orig_mod

                checkpoint = {
                    "network_weights": mod.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "grad_scaler_state": self.grad_scaler.state_dict()
                    if self.grad_scaler is not None
                    else None,
                    "logging": self.logger.get_checkpoint(),
                    "_best_ema": self._best_ema,
                    "current_epoch": self.current_epoch + 1,
                    "init_args": self.my_init_kwargs,
                    "trainer_name": self.__class__.__name__,
                    "inference_allowed_mirroring_axes": self.inference_allowed_mirroring_axes,
                    "stage": int(stage_1)
                }

                for i in range(len(self.network_domainpredictors)):
                    network_domainpredictor = self.network_domainpredictors[i]

                    if not stage_1:
                        optimizer = self.optimizers_stage0_domainpred[i]
                        grad_scaler = self.grad_scalers_stage0_domainpred[i]
                    else:
                        optimizer = self.optimizers_stage1_domainpred[i]
                        grad_scaler = self.grad_scalers_stage1_domainpred[i]

                    if self.is_ddp:
                        mod = network_domainpredictor.module
                    else:
                        mod = network_domainpredictor
                    if isinstance(mod, OptimizedModule):
                        mod = mod._orig_mod

                    checkpoint_domain = {
                        f"domainpred_network_weights_{i}": mod.state_dict(),
                        f"domainpred_optimizer_state_{i}": optimizer.state_dict(),
                        f"domainpred_grad_scaler_state_{i}": grad_scaler.state_dict() if grad_scaler is not None else None,
                    }

                    checkpoint.update(checkpoint_domain)

                torch.save(checkpoint, filename)
            else:
                self.print_to_log_file(
                    "No checkpoint written, checkpointing is disabled"
                )

    def load_checkpoint(self, filename_or_checkpoint: Union[dict, str]) -> None:
        if not self.was_initialized:
            self.initialize()

        if isinstance(filename_or_checkpoint, str):
            checkpoint = torch.load(filename_or_checkpoint, map_location=self.device)
        # if state dict comes from nn.DataParallel but we use non-parallel model here then the state dict keys do not
        # match. Use heuristic to make it match
        new_state_dict = {}
        for k, value in checkpoint['network_weights'].items():
            key = k
            if key not in self.network.state_dict().keys() and key.startswith('module.'):
                key = key[7:]
            new_state_dict[key] = value

        self.my_init_kwargs = checkpoint['init_args']
        self.current_epoch = checkpoint['current_epoch']
        self.logger.load_checkpoint(checkpoint['logging'])
        self._best_ema = checkpoint['_best_ema']
        self.inference_allowed_mirroring_axes = checkpoint[
            'inference_allowed_mirroring_axes'] if 'inference_allowed_mirroring_axes' in checkpoint.keys() else self.inference_allowed_mirroring_axes

        # messing with state dict naming schemes. Facepalm.
        if self.is_ddp:
            if isinstance(self.network.module, OptimizedModule):
                self.network.module._orig_mod.load_state_dict(new_state_dict)
            else:
                self.network.module.load_state_dict(new_state_dict)
        else:
            if isinstance(self.network, OptimizedModule):
                self.network._orig_mod.load_state_dict(new_state_dict)
            else:
                self.network.load_state_dict(new_state_dict)

        # different optimizer loading than original
        self.optimizer_stage0_unet.load_state_dict(checkpoint['optimizer_state'])
        self.optimizer_stage1_unet.load_state_dict(checkpoint['optimizer_state'])
        if self.grad_scaler is not None:
            if checkpoint['grad_scaler_state'] is not None:
                self.grad_scaler.load_state_dict(checkpoint['grad_scaler_state'])

        # domainpred
        domainpred_keys = [key for key in checkpoint if key.startswith("domainpred_network_weights_")]
        stage_1 = bool(checkpoint.get('stage', False))
        for i in range(len(domainpred_keys)):
            new_state_dict = {}
            for k, value in checkpoint[f'domainpred_network_weights_{i}'].items():
                key = k
                if key not in self.network_domainpredictors[i].state_dict().keys() and key.startswith('module.'):
                    key = key[7:]
                new_state_dict[key] = value

            if self.is_ddp:
                if isinstance(self.network_domainpredictors[i].module, OptimizedModule):
                    self.network_domainpredictors[i].module._orig_mod.load_state_dict(new_state_dict)
                else:
                    self.network_domainpredictors[i].module.load_state_dict(new_state_dict)
            else:
                if isinstance(self.network_domainpredictors[i], OptimizedModule):
                    self.network_domainpredictors[i]._orig_mod.load_state_dict(new_state_dict)
                else:
                    self.network_domainpredictors[i].load_state_dict(new_state_dict)

            if not stage_1:
                optimizer = self.optimizers_stage0_domainpred[i]
                grad_scaler = self.grad_scalers_stage0_domainpred[i]
            else:
                optimizer = self.optimizers_stage1_domainpred[i]
                grad_scaler = self.grad_scalers_stage1_domainpred[i]

            optimizer_state = checkpoint.get(f'domainpred_optimizer_state_{i}', None)
            optimizer.load_state_dict(optimizer_state)

            if grad_scaler is not None:
                grad_scaler_state = checkpoint.get(f'domainpred_grad_scaler_state_{i}', None)
                if grad_scaler_state is not None:
                    grad_scaler.load_state_dict(grad_scaler_state)
