import torch
from nnunetv2.training.nnUNetTrainer.variants.unlearning.nnUNetTrainer_UnlearnBase import (
    nnUNetTrainer_UnlearnBase,
)


class nnUNetTrainer_Unlearn(nnUNetTrainer_UnlearnBase):

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        experiment_identifier: str = "",
        unpack_dataset: bool = True,
        device: torch.device = ...,
        dir_can_exist: bool = False,
    ):
        super().__init__(
            plans,
            configuration,
            fold,
            dataset_json,
            experiment_identifier,
            unpack_dataset,
            device,
            dir_can_exist,
        )
