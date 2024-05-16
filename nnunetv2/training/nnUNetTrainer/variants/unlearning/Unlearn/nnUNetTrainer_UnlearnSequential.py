import torch
from nnunetv2.training.nnUNetTrainer.variants.unlearning.Unlearn.nnUNetTrainer_UnlearnPeriodicPerBatch import nnUNetTrainer_UnlearnPeriodicPerBatch

class nnUNetTrainer_UnlearnSequential(nnUNetTrainer_UnlearnPeriodicPerBatch):
    """This is just a UnlearnPeriodicPerBatch where count of epochs is 1"""

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

        self.unlearn_phase_number_of_network_bp = 1
        self.unlearn_phase_number_of_unlearn_bp = 1
        self.phase_current = 0

