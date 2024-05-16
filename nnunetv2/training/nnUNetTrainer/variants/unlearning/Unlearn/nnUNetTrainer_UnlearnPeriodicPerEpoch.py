import torch

from nnunetv2.training.nnUNetTrainer.variants.unlearning.nnUNetTrainer_UnlearnBase import nnUNetTrainer_UnlearnBase


class nnUNetTrainer_UnlearnPeriodicPerEpoch(nnUNetTrainer_UnlearnBase):

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

        self.unlearn_phase_number_of_network_bp = self.configuration_manager.configuration.get('unlearn_phase_number_of_network_bp', 5)
        self.unlearn_phase_number_of_unlearn_bp = self.configuration_manager.configuration.get('unlearn_phase_number_of_unlearn_bp', 5)
        self.phase_current = 0

    def update_training_phase(self):
        if not self.is_unlearn_step:
            self.unlearn_step_network_bp = True
            self.unlearn_step_classifier_bp = True
            self.unlearn_step_unlearn_bp = False
        else:
            if self.phase_current <= self.unlearn_phase_number_of_network_bp:
                self.unlearn_step_network_bp = True
                self.unlearn_step_unlearn_bp = False
            else:
                self.unlearn_step_network_bp = False
                self.unlearn_step_unlearn_bp = True
            self.phase_current += 1

            if self.phase_current > (self.unlearn_phase_number_of_network_bp + self.unlearn_phase_number_of_unlearn_bp):
                self.phase_current = 0

    def print_unlearning_parameters(self):
        super().print_unlearning_parameters()
        self.print_to_log_file(f"{self.unlearn_phase_number_of_network_bp = }")
        self.print_to_log_file(f"{self.unlearn_phase_number_of_unlearn_bp = }")

    def run_training(self):
        self.on_train_start()

        for epoch in range(self.current_epoch, self.num_epochs):
            self.on_epoch_start()

            self.is_unlearn_step = self.current_epoch > self.num_epochs_stage_0

            self.on_train_epoch_start()
            self.update_training_phase()

            train_outputs = []
            for batch_id in range(self.num_iterations_per_epoch):

                # self.print_to_log_file(f"[{epoch:04d}|{batch_id:04d}] unlearn_step_network_bp: {self.unlearn_step_network_bp}")
                # self.print_to_log_file(f"[{epoch:04d}|{batch_id:04d}] unlearn_step_unlearn_bp: {self.unlearn_step_unlearn_bp}")

                if not self.is_unlearn_step:
                    out = self.train_step(next(self.dataloader_train))
                else:
                    out = self.train_step_unlearn(next(self.dataloader_train))

                train_outputs.append(out)
            self.on_train_epoch_end(train_outputs)

            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []
                for _ in range(self.num_val_iterations_per_epoch):
                    val_outputs.append(self.validation_step(next(self.dataloader_val)))
                self.on_validation_epoch_end(val_outputs)

            self.on_epoch_end()

        self.on_train_end()
