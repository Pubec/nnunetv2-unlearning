from nnunetv2.training.nnUNetTrainer.variants.unlearning.models import RefConfusionLoss
from nnunetv2.training.nnUNetTrainer.variants.unlearning.nnUNetTrainer_UnlearnBase import nnUNetTrainer_UnlearnBase


class nnUNetTrainer_UnlearnRefConfusionLoss(nnUNetTrainer_UnlearnBase):

    def _build_unlearn_loss(self):
        return [RefConfusionLoss().to(self.device) for _ in self.network_domainpredictors]
