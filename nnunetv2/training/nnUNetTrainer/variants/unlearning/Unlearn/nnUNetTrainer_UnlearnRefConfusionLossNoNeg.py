from nnunetv2.training.nnUNetTrainer.variants.unlearning.models import RefConfusionLossNoNeg
from nnunetv2.training.nnUNetTrainer.variants.unlearning.nnUNetTrainer_UnlearnBase import nnUNetTrainer_UnlearnBase


class nnUNetTrainer_UnlearnRefConfusionLossNoNeg(nnUNetTrainer_UnlearnBase):

    def _build_unlearn_loss(self):
        return [RefConfusionLossNoNeg().to(self.device) for _ in self.network_domainpredictors]
