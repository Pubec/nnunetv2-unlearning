from nnunetv2.training.nnUNetTrainer.variants.unlearning.models import RefConfusionLossEpsilon
from nnunetv2.training.nnUNetTrainer.variants.unlearning.nnUNetTrainer_UnlearnBase import nnUNetTrainer_UnlearnBase


class nnUNetTrainer_UnlearnRefConfusionLossEpsilon(nnUNetTrainer_UnlearnBase):

    def _build_unlearn_loss(self):
        return [RefConfusionLossEpsilon().to(self.device) for _ in self.network_domainpredictors]
