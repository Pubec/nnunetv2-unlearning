from nnunetv2.training.nnUNetTrainer.variants.unlearning.models import RefConfusionLoss
from nnunetv2.training.nnUNetTrainer.variants.unlearning.Unlearn.nnUNetTrainer_UnlearnPeriodicPerBatch import nnUNetTrainer_UnlearnPeriodicPerBatch


class nnUNetTrainer_UnlearnPeriodicPerBatchRefConfusionLoss(nnUNetTrainer_UnlearnPeriodicPerBatch):

    def _build_unlearn_loss(self):
        return [RefConfusionLoss().to(self.device) for _ in self.network_domainpredictors]
