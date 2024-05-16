from nnunetv2.training.nnUNetTrainer.variants.unlearning.models import KLDivergenceConfussionLoss
from nnunetv2.training.nnUNetTrainer.variants.unlearning.Unlearn.nnUNetTrainer_UnlearnPeriodicPerBatch import nnUNetTrainer_UnlearnPeriodicPerBatch


class nnUNetTrainer_UnlearnPeriodicPerBatchKLDivConfusionLoss(nnUNetTrainer_UnlearnPeriodicPerBatch):

    def _build_unlearn_loss(self):
        return [KLDivergenceConfussionLoss(domains=self.domains, device=self.device).to(self.device) for _ in self.network_domainpredictors]
