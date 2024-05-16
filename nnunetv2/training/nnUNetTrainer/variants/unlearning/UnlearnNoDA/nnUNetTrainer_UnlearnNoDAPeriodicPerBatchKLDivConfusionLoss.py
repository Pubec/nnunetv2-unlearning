from nnunetv2.training.nnUNetTrainer.variants.unlearning.models import KLDivergenceConfussionLoss
from nnunetv2.training.nnUNetTrainer.variants.unlearning.UnlearnNoDA.nnUNetTrainer_UnlearnNoDAPeriodicPerBatch import nnUNetTrainer_UnlearnNoDAPeriodicPerBatch


class nnUNetTrainer_UnlearnNoDAPeriodicPerBatchKLDivConfusionLoss(nnUNetTrainer_UnlearnNoDAPeriodicPerBatch):

    def _build_unlearn_loss(self):
        return [KLDivergenceConfussionLoss(domains=self.domains, device=self.device).to(self.device) for _ in self.network_domainpredictors]
