from collections import Counter
import random
from typing import Tuple
from batchgenerators.utilities.file_and_folder_operations import List
import numpy as np
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3D
from nnunetv2.utilities.label_handling.label_handling import LabelManager


class nnUNetDataLoader3DUnlearn(nnUNetDataLoader3D):
    def __init__(
        self,
        data: nnUNetDataset,
        batch_size: int,
        patch_size: List[int] | Tuple[int, ...] | np.ndarray,
        final_patch_size: List[int] | Tuple[int, ...] | np.ndarray,
        label_manager: LabelManager,
        oversample_foreground_percent: float = 0,
        sampling_probabilities: List[int] | Tuple[int, ...] | np.ndarray = None,
        pad_sides: List[int] | Tuple[int, ...] | np.ndarray = None,
        probabilistic_oversampling: bool = False,
        domain_constraint: bool = False,
        domain_count: int = None
    ):
        super().__init__(
            data,
            batch_size,
            patch_size,
            final_patch_size,
            label_manager,
            oversample_foreground_percent,
            sampling_probabilities,
            pad_sides,
            probabilistic_oversampling,
        )
        self.domain_constraint = domain_constraint
        self.domain_count = domain_count

        if self.domain_constraint:
            self.domain_list = [data[key]['properties']['domain'] for key in data.keys()]

        print(f"Loader found: {Counter(self.domain_list) = }")
        if (len(set(self.domain_list))) != self.domain_count:
            raise ValueError("Domain count and unique domains in dataset do not match")


    def get_indices(self):
        if self.domain_constraint:
            # create a list of indices for each domain
            domain_indices = [
                [i for i, domain in enumerate(self.domain_list) if domain == d]
                for d in range(self.domain_count)
            ]

            # calculate how many samples to take from each domain
            n_samples_per_domain = self.batch_size // self.domain_count

            # calculate how many additional samples to take due to the batch size not being evenly divisible by the domain count
            remainder = self.batch_size % self.domain_count

            # select the required number of random indices from each domain list
            idx = []
            for i, indices in enumerate(domain_indices):
                n_samples = n_samples_per_domain + (i < remainder)  # add one extra sample for the domains in the remainder
                idx.extend(np.random.choice(indices, n_samples))

            # return the indices
            random.shuffle(idx)
            return [self.indices[i] for i in idx]

        if self.infinite:
            return np.random.choice(
                self.indices,
                self.batch_size,
                replace=True,
                p=self.sampling_probabilities,
            )

        if self.last_reached:
            self.reset()
            raise StopIteration

        if not self.was_initialized:
            self.reset()

        indices = []

        for b in range(self.batch_size):
            if self.current_position < len(self.indices):
                indices.append(self.indices[self.current_position])

                self.current_position += 1
            else:
                self.last_reached = True
                break

        if len(indices) > 0 and ((not self.last_reached) or self.return_incomplete):
            self.current_position += (
                self.number_of_threads_in_multithreaded - 1
            ) * self.batch_size
            return indices
        else:
            self.reset()
            raise StopIteration
