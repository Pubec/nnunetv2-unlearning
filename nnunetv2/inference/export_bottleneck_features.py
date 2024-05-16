import inspect
import multiprocessing
import os
from pathlib import Path
import traceback
from copy import deepcopy
from time import sleep
from typing import Tuple, Union, List

import numpy as np
import torch
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import load_json, join, isfile, maybe_mkdir_p, isdir, subdirs, \
    save_json
from torch._dynamo import OptimizedModule
from tqdm import tqdm

from nnunetv2.configuration import default_num_processes
from nnunetv2.inference.data_iterators import preprocessing_iterator_fromfiles
from nnunetv2.inference.sliding_window_prediction import compute_gaussian, \
    compute_steps_for_sliding_window
from nnunetv2.utilities.file_path_utilities import get_output_folder, check_workers_alive_and_busy
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.utilities.json_export import recursive_fix_for_json_export
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.utilities.utils import create_lists_from_splitted_dataset_folder
from development.nnunetv2.source.nnunetv2.training.nnUNetTrainer.variants.unlearning.nnUNetTrainer_UnlearnBase import nnUNetTrainer_UnlearnBase



class nnUNetPredictor(object):
    def __init__(self,
                 tile_step_size: float = 0.5,
                 use_gaussian: bool = True,
                 use_mirroring: bool = True,
                 perform_everything_on_gpu: bool = True,
                 device: torch.device = torch.device('cuda'),
                 verbose: bool = False,
                 verbose_preprocessing: bool = False,
                 allow_tqdm: bool = True):
        self.verbose = verbose
        self.verbose_preprocessing = verbose_preprocessing
        self.allow_tqdm = allow_tqdm

        self.plans_manager, self.configuration_manager, self.list_of_parameters, self.network, self.dataset_json, \
        self.trainer_name, self.allowed_mirroring_axes, self.label_manager = None, None, None, None, None, None, None, None

        self.tile_step_size = tile_step_size
        self.use_gaussian = use_gaussian
        self.use_mirroring = use_mirroring
        if device.type == 'cuda':
            # device = torch.device(type='cuda', index=0)  # set the desired GPU with CUDA_VISIBLE_DEVICES!
            # why would I ever want to do that. Stupid dobby. This kills DDP inference...
            pass
        if device.type != 'cuda':
            print(f'perform_everything_on_gpu=True is only supported for cuda devices! Setting this to False')
            perform_everything_on_gpu = False
        self.device = device
        self.perform_everything_on_gpu = perform_everything_on_gpu

    def initialize_from_trained_model_folder(self, model_training_output_dir: str,
                                             use_folds: Union[Tuple[Union[int, str]], None],
                                             checkpoint_name: str = 'checkpoint_final.pth',
                                             domainpredictor_type: str = "single"):
        """
        This is used when making predictions with a trained model
        """
        if use_folds is None:
            use_folds = nnUNetPredictor.auto_detect_available_folds(model_training_output_dir, checkpoint_name)

        dataset_json = load_json(join(model_training_output_dir, 'dataset.json'))
        plans = load_json(join(model_training_output_dir, 'plans.json'))
        plans_manager = PlansManager(plans)

        if isinstance(use_folds, str):
            use_folds = [use_folds]

        parameters = []
        for i, f in enumerate(use_folds):
            f = int(f) if f != 'all' else f
            checkpoint = torch.load(join(model_training_output_dir, f'fold_{f}', checkpoint_name),
                                    map_location=torch.device('cpu'))
            if i == 0:
                trainer_name = checkpoint['trainer_name']
                configuration_name = checkpoint['init_args']['configuration']
                inference_allowed_mirroring_axes = checkpoint['inference_allowed_mirroring_axes'] if \
                    'inference_allowed_mirroring_axes' in checkpoint.keys() else None

            parameters.append(checkpoint['network_weights'])

        configuration_manager = plans_manager.get_configuration(configuration_name)
        # restore network
        num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
        network = nnUNetTrainer_UnlearnBase.build_network_architecture(plans_manager, dataset_json, configuration_manager,
                                                           num_input_channels, enable_deep_supervision=False)
        self.plans_manager = plans_manager
        self.configuration_manager = configuration_manager
        self.list_of_parameters = parameters
        self.network = network
        self.dataset_json = dataset_json
        self.trainer_name = trainer_name
        self.allowed_mirroring_axes = inference_allowed_mirroring_axes
        self.label_manager = plans_manager.get_label_manager(dataset_json)

        if ('nnUNet_compile' in os.environ.keys()) and (os.environ['nnUNet_compile'].lower() in ('true', '1', 't')) \
                and not isinstance(self.network, OptimizedModule):
            print('compiling network')
            self.network = torch.compile(self.network)

        """
        Domain prediction initialization
        """
        # self.network_domainpredictors = []
        # domainpred_stages = None
        # if domainpredictor_type == "single":
        #     domainpred_stages = nnUNetTrainer_UnlearnSingle.domainpred_stages
        # elif domainpredictor_type == "multi":
        #     domainpred_stages == nnUNetTrainer_Unlearn.domainpred_stages
        # else:
        #     raise ValueError(f"Domain predictor type not valid. Expected single/multi, got {domainpredictor_type}")

        # index = 0
        # for kernel_size, feature_size in domainpred_stages:
        #     domain_predictor = DomainPred_ConvReluNormLinear(
        #         feature_size, kernel_size, self.domains, index
        #     ).to(self.device)
        #     self.network_domainpredictors.append(domain_predictor)
        #     index += 1

        # if ('nnUNet_compile' in os.environ.keys()) and (os.environ['nnUNet_compile'].lower() in ('true', '1', 't')) \
        #         and not isinstance(self.network, OptimizedModule):
        #     for i in range(self.network_domainpredictors):
        #         print(f'compiling {i} domain predictor network')
        #         self.network_domainpredictors[i] = torch.compile(self.network_domainpredictors[i])

    @staticmethod
    def auto_detect_available_folds(model_training_output_dir, checkpoint_name):
        print('use_folds is None, attempting to auto detect available folds')
        fold_folders = subdirs(model_training_output_dir, prefix='fold_', join=False)
        fold_folders = [i for i in fold_folders if i != 'fold_all']
        fold_folders = [i for i in fold_folders if isfile(join(model_training_output_dir, i, checkpoint_name))]
        use_folds = [int(i.split('_')[-1]) for i in fold_folders]
        print(f'found the following folds: {use_folds}')
        return use_folds

    def _manage_input_and_output_lists(self, list_of_lists_or_source_folder: Union[str, List[List[str]]],
                                       output_folder_or_list_of_truncated_output_files: Union[None, str, List[str]],
                                       folder_with_segs_from_prev_stage: str = None,
                                       overwrite: bool = True,
                                       part_id: int = 0,
                                       num_parts: int = 1,
                                       save_probabilities: bool = False):
        if isinstance(list_of_lists_or_source_folder, str):
            list_of_lists_or_source_folder = create_lists_from_splitted_dataset_folder(list_of_lists_or_source_folder,
                                                                                       self.dataset_json['file_ending'])
        print(f'There are {len(list_of_lists_or_source_folder)} cases in the source folder')
        list_of_lists_or_source_folder = list_of_lists_or_source_folder[part_id::num_parts]
        caseids = [os.path.basename(i[0])[:-(len(self.dataset_json['file_ending']) + 5)] for i in
                   list_of_lists_or_source_folder]
        print(
            f'I am process {part_id} out of {num_parts} (max process ID is {num_parts - 1}, we start counting with 0!)')
        print(f'There are {len(caseids)} cases that I would like to predict')

        if isinstance(output_folder_or_list_of_truncated_output_files, str):
            output_filename_truncated = [join(output_folder_or_list_of_truncated_output_files, i) for i in caseids]
        else:
            output_filename_truncated = output_folder_or_list_of_truncated_output_files

        seg_from_prev_stage_files = [join(folder_with_segs_from_prev_stage, i + self.dataset_json['file_ending']) if
                                     folder_with_segs_from_prev_stage is not None else None for i in caseids]
        # remove already predicted files form the lists
        if not overwrite and output_filename_truncated is not None:
            tmp = [isfile(i + self.dataset_json['file_ending']) for i in output_filename_truncated]
            if save_probabilities:
                tmp2 = [isfile(i + '.npz') for i in output_filename_truncated]
                tmp = [i and j for i, j in zip(tmp, tmp2)]
            not_existing_indices = [i for i, j in enumerate(tmp) if not j]

            output_filename_truncated = [output_filename_truncated[i] for i in not_existing_indices]
            list_of_lists_or_source_folder = [list_of_lists_or_source_folder[i] for i in not_existing_indices]
            seg_from_prev_stage_files = [seg_from_prev_stage_files[i] for i in not_existing_indices]
            print(f'overwrite was set to {overwrite}, so I am only working on cases that haven\'t been predicted yet. '
                  f'That\'s {len(not_existing_indices)} cases.')
        return list_of_lists_or_source_folder, output_filename_truncated, seg_from_prev_stage_files

    def export_bn_features_from_files(self,
                           list_of_lists_or_source_folder: Union[str, List[List[str]]],
                           output_folder_or_list_of_truncated_output_files: Union[str, None, List[str]],
                           save_probabilities: bool = False,
                           overwrite: bool = True,
                           num_processes_preprocessing: int = default_num_processes,
                           num_processes_segmentation_export: int = default_num_processes,
                           folder_with_segs_from_prev_stage: str = None,
                           num_parts: int = 1,
                           part_id: int = 0):
        """
        This is nnU-Net's default function for making predictions. It works best for batch predictions
        (predicting many images at once).
        """
        if isinstance(output_folder_or_list_of_truncated_output_files, str):
            output_folder = output_folder_or_list_of_truncated_output_files
        elif isinstance(output_folder_or_list_of_truncated_output_files, list):
            output_folder = os.path.dirname(output_folder_or_list_of_truncated_output_files[0])
        else:
            output_folder = None

        ########################
        # let's store the input arguments so that its clear what was used to generate the prediction
        if output_folder is not None:
            my_init_kwargs = {}
            for k in inspect.signature(self.export_bn_features_from_files).parameters.keys():
                my_init_kwargs[k] = locals()[k]
            my_init_kwargs = deepcopy(
                my_init_kwargs)  # let's not unintentionally change anything in-place. Take this as a
            recursive_fix_for_json_export(my_init_kwargs)
            maybe_mkdir_p(output_folder)
            save_json(my_init_kwargs, join(output_folder, 'predict_from_raw_data_args.json'))

            # we need these two if we want to do things with the predictions like for example apply postprocessing
            save_json(self.dataset_json, join(output_folder, 'dataset.json'), sort_keys=False)
            save_json(self.plans_manager.plans, join(output_folder, 'plans.json'), sort_keys=False)
        #######################

        # check if we need a prediction from the previous stage
        if self.configuration_manager.previous_stage_name is not None:
            assert folder_with_segs_from_prev_stage is not None, \
                f'The requested configuration is a cascaded network. It requires the segmentations of the previous ' \
                f'stage ({self.configuration_manager.previous_stage_name}) as input. Please provide the folder where' \
                f' they are located via folder_with_segs_from_prev_stage'

        # sort out input and output filenames
        list_of_lists_or_source_folder, output_filename_truncated, seg_from_prev_stage_files = \
            self._manage_input_and_output_lists(list_of_lists_or_source_folder,
                                                output_folder_or_list_of_truncated_output_files,
                                                folder_with_segs_from_prev_stage, overwrite, part_id, num_parts,
                                                save_probabilities)
        if len(list_of_lists_or_source_folder) == 0:
            return

        data_iterator = self._internal_get_data_iterator_from_lists_of_filenames(list_of_lists_or_source_folder,
                                                                                 seg_from_prev_stage_files,
                                                                                 output_filename_truncated,
                                                                                 num_processes_preprocessing)

        return self.predict_from_data_iterator(data_iterator, save_probabilities, num_processes_segmentation_export)

    def _internal_get_data_iterator_from_lists_of_filenames(self,
                                                            input_list_of_lists: List[List[str]],
                                                            seg_from_prev_stage_files: Union[List[str], None],
                                                            output_filenames_truncated: Union[List[str], None],
                                                            num_processes: int):
        return preprocessing_iterator_fromfiles(input_list_of_lists, seg_from_prev_stage_files,
                                                output_filenames_truncated, self.plans_manager, self.dataset_json,
                                                self.configuration_manager, num_processes, self.device.type == 'cuda',
                                                self.verbose_preprocessing)

    def predict_from_data_iterator(self,
                                   data_iterator,
                                   save_probabilities: bool = False,
                                   num_processes_segmentation_export: int = default_num_processes):
        """
        each element returned by data_iterator must be a dict with 'data', 'ofile' and 'data_properites' keys!
        If 'ofile' is None, the result will be returned instead of written to a file
        """
        with multiprocessing.get_context("spawn").Pool(num_processes_segmentation_export) as export_pool:
            worker_list = [i for i in export_pool._pool]
            r = []
            for preprocessed in data_iterator:
                data = preprocessed['data']
                if isinstance(data, str):
                    delfile = data
                    data = torch.from_numpy(np.load(data))
                    os.remove(delfile)

                ofile = preprocessed['ofile']
                ofile_bn_root = Path(ofile).parent / "bn_features"
                ofile_bn_root.mkdir(exist_ok=True)
                ofile_bn = ofile_bn_root / os.path.basename(ofile)
                if ofile is not None:
                    print(f'\nPredicting {os.path.basename(ofile)}:')
                else:
                    print(f'\nPredicting image of shape {data.shape}:')

                print(f'perform_everything_on_gpu: {self.perform_everything_on_gpu}')

                properties = preprocessed['data_properites']

                # let's not get into a runaway situation where the GPU predicts so fast that the disk has to b swamped with
                # npy files
                proceed = not check_workers_alive_and_busy(export_pool, worker_list, r, allowed_num_queued=2)
                while not proceed:
                    # print('sleeping')
                    sleep(0.1)
                    proceed = not check_workers_alive_and_busy(export_pool, worker_list, r, allowed_num_queued=2)

                prediction, encoder_outputs  = self.predict_logits_from_preprocessed_data(data)
                prediction = prediction.cpu()

                # saving encoder outputs as numpy arrays
                for out_enum, inner_list in enumerate(encoder_outputs):
                    for tens_enum, tens in enumerate(inner_list):
                        filename = ofile_bn.as_posix() + f"_slicer_{out_enum}_stage_{tens_enum}.npy"
                        np.save(filename, tens.cpu().numpy())
                    print(f"Saved featuremaps {out_enum:02d}")

                print(f'done with {os.path.basename(ofile)}')

                """
                This is how Domain Prediction should look like
                """
                # for i in range(len(self.network_domainpredictors)):
                #     domain_output = self.network_domainpredictors[i](
                #         encoder_outputs[-1 - i].detach()
                #     )
                #     domain_loss = self.loss_domain_predictors[i](
                #         domain_output, domain.flatten()
                #     )
                #     l_domain.append(domain_loss)

        if isinstance(data_iterator, MultiThreadedAugmenter):
            data_iterator._finish()

        # clear lru cache
        compute_gaussian.cache_clear()
        # clear device cache
        empty_cache(self.device)
        return True

    def predict_logits_from_preprocessed_data(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        IMPORTANT! IF YOU ARE RUNNING THE CASCADE, THE SEGMENTATION FROM THE PREVIOUS STAGE MUST ALREADY BE STACKED ON
        TOP OF THE IMAGE AS ONE-HOT REPRESENTATION! SEE PreprocessAdapter ON HOW THIS SHOULD BE DONE!

        RETURNED LOGITS HAVE THE SHAPE OF THE INPUT. THEY MUST BE CONVERTED BACK TO THE ORIGINAL IMAGE SIZE.
        SEE convert_predicted_logits_to_segmentation_with_correct_shape
        """
        # we have some code duplication here but this allows us to run with perform_everything_on_gpu=True as
        # default and not have the entire program crash in case of GPU out of memory. Neat. That should make
        # things a lot faster for some datasets.
        original_perform_everything_on_gpu = self.perform_everything_on_gpu
        with torch.no_grad():
            prediction = None
            if self.perform_everything_on_gpu:
                try:
                    for params in self.list_of_parameters:

                        # messing with state dict names...
                        if not isinstance(self.network, OptimizedModule):
                            self.network.load_state_dict(params)
                        else:
                            self.network._orig_mod.load_state_dict(params)

                        if prediction is None:
                            prediction, encoder_outputs = self.predict_sliding_window_return_logits(data)
                        else:
                            prediction_temp, encoder_outputs = self.predict_sliding_window_return_logits(data)
                            prediction += prediction_temp

                    if len(self.list_of_parameters) > 1:
                        prediction /= len(self.list_of_parameters)

                except RuntimeError:
                    print('Prediction with perform_everything_on_gpu=True failed due to insufficient GPU memory. '
                          'Falling back to perform_everything_on_gpu=False. Not a big deal, just slower...')
                    print('Error:')
                    traceback.print_exc()
                    prediction = None
                    self.perform_everything_on_gpu = False

            if prediction is None:
                for params in self.list_of_parameters:
                    # messing with state dict names...
                    if not isinstance(self.network, OptimizedModule):
                        self.network.load_state_dict(params)
                    else:
                        self.network._orig_mod.load_state_dict(params)

                    if prediction is None:
                        prediction, encoder_outputs = self.predict_sliding_window_return_logits(data)
                    else:
                        prediction_temp, encoder_outputs = self.predict_sliding_window_return_logits(data)
                        prediction += prediction_temp

                if len(self.list_of_parameters) > 1:
                    prediction /= len(self.list_of_parameters)

            print('Prediction done, transferring to CPU if needed')
            prediction = prediction.to('cpu')
            self.perform_everything_on_gpu = original_perform_everything_on_gpu
        return prediction, encoder_outputs

    def _internal_get_sliding_window_slicers(self, image_size: Tuple[int, ...]):
        slicers = []
        if len(self.configuration_manager.patch_size) < len(image_size):
            assert len(self.configuration_manager.patch_size) == len(
                image_size) - 1, 'if tile_size has less entries than image_size, ' \
                                 'len(tile_size) ' \
                                 'must be one shorter than len(image_size) ' \
                                 '(only dimension ' \
                                 'discrepancy of 1 allowed).'
            steps = compute_steps_for_sliding_window(image_size[1:], self.configuration_manager.patch_size,
                                                     self.tile_step_size)
            if self.verbose: print(f'n_steps {image_size[0] * len(steps[0]) * len(steps[1])}, image size is'
                                   f' {image_size}, tile_size {self.configuration_manager.patch_size}, '
                                   f'tile_step_size {self.tile_step_size}\nsteps:\n{steps}')
            for d in range(image_size[0]):
                for sx in steps[0]:
                    for sy in steps[1]:
                        slicers.append(
                            tuple([slice(None), d, *[slice(si, si + ti) for si, ti in
                                                     zip((sx, sy), self.configuration_manager.patch_size)]]))
        else:
            steps = compute_steps_for_sliding_window(image_size, self.configuration_manager.patch_size,
                                                     self.tile_step_size)
            if self.verbose: print(
                f'n_steps {np.prod([len(i) for i in steps])}, image size is {image_size}, tile_size {self.configuration_manager.patch_size}, '
                f'tile_step_size {self.tile_step_size}\nsteps:\n{steps}')
            for sx in steps[0]:
                for sy in steps[1]:
                    for sz in steps[2]:
                        slicers.append(
                            tuple([slice(None), *[slice(si, si + ti) for si, ti in
                                                  zip((sx, sy, sz), self.configuration_manager.patch_size)]]))
        return slicers

    def _internal_maybe_mirror_and_predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mirror_axes = self.allowed_mirroring_axes if self.use_mirroring else None
        prediction, encoder_outputs = self.network(x, return_skips=True)

        if mirror_axes is not None:
            # check for invalid numbers in mirror_axes
            # x should be 5d for 3d images and 4d for 2d. so the max value of mirror_axes cannot exceed len(x.shape) - 3
            assert max(mirror_axes) <= len(x.shape) - 3, 'mirror_axes does not match the dimension of the input!'

            num_predictons = 2 ** len(mirror_axes)
            if 0 in mirror_axes:
                prediction += torch.flip(self.network(torch.flip(x, (2,))), (2,))
            if 1 in mirror_axes:
                prediction += torch.flip(self.network(torch.flip(x, (3,))), (3,))
            if 2 in mirror_axes:
                prediction += torch.flip(self.network(torch.flip(x, (4,))), (4,))
            if 0 in mirror_axes and 1 in mirror_axes:
                prediction += torch.flip(self.network(torch.flip(x, (2, 3))), (2, 3))
            if 0 in mirror_axes and 2 in mirror_axes:
                prediction += torch.flip(self.network(torch.flip(x, (2, 4))), (2, 4))
            if 1 in mirror_axes and 2 in mirror_axes:
                prediction += torch.flip(self.network(torch.flip(x, (3, 4))), (3, 4))
            if 0 in mirror_axes and 1 in mirror_axes and 2 in mirror_axes:
                prediction += torch.flip(self.network(torch.flip(x, (2, 3, 4))), (2, 3, 4))
            prediction /= num_predictons
        return prediction, encoder_outputs

    def predict_sliding_window_return_logits(self, input_image: torch.Tensor) \
            -> Union[np.ndarray, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        assert isinstance(input_image, torch.Tensor)
        self.network = self.network.to(self.device)
        self.network.eval()
        encoder_outputs = []

        empty_cache(self.device)

        # Autocast is a little bitch.
        # If the device_type is 'cpu' then it's slow as heck on some CPUs (no auto bfloat16 support detection)
        # and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False
        # is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with torch.no_grad():
            with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
                assert len(input_image.shape) == 4, 'input_image must be a 4D np.ndarray or torch.Tensor (c, x, y, z)'

                if self.verbose: print(f'Input shape: {input_image.shape}')
                if self.verbose: print("step_size:", self.tile_step_size)
                if self.verbose: print("mirror_axes:", self.allowed_mirroring_axes if self.use_mirroring else None)

                # if input_image is smaller than tile_size we need to pad it to tile_size.
                data, slicer_revert_padding = pad_nd_image(input_image, self.configuration_manager.patch_size,
                                                           'constant', {'value': 0}, True,
                                                           None)

                slicers = self._internal_get_sliding_window_slicers(data.shape[1:])

                # preallocate results and num_predictions
                results_device = self.device if self.perform_everything_on_gpu else torch.device('cpu')
                if self.verbose: print('preallocating arrays')
                try:
                    data = data.to(self.device)
                    predicted_logits = torch.zeros((self.label_manager.num_segmentation_heads, *data.shape[1:]),
                                                   dtype=torch.half,
                                                   device=results_device)
                    n_predictions = torch.zeros(data.shape[1:], dtype=torch.half,
                                                device=results_device)
                    if self.use_gaussian:
                        gaussian = compute_gaussian(tuple(self.configuration_manager.patch_size), sigma_scale=1. / 8,
                                                    value_scaling_factor=1000,
                                                    device=results_device)
                except RuntimeError:
                    # sometimes the stuff is too large for GPUs. In that case fall back to CPU
                    results_device = torch.device('cpu')
                    data = data.to(results_device)
                    predicted_logits = torch.zeros((self.label_manager.num_segmentation_heads, *data.shape[1:]),
                                                   dtype=torch.half,
                                                   device=results_device)
                    n_predictions = torch.zeros(data.shape[1:], dtype=torch.half,
                                                device=results_device)
                    if self.use_gaussian:
                        gaussian = compute_gaussian(tuple(self.configuration_manager.patch_size), sigma_scale=1. / 8,
                                                    value_scaling_factor=1000,
                                                    device=results_device)
                finally:
                    empty_cache(self.device)

                if self.verbose: print('running prediction')
                for sl in tqdm(slicers, disable=not self.allow_tqdm):
                    workon = data[sl][None]
                    workon = workon.to(self.device, non_blocking=False)

                    prediction, encoder_output = self._internal_maybe_mirror_and_predict(workon)
                    prediction = prediction[0].to(results_device)

                    predicted_logits[sl] += (prediction * gaussian if self.use_gaussian else prediction)
                    n_predictions[sl[1:]] += (gaussian if self.use_gaussian else 1)
                    encoder_outputs.append(encoder_output)

                predicted_logits /= n_predictions
        empty_cache(self.device)
        return predicted_logits[tuple([slice(None), *slicer_revert_padding[1:]])], encoder_outputs


def predict_bottleneck_features_entry_point():
    import argparse
    parser = argparse.ArgumentParser(description='Use this to run inference with nnU-Net. This function is used when '
                                                 'you want to manually specify a folder containing a trained nnU-Net '
                                                 'model. This is useful when the nnunet environment variables '
                                                 '(nnUNet_results) are not set.')
    parser.add_argument('-i', type=str, required=True,
                        help='input folder. Remember to use the correct channel numberings for your files (_0000 etc). '
                             'File endings must be the same as the training dataset!')
    parser.add_argument('-o', type=str, required=True,
                        help='Output folder. If it does not exist it will be created. Predicted segmentations will '
                             'have the same name as their source images.')
    parser.add_argument('-d', type=str, required=True,
                        help='Dataset with which you would like to predict. You can specify either dataset name or id')
    parser.add_argument('-p', type=str, required=False, default='nnUNetPlans',
                        help='Plans identifier. Specify the plans in which the desired configuration is located. '
                             'Default: nnUNetPlans')
    parser.add_argument('-tr', type=str, required=False, default='nnUNetTrainer',
                        help='What nnU-Net trainer class was used for training? Default: nnUNetTrainer')
    parser.add_argument('-c', type=str, required=True,
                        help='nnU-Net configuration that should be used for prediction. Config must be located '
                             'in the plans specified with -p')
    parser.add_argument('-f', nargs='+', type=str, required=False, default=(0, 1, 2, 3, 4),
                        help='Specify the folds of the trained model that should be used for prediction. '
                             'Default: (0, 1, 2, 3, 4)')
    parser.add_argument('-step_size', type=float, required=False, default=0.5,
                        help='Step size for sliding window prediction. The larger it is the faster but less accurate '
                             'the prediction. Default: 0.5. Cannot be larger than 1. We recommend the default.')
    parser.add_argument('--disable_tta', action='store_true', required=False, default=False,
                        help='Set this flag to disable test time data augmentation in the form of mirroring. Faster, '
                             'but less accurate inference. Not recommended.')
    parser.add_argument('--verbose', action='store_true', help="Set this if you like being talked to. You will have "
                                                               "to be a good listener/reader.")
    parser.add_argument('--save_probabilities', action='store_true',
                        help='Set this to export predicted class "probabilities". Required if you want to ensemble '
                             'multiple configurations.')
    parser.add_argument('--continue_prediction', action='store_true',
                        help='Continue an aborted previous prediction (will not overwrite existing files)')
    parser.add_argument('-chk', type=str, required=False, default='checkpoint_final.pth',
                        help='Name of the checkpoint you want to use. Default: checkpoint_final.pth')
    parser.add_argument('-npp', type=int, required=False, default=3,
                        help='Number of processes used for preprocessing. More is not always better. Beware of '
                             'out-of-RAM issues. Default: 3')
    parser.add_argument('-nps', type=int, required=False, default=3,
                        help='Number of processes used for segmentation export. More is not always better. Beware of '
                             'out-of-RAM issues. Default: 3')
    parser.add_argument('-prev_stage_predictions', type=str, required=False, default=None,
                        help='Folder containing the predictions of the previous stage. Required for cascaded models.')
    parser.add_argument('-num_parts', type=int, required=False, default=1,
                        help='Number of separate nnUNetv2_predict call that you will be making. Default: 1 (= this one '
                             'call predicts everything)')
    parser.add_argument('-part_id', type=int, required=False, default=0,
                        help='If multiple nnUNetv2_predict exist, which one is this? IDs start with 0 can end with '
                             'num_parts - 1. So when you submit 5 nnUNetv2_predict calls you need to set -num_parts '
                             '5 and use -part_id 0, 1, 2, 3 and 4. Simple, right? Note: You are yourself responsible '
                             'to make these run on separate GPUs! Use CUDA_VISIBLE_DEVICES (google, yo!)')
    parser.add_argument('-device', type=str, default='cuda', required=False,
                        help="Use this to set the device the inference should run with. Available options are 'cuda' "
                             "(GPU), 'cpu' (CPU) and 'mps' (Apple M1/M2). Do NOT use this to set which GPU ID! "
                             "Use CUDA_VISIBLE_DEVICES=X nnUNetv2_predict [...] instead!")

    print(
        "\n#######################################################################\nPlease cite the following paper "
        "when using nnU-Net:\n"
        "Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). "
        "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. "
        "Nature methods, 18(2), 203-211.\n#######################################################################\n")

    args = parser.parse_args()
    args.f = [i if i == 'all' else int(i) for i in args.f]

    model_folder = get_output_folder(args.d, args.tr, args.p, args.c)

    if not isdir(args.o):
        maybe_mkdir_p(args.o)

    # slightly passive agressive haha
    assert args.part_id < args.num_parts, 'Do you even read the documentation? See nnUNetv2_predict -h.'

    assert args.device in ['cpu', 'cuda',
                           'mps'], f'-device must be either cpu, mps or cuda. Other devices are not tested/supported. Got: {args.device}.'
    if args.device == 'cpu':
        # let's allow torch to use hella threads
        import multiprocessing
        torch.set_num_threads(multiprocessing.cpu_count())
        device = torch.device('cpu')
    elif args.device == 'cuda':
        # multithreading in torch doesn't help nnU-Net if run on GPU
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        device = torch.device('cuda')
    else:
        device = torch.device('mps')

    predictor = nnUNetPredictor(tile_step_size=args.step_size,
                                use_gaussian=True,
                                use_mirroring=not args.disable_tta,
                                perform_everything_on_gpu=True,
                                device=device,
                                verbose=args.verbose,
                                verbose_preprocessing=False)
    predictor.initialize_from_trained_model_folder(
        model_folder,
        args.f,
        checkpoint_name=args.chk
    )
    predictor.export_bn_features_from_files(args.i, args.o, save_probabilities=args.save_probabilities,
                                 overwrite=not args.continue_prediction,
                                 num_processes_preprocessing=args.npp,
                                 num_processes_segmentation_export=args.nps,
                                 folder_with_segs_from_prev_stage=args.prev_stage_predictions,
                                 num_parts=args.num_parts,
                                 part_id=args.part_id)


if __name__ == '__main__':
    # predict a bunch of files
    predict_bottleneck_features_entry_point()
    exit()

