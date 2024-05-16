import matplotlib
from batchgenerators.utilities.file_and_folder_operations import join

matplotlib.use('agg')
import seaborn as sns
import matplotlib.pyplot as plt

from nnunetv2.training.logging.nnunet_logger import nnUNetLogger


class nnUNetUnlearnLogger(nnUNetLogger):
    """Additionaly log unlearn steps. Number of axes is defined by "domains" parameter
    """

    def __init__(self, verbose: bool = False, unlearn_plots: int = 0):
        super().__init__(verbose)
        self.unlearn_plots = unlearn_plots

    def log(self, key, value, epoch: int):
        """
        sometimes shit gets messed up. We try to catch that here
        """
        assert key in self.my_fantastic_logging.keys() and isinstance(self.my_fantastic_logging[key], list), \
            'This function is only intended to log stuff to lists and to have one entry per epoch'

        if self.verbose: print(f'logging {key}: {value} for epoch {epoch}')

        if len(self.my_fantastic_logging[key]) < (epoch + 1):
            self.my_fantastic_logging[key].append(value)
        else:
            assert len(self.my_fantastic_logging[key]) == (epoch + 1), 'something went horribly wrong. My logging ' \
                                                                       'lists length is off by more than 1'
            print(f'maybe some logging issue!? logging {key} and {value}')
            self.my_fantastic_logging[key][epoch] = value

        # handle the ema_fg_dice special case! It is automatically logged when we add a new mean_fg_dice
        if key == 'mean_fg_dice':
            new_ema_pseudo_dice = self.my_fantastic_logging['ema_fg_dice'][epoch - 1] * 0.9 + 0.1 * value \
                if len(self.my_fantastic_logging['ema_fg_dice']) > 0 else value
            self.log('ema_fg_dice', new_ema_pseudo_dice, epoch)

        # compute moving average for validation accuracy of domain predictor
        if key.startswith("val_acc_domain"):
            domain_idx = key[-1]
            new_ema_val_acc_domain = self.my_fantastic_logging[f'ema_val_acc_domain_{domain_idx}'][epoch - 1] * 0.9 + 0.1 * value \
                if len(self.my_fantastic_logging[f'ema_val_acc_domain_{domain_idx}']) > 0 else value
            self.log(f'ema_val_acc_domain_{domain_idx}', new_ema_val_acc_domain, epoch)

        if key.startswith("train_acc_domain"):
            domain_idx = key[-1]
            new_ema_train_acc_domain = self.my_fantastic_logging[f'ema_train_acc_domain_{domain_idx}'][epoch - 1] * 0.9 + 0.1 * value \
                if len(self.my_fantastic_logging[f'ema_train_acc_domain_{domain_idx}']) > 0 else value
            self.log(f'ema_train_acc_domain_{domain_idx}', new_ema_train_acc_domain, epoch)

    def plot_progress_png(self, output_folder):
        # we infer the epoch form our internal logging
        epoch = min([len(i) for i in self.my_fantastic_logging.values()]) - 1
        sns.set(font_scale=2.5)

        subplots_base = 3
        subplots_total = subplots_base + self.unlearn_plots + 1   # +1 for cunfusion
        fig, ax_all = plt.subplots(subplots_total, 1, figsize=(50, 108))

        ax = ax_all[0]
        ax2 = ax.twinx()
        x_values = list(range(epoch + 1))
        ax.plot(x_values, self.my_fantastic_logging['train_losses'][:epoch + 1], color='b', ls='-', label="loss_tr", linewidth=4)
        ax.plot(x_values, self.my_fantastic_logging['val_losses'][:epoch + 1], color='r', ls='-', label="loss_val", linewidth=4)
        ax2.plot(x_values, self.my_fantastic_logging['mean_fg_dice'][:epoch + 1], color='g', ls='dotted', label="pseudo dice", linewidth=3)
        ax2.plot(x_values, self.my_fantastic_logging['ema_fg_dice'][:epoch + 1], color='g', ls='-', label="pseudo dice (mov. avg.)", linewidth=4)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax2.set_ylabel("pseudo dice")
        ax.legend(loc=(0, 1))
        ax2.legend(loc=(0.4, 1))

        # epoch times to see whether the training speed is consistent (inconsistent means there are other jobs
        # clogging up the system)
        ax = ax_all[1]
        ax.plot(x_values, [i - j for i, j in zip(self.my_fantastic_logging['epoch_end_timestamps'][:epoch + 1],
                                                 self.my_fantastic_logging['epoch_start_timestamps'])][:epoch + 1], color='b',
                ls='-', label="epoch duration", linewidth=4)
        ylim = [0] + [ax.get_ylim()[1]]
        ax.set(ylim=ylim)
        ax.set_xlabel("epoch")
        ax.set_ylabel("time [s]")
        ax.legend(loc=(0, 1))

        # learning rate
        ax = ax_all[2]
        ax.plot(x_values, self.my_fantastic_logging['lrs'][:epoch + 1], color='b', ls='-', label="learning rate", linewidth=4)
        ax.set_xlabel("epoch")
        ax.set_ylabel("learning rate")
        ax.legend(loc=(0, 1))

        # TODO: each domain loss has its‹ own plot space
        colors = ['b','g','r','c','m','y','k','w']
        for domain_ax in range(self.unlearn_plots):
            ax = ax_all[domain_ax + subplots_base]
            ax2 = ax.twinx()
            ax.set_xlabel("epoch")

            # loss
            ax.plot(x_values, self.my_fantastic_logging[f'train_losses_domain_{domain_ax}'][:epoch + 1], color='orange', ls='-', label=f"domain_loss TRAIN {domain_ax}", linewidth=4)
            # ax.plot(x_values, self.my_fantastic_logging[f'val_losses_domain_{domain_ax}'][:epoch + 1], color='gold', ls='--', label=f"domain_loss VAL {domain_ax}", linewidth=4)
            ax.plot(x_values, self.my_fantastic_logging[f'train_losses_confusion_{domain_ax}'][:epoch + 1], color='orangered', ls='-', label=f"domain_confusion_loss {domain_ax}", linewidth=4)
            # legend
            ax.set_ylabel("loss")
            ax.legend(loc=(0, 1))

            # accuracy
            ax2.plot(x_values, self.my_fantastic_logging[f"train_acc_domain_{domain_ax}"][:epoch + 1], color="green", ls='dotted', label=f"domain_ACC_TRAIN {domain_ax}", linewidth=4)
            ax2.plot(x_values, self.my_fantastic_logging[f"ema_train_acc_domain_{domain_ax}"][:epoch + 1], color="green", ls='-', label=f"Moving Average ACC TRAIN {domain_ax}", linewidth=4)
            ax2.plot(x_values, self.my_fantastic_logging[f"val_acc_domain_{domain_ax}"][:epoch + 1], color="cyan", ls='dotted', label=f"domain_ACC_VAL {domain_ax}", linewidth=4)
            ax2.plot(x_values, self.my_fantastic_logging[f"ema_val_acc_domain_{domain_ax}"][:epoch + 1], color="cyan", ls='-', label=f"Moving Average ACC VAL {domain_ax}", linewidth=4)
            # legend
            ax2.set_ylabel("accuracy")
            ax2.legend(loc=(0.4, 1))

        # confusion loss during training
        ax = ax_all[-1]
        colors = ['b','g','r','c','m','y','k','w']
        for enum, key in enumerate([x for x in self.my_fantastic_logging.keys() if x.startswith('train_losses_confusion')]):
            if self.my_fantastic_logging[key]:
                color = colors[enum % len(colors)]
                ax.plot(x_values, self.my_fantastic_logging[key][:epoch + 1], color=color, ls='dotted', label=f"Confusion L TRAIN {enum}", linewidth=4)
        ax.legend(loc=(0, 1))

        plt.tight_layout()

        fig.savefig(join(output_folder, "progress.png"))
        plt.close()
