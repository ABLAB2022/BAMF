from IPython.display import clear_output
from statistics import mean
import torch
import torch.nn as nn
from DATA_SPLITTER.dataloader.pointwise import CustomizedDataLoader as PointwiseDataLoader
from DATA_SPLITTER.dataloader.pairwise import CustomizedDataLoader as PairwiseDataLoader
from DATA_SPLITTER.dataloader.listwise import CustomizedDataLoader as ListwiseDataLoader
from .trainer.pointwise import CustomizedTrainer as PointwiseTrainer
from .trainer.pairwise import CustomizedTrainer as PairwiseTrainer
from .trainer.listwise import CustomizedTrainer as ListwiseTrainer
from .monitor import monitor


class Runner:
    def __init__(
        self, 
        model: nn.Module, 
        trainer: PointwiseTrainer | PairwiseTrainer | ListwiseTrainer,
        monitor: monitor.EarlyStoppingMonitor,
    ):
        """
        CustomizedTrainer Runner for Latent Factor Model
        -----
        created by @jayarnim

        Args:
            model (nn.Module):
                latent factor model instance.
            trainer (CustomizedTrainer):
                single epoch trainer instance, `pointwise`, `pairwise`, or `listwise`.
            monitor (CustomizedTrainer):
                early stopping monitor.
        """
        # device setting
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(DEVICE)

        # global attr
        self.model = model.to(self.device)
        self.trainer = trainer
        self.monitor = monitor

    def fit(
        self, 
        trn_loader: PointwiseDataLoader | PairwiseDataLoader | ListwiseDataLoader, 
        val_loader: PointwiseDataLoader | PairwiseDataLoader | ListwiseDataLoader, 
        loo_loader: PointwiseDataLoader, 
        n_epochs: int, 
        annealing: int,
        warm_up: int=10,
        interval: int=1,
    ):
        """
        Executing Trainer Method

        Args:
            trn_loader (CustomizedDataLoader):
                DataLoader for the training set.
            val_loader (CustomizedDataLoader):
                DataLoader for the validation set.
            loo_loader (CustomizedDataLoader):
                DataLoader for leave-one-out evaluation, used to monitor early stopping performance.
            n_epochs (int):
                Total number of epochs for training.
            warm_up (int):
                Number of initial epochs before starting early stopping monitoring.
            interval (int):
                Interval (in epochs) between validation checks for early stopping.

        Returns:
            history (dict): 
                - `trn`: loss values recorded during training epochs.
                - `val`: loss values recorded during validation epochs.
        """
        trn_nll_list, trn_kl_list = [], []
        val_nll_list, val_kl_list = [], []
        loo_score_list = []
        computing_cost_list = []

        for epoch in range(n_epochs):
            if epoch % 10 == 0:
                print(f"EPOCH {epoch+1} START ---->>>>")

            # trn, val
            kwargs = dict(
                trn_loader=trn_loader, 
                val_loader=val_loader, 
                epoch=epoch,
                n_epochs=n_epochs,
                annealing=annealing,
            )
            (trn_nll, trn_kl), (val_nll, val_kl), computing_cost = self._run_trainer(**kwargs)

            # loo
            kwargs = dict(
                loo_loader=loo_loader,
                epoch=epoch,
                n_epochs=n_epochs,
                warm_up=warm_up,
                interval=interval,
            )
            loo_score = self._run_monitor(**kwargs)

            # accumulate
            trn_nll_list.append(trn_nll)
            trn_kl_list.append(trn_kl)
            val_nll_list.append(val_nll)
            val_kl_list.append(val_kl)
            loo_score_list.append(loo_score)
            computing_cost_list.extend(computing_cost)

            # early stopping
            if self.monitor.get_should_stop:
                break

            # log reset
            if (epoch + 1) % 50 == 0:
                clear_output(wait=False)

        # log reset
        clear_output(wait=False)

        kwargs = dict(
            n_epochs=n_epochs, 
            trn_nll_list=trn_nll_list, 
            trn_kl_list=trn_kl_list,
            val_nll_list=val_nll_list, 
            val_kl_list=val_kl_list,
            loo_score_list=loo_score_list,
            computing_cost_list=computing_cost_list,
        )
        return self._finalizer(**kwargs)

    def _run_trainer(self, trn_loader, val_loader, epoch, n_epochs, annealing):
        kwargs = dict(
            trn_loader=trn_loader, 
            val_loader=val_loader, 
            epoch=epoch,
            n_epochs=n_epochs,
            annealing=annealing,
        )
        (trn_nll, trn_kl), (val_nll, val_kl), computing_cost = self.trainer(**kwargs)

        print(
            f"TRN NLL: {trn_nll:.4f}",
            f"TRN KL: {trn_kl:.4f}",
            sep='\t\t',
        )
        print(
            f"VAL NLL: {val_nll:.4f}",
            f"VAL KL: {val_kl:.4f}",
            sep='\t\t',
        )

        return (trn_nll, trn_kl), (val_nll, val_kl), computing_cost

    def _run_monitor(self, loo_loader, epoch, n_epochs, warm_up, interval):
        if ((epoch+1) > warm_up) and ((epoch+1) % interval == 0):
            kwargs = dict(
                loo_loader=loo_loader, 
                epoch=epoch,
                n_epochs=n_epochs,
            )
            loo_score = self.monitor(**kwargs)

            print(
                f"CURRENT SCORE: {loo_score:.4f}",
                f"BEST SCORE: {self.monitor.get_best_score:.4f}",
                f"BEST EPOCH: {self.monitor.get_best_epoch}",
                sep='\t',
            )

            return loo_score

    def _finalizer(self, n_epochs, trn_nll_list, trn_kl_list, val_nll_list, val_kl_list, loo_score_list, computing_cost_list):
        best_epoch = self.monitor.get_best_epoch
        best_score = self.monitor.get_best_score
        best_model_state = self.monitor.get_best_model_state

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        print(
            "LEAVE ONE OUT",
            f"\tBEST SCORE: {best_score:.4f}",
            f"\tBEST EPOCH: {best_epoch}",
            sep="\n",
        )
        print(
            "COMPUTING COST FOR LEARNING",
            f"\t(s/epoch): {sum(computing_cost_list)/n_epochs:.4f}",
            f"\t(epoch/s): {n_epochs/sum(computing_cost_list):.4f}",
            f"\t(s/batch): {mean(computing_cost_list):.4f}",
            f"\t(batch/s): {1.0/mean(computing_cost_list):.4f}",
            sep="\n",
        )

        history_nll = dict(
            trn=trn_nll_list,
            val=val_nll_list,
        )
        history_kl = dict(
            trn=trn_kl_list,
            val=val_kl_list,
        )

        return dict(
            nll=history_nll,
            kl=history_kl,
            loo=loo_score_list,
        )