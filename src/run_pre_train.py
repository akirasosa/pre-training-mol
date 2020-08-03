import copy
from typing import Dict

import pandas as pd
import torch
from omegaconf import DictConfig

from mol.dimenet.dimenet import DimeNet
from mol.dimenet.loader import AtomsBatch
from mol.logging import configure_logging
from mol.loss import mae_loss
from mol.params import Params
from mol.train_base import PLBaseModule, train
from mylib.sklearn.split import KBinsStratifiedKFold
from mylib.torch.data.dataset import PandasDataset


class PLModule(PLBaseModule):
    def __init__(self, hparams: DictConfig):
        super().__init__()
        self.hparams = hparams
        self.model = DimeNet(
            num_targets=3,
            return_hidden_outputs=True,
        )

    def setup(self, stage: str):
        df = pd.read_parquet(self.hp.db_path)

        if self.train_all:
            self.train_dataset = PandasDataset(df)
            return

        n_atoms = df.Z.apply(len).values.reshape(-1, 1)

        folds = KBinsStratifiedKFold(
            n_splits=self.hp.n_splits,
            random_state=self.hp.seed,
            shuffle=True,
            n_bins=5,
        )
        train_idx, val_idx = list(folds.split(df, n_atoms))[self.hp.fold]

        self.train_dataset = PandasDataset(df.iloc[train_idx])
        self.val_dataset = PandasDataset(df.iloc[val_idx])

    def step(self, batch, prefix: str, model=None) -> Dict:
        batch = AtomsBatch(**batch)
        y_true = batch.R_orig

        if model is None:
            y_pred = self.forward(batch)[1]
        else:
            y_pred = model(batch)[1]

        # predict residual
        y_pred = y_pred + batch.R

        assert y_pred.shape == y_true.shape, f'{y_pred.shape}, {y_true.shape}'

        # TODO try to group by molecule.
        mae = mae_loss(torch.pdist(y_pred), torch.pdist(y_true))
        loss = torch.log(mae)
        size = len(y_true)

        return {
            f'{prefix}_loss': loss,
            f'{prefix}_mae': mae,
            f'{prefix}_size': size,
        }


if __name__ == '__main__':
    configure_logging()
    params = Params.load()
    train(PLModule, params)
