import copy
from collections import OrderedDict
from typing import Dict, Optional

import torch
from omegaconf import DictConfig

from mol.dimenet.dimenet import DimeNet
from mol.dimenet.loader import AtomsBatch
from mol.logging import configure_logging
from mol.loss import mae_loss
from mol.params import Params
from mol.train_base import PLBaseModule, train


class Net(PLBaseModule):
    def __init__(self, hparams: DictConfig):
        super().__init__()
        self.hparams = hparams
        self.model = dimenet(self.hparams.pretrained_ckpt_path)

        if self.hp.ema_decay is not None:
            self.ema_model = copy.deepcopy(self.model)
            for p in self.ema_model.parameters():
                p.requires_grad_(False)

    def step(self, batch, prefix: str, model=None) -> Dict:
        batch = AtomsBatch(**batch)
        y_true = batch.mu.unsqueeze(-1)

        if model is None:
            y_pred = self.forward(batch)[:, :1]
        else:
            y_pred = model(batch)[:, :1]

        assert y_pred.shape == y_true.shape, f'{y_pred.shape}, {y_true.shape}'

        mae = mae_loss(y_pred, y_true)
        lmae = torch.log(mae)
        size = len(y_true)

        return {
            f'{prefix}_loss': lmae,
            f'{prefix}_mae': mae,
            f'{prefix}_size': size,
        }


def dimenet(ckpt_path: Optional[str]) -> DimeNet:
    model = DimeNet(
        num_targets=3,  # But use only first one.
        return_hidden_outputs=False,
    )
    if ckpt_path is None:
        return model

    print(f'Load {ckpt_path}...')

    ckpt = torch.load(ckpt_path)

    new_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        if not k.startswith('ema_model'):
            continue
        new_dict[k[10:]] = v

    model.load_state_dict(new_dict)

    return model


if __name__ == '__main__':
    configure_logging()
    params = Params.load()
    train(Net, params)
