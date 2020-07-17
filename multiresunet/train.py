import argparse

from multiresunet.splitter import splitter
from multiresunet.dataset import ThyroidNoduleDataset
from multiresunet.transform import preprocessing
from multiresunet.model.MultiResUNet import MultiResUNet
from multiresunet.trainer import Trainer
from multiresunet.metric import dice_coeff

import torch
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser()
parser.add_argument(
    '--val_ratio', type=int, default=30
)
parser.add_argument(
    '--batch_size', type=int, default=5
)
parser.add_argument(
    '--epoch', type=int, default=20
)
parser.add_argument(
    '--lr', type=float, default=0.001
)
parser.add_argument(
    '--dataset', type=str, default='./data/'
)
parser.add_argument(
    '--workers', type=int, default=4
)
parser.add_argument(
    '--save_model', type=str, default='./save_model/'
)

cfg = parser.parse_args()
print(cfg)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# splitter(cfg.dataset, cfg.val_ratio)

if __name__ == "__main__":
    ds_train = ThyroidNoduleDataset(root='./data/', split='train', transform=preprocessing)
    ds_test = ThyroidNoduleDataset(root='./data/', split='val', transform=preprocessing)
    dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.workers)
    dl_test = DataLoader(ds_test, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.workers)
    print("DATA LOADED")

    model = MultiResUNet(3, 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = torch.nn.BCEWithLogitsLoss()
    success_metric = dice_coeff

    trainer = Trainer(model, criterion, optimizer, success_metric, device, None)
    fit = trainer.fit(dl_train, dl_test, num_epochs=cfg.epoch, checkpoints=cfg.save_model+model.__class__.__name__+'.pt')
    torch.save(model.state_dict(), './model_ckpt/final_state_dict.pt')
    torch.save(model, './model_ckpt/final.pt')

    loss_fn_name = "BCELoss"
    best_score = str(fit.best_score)
    print(f"Best loss score(loss function = {loss_fn_name}): {best_score}")
