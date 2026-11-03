import warnings
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from utils.metrics import mIoUMetric
from utils.tools import (
    DURD_Loss,
    SoftIoULoss,
    make_dir,
    para_parser,
    train_head,
)
warnings.filterwarnings("ignore")

args = para_parser()
# NUDT-SIRST  NUST-SIRST  SIRST
args.dataset = 'NUDT-SIRST'
args.paper = 'DURD'
args.save_path = make_dir(args.dataset)

model, train_dataloader, test_dataloader = train_head(
    args)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)

# ------------------------------------------------------------------
optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                              model.parameters()),
                       lr=args.lr)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                 T_max=args.epochs,
                                                 eta_min=args.min_lr)
# ------------------------------------------------------------------
mIoU = mIoUMetric(1)
criterion = DURD_Loss(pred_loss_fn=SoftIoULoss,
                      recon_loss_fn=torch.nn.MSELoss(),
                      recon_weight=0.1)


def training_epoch(e):
    model.train()
    loss_set = []
    tbar = tqdm(train_dataloader)
    for _, batch_image, batch_label in tbar:
        batch_image = batch_image.to(device)
        batch_label = batch_label.to(device)

        outputs = model(batch_image)
        loss = criterion(outputs, batch_image, batch_label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_set.append(loss.item())
        tbar.set_description('Epoch %d, training loss %.4f, lr %.6f' %
                             (e, np.mean(loss_set), optimizer.param_groups[0]['lr']))
    losses_avg = np.mean(loss_set)
    return losses_avg


def testing_epoch(e):
    model.eval()
    loss_set = []
    tbar = tqdm(test_dataloader)
    with torch.no_grad():
        mIoU.reset()
        for _, batch_image, batch_label in tbar:
            batch_image = batch_image.to(device)
            batch_label = batch_label.to(device)

            outputs = model(batch_image)
            loss = criterion(outputs, batch_image, batch_label)

            final_pred = outputs[-1]

            loss_set.append(loss.item())
            mIoU.update(final_pred, batch_label)
            tbar.set_description('Epoch %d, testing loss %.4f' %
                                 (e, np.mean(loss_set)))
        _, mean_IOU = mIoU.get()
        test_loss = np.mean(loss_set)
        print(f'Epoch {e}, Mean IoU: {mean_IOU}， test loss: {test_loss}')
    return test_loss, mean_IOU


def main():
    for epoch in range(args.epochs):
        train_loss = training_epoch(epoch)
        test_loss, mean_IOU = testing_epoch(epoch)

        scheduler.step()


if __name__ == "__main__":
    main()
