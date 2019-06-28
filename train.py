import argparse
import torch
import tqdm
import logger
import numpy as np
import torch.nn as nn
import pickle
import metrics
from skimage import io
from skimage import transform
from model import FusionNet, DilationCNN, UNet
from dataset import NucleiDataset
from torch.utils.data import DataLoader
from loss import dice_loss
import imageio
import glob
import os


def main(args):

    # tensorboard
    logger_tb = logger.Logger(log_dir=args.experiment_name)

    # train dataloader and val dataset
    train_dataset = NucleiDataset(args.train_data, 'train')
    val_dataset = NucleiDataset(args.val_data, 'val')

    train_params = {'batch_size': args.batch_size,
                    'shuffle': False,
                    'num_workers': args.num_workers}

    train_dataloader = DataLoader(train_dataset, **train_params)

    # device
    device = torch.device(args.device)

    # model
    if args.model == "fusion":
        model = FusionNet(args, train_dataset.dim)
    elif args.model == "dilation":
        model = DilationCNN(train_dataset.dim)
    elif args.model == "unet":
        model = UNet(args.num_kernel, args.kernel_size, train_dataset.dim)

    if args.device == "cuda":
        # parse gpu_ids for data paralle
        if ',' in args.gpu_ids:
            gpu_ids = [int(ids) for ids in args.gpu_ids.split(',')]
        else:
            gpu_ids = int(args.gpu_ids)

        # parallelize computation
        if type(gpu_ids) is not int:
            model = nn.DataParallel(model, gpu_ids)
    model.to(device)

    # optimizer
    parameters = model.parameters()
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(parameters, args.lr)
    else:
        optimizer = torch.optim.SGD(parameters, args.lr)

    # loss 
    loss_function = dice_loss

    # train model
    for epoch in range(args.epoch):
        model.train()

        with tqdm.tqdm(total=len(train_dataloader.dataset), unit=f"epoch {epoch} itr") as progress_bar:
            total_loss = []
            total_iou = []
            total_precision = []
            for i, (x_train, y_train) in enumerate(train_dataloader):

                with torch.set_grad_enabled(True):

                    # send data and label to device
                    x = torch.Tensor(x_train.float()).to(device)
                    y = torch.Tensor(y_train.float()).to(device)

                    # predict segmentation
                    pred = model.forward(x)

                    # calculate loss
                    loss = loss_function(pred, y)
                    total_loss.append(loss.item()) 

                    # calculate IoU precision

                    predictions = pred.clone().squeeze().detach().cpu().numpy()
                    gt = y.clone().squeeze().detach().cpu().numpy()
                    ious = [metrics.get_ious(p, g, 0.5) for p,g in zip(predictions, gt)]
                    total_iou.append(np.mean(ious))


                    # back prop
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # log loss and iou 
                avg_loss = np.mean(total_loss)
                avg_iou = np.mean(total_iou)

                logger_tb.update_value('train loss', avg_loss, epoch)
                logger_tb.update_value('train iou', avg_iou, epoch)

                progress_bar.update(len(x))

        # validation
        model.eval()
        for idx in range(len(val_dataset)):
            x_val, y_val, mask_val = val_dataset.__getitem__(idx)

            total_precision = []
            total_iou = []
            total_loss = []
            with torch.no_grad():

                # send data and label to device
                x_val = np.expand_dims(x_val, axis=0)
                x = torch.Tensor(torch.tensor(x_val).float()).to(device)
                y = torch.Tensor(torch.tensor(y_val).float()).to(device)

                # predict segmentation
                pred = model.forward(x)

                # calculate loss
                loss = loss_function(pred, y)
                total_loss.append(loss.item())

                # calculate IoU
                prediction = pred.clone().squeeze().detach().cpu().numpy()
                gt = y.clone().squeeze().detach().cpu().numpy()
                iou = metrics.get_ious(prediction, gt, 0.5)
                total_iou.append(iou)
                
                # calculate precision
                precision = metrics.compute_precision(prediction, mask_val, 0.5)
                total_precision.append(precision)

                # display segmentation on tensorboard 
                if idx == 0:
                    original = x_val
                    truth = np.expand_dims(y_val,axis=0)
                    seg = pred.cpu().squeeze().detach().numpy()
                    seg = np.expand_dims(seg, axis=0)

                    logger_tb.update_image("original", original, 0)
                    logger_tb.update_image("ground truth", truth, 0)
                    logger_tb.update_image("segmentation", seg, epoch)
              

        # log metrics
        logger_tb.update_value('val loss', np.mean(total_loss), epoch)
        logger_tb.update_value('val iou', np.mean(total_iou), epoch)
        logger_tb.update_value('val precision', np.mean(total_precision), epoch)
                

    # save model 
    ckpt_dict = {'model_name': model.__class__.__name__, 
                 'model_args': model.args_dict(), 
                 'model_state': model.to('cpu').state_dict()}
    ckpt_path = os.path.join(args.save_dir, f"{model.__class__.__name__}.pth")
    torch.save(ckpt_dict, ckpt_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--num_kernel', type=int, default=8)
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--data_dir', type=str, default="/home/mars/CZI/data")
    parser.add_argument('--train_data', type=str, default="/home/mars/CZI/data/train.hdf5")
    parser.add_argument('--val_data', type=str, default="/home/mars/CZI/data/val.hdf5")
    parser.add_argument('--save_dir', type=str, default="./")
    parser.add_argument('--device', type=str, default=0.1)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--model', type=str, default='fusion')
    parser.add_argument('--batch_size', type=int, default='8')
    parser.add_argument('--shuffle', type=bool, default=False)
    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument('--num_workers', type=int, default='16')
    parser.add_argument('--experiment_name', type=str, default='test')
    args = parser.parse_args()

    main(args)
