import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from data_loaders import *
from torchvision import transforms
from models import MARNDD,BasicBlock
from dataset import prepare_data, Dataset
from utils import *
import time

import shutil

torch.backends.cudnn.benchmark = True
parser = argparse.ArgumentParser(description="DMARNDD")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser.add_argument("--preprocess", type=bool, default=False, help='run prepare_data or not')
parser.add_argument("--channels", type=int, default=3, help="number of channel")
parser.add_argument('--epochs', action="store", type=int, default=100, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=30, help="When to decay learning rate; should be less than epochs")
parser.add_argument("--batch_size", type=int, default=2, help="Training batch size")
parser.add_argument('--num_of_layers', action="store", type=int, default=20, help="Number of total layers")
parser.add_argument('--init', action="store_true", dest="init", default=False,
                    help="Initialize input with Bilinear Interpolation")
parser.add_argument('--save_images', action="store_true", dest="save_images", default=True)
parser.add_argument('--save_path', action="store", dest="save_path", default='results/',
                    help="Path to save model and results")
parser.add_argument('--gpu', action="store_true", dest="use_gpu", default=False)
parser.add_argument("--mode", type=str, default="B", help='with known noise level (S) or blind training (B)')
parser.add_argument('--num_gpus', action="store", dest="num_gpus", type=int, default=1)
parser.add_argument('--max_iter', action="store", dest="max_iter", type=int, default=10,
                    help="Total number of iterations to use")
parser.add_argument('--lr', action="store", dest="lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--outf", type=str, default="logs", help='path of log files')
parser.add_argument('--k1', action="store", dest="k1", type=int, default=5, help="Number of iterations to unroll")
parser.add_argument('--k2', action="store", dest="k2", type=int, default=5,
                    help="Number of iterations to backpropagate. Use the same value as k1 for TBPTT")
parser.add_argument('--clip', action="store", dest="clip", type=float, default=0.25, help="Gradient Clip")
parser.add_argument('--estimate_noise', action="store_true", dest="noise_estimation", default=False,
                    help="Estimate noise std via WMAD estimator")
parser.add_argument('--val_noiseL', type=float, default=25, help='noise level used on validation set')

opt = parser.parse_args()
print(opt)


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=0, bias=True)
def main():
    # Load dataset
    print('Loading dataset ...\n')
    # if opt.preprocess:
    # if opt.mode == 'S':
    #  prepare_data(data_path='data', patch_size=40, stride=10, aug_times=1)
    # if opt.mode == 'B':
    #  prepare_data(data_path='data', patch_size=50, stride=10, aug_times=2)

    apply_bilinear = False

    # if args.demosaic: # NOTE: currently unused
    #    apply_bilinear=True

    demosaic_dataset = MSRDemosaicDataset(
        root_dir='drive/My Drive/gputest/data/MSR-Demosaicing/Dataset_LINEAR_with_noise/bayer_panasonic/',
        selection_file='drive/My Drive/gputest/data/MSR-Demosaicing/Dataset_LINEAR_with_noise/bayer_panasonic/train.txt',
        apply_bilinear=apply_bilinear, transform=transforms.Compose(
            [RandomHorizontalFlip(), RandomVerticalFlip(), ToTensor()]))

    loader_train = DataLoader(demosaic_dataset, num_workers=4, batch_size=opt.batch_size, shuffle=True, pin_memory=True)

    demosaic_dataset_val = MSRDemosaicDataset(
        root_dir='drive/My Drive/gputest/data/MSR-Demosaicing/Dataset_LINEAR_with_noise/bayer_panasonic/',
        selection_file='drive/My Drive/gputest/data/MSR-Demosaicing/Dataset_LINEAR_with_noise/bayer_panasonic/validation.txt',
        apply_bilinear=apply_bilinear, transform=ToTensor())

    loader_val = DataLoader(demosaic_dataset_val, batch_size=opt.batch_size, shuffle=False, pin_memory=True)


    print("# of training samples: %d\n" % int(len(loader_train)))

    if not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path)
    with open(opt.save_path + 'args.txt', 'wb') as fout:
        fout.write(str.encode(str(opt)))\
    # Build model
    net = MARNDD(BasicBlock,num_of_layers=opt.num_of_layers)
    net.apply(weigths_init_xavier)

    criterion = nn.MSELoss(size_average=False)
    # Move to GPU
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    criterion.cuda()
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    # training
    writer = SummaryWriter(opt.outf)
    step = 0
    noiseL_B = [0, 55]  # ingnored when opt.mode=='S'
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    print("abcdegfh")
    end = time.time()
    for epoch in range(opt.epochs):
        if epoch < opt.milestone:
            current_lr = opt.lr
        else:
            current_lr = opt.lr / 10.
        # set learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print('learning rate %f' % current_lr)
        # train
        for i, data in enumerate(loader_train, 0):
            # training step
            data_time.update(time.time() - end)
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            img_train = data["image_input"].type(torch.FloatTensor)
            
            if opt.mode == 'S':
                noise = torch.cuda.FloatTensor(img_train.size()).normal_(mean=0, std=opt.noiseL / 255.)
            if opt.mode == 'B':
                noise = torch.zeros(img_train.size())
                stdN = np.random.uniform(noiseL_B[0], noiseL_B[1], size=noise.size()[0])
                for n in range(noise.size()[0]):
                    sizeN = noise[0, :, :, :].size()
                    noise[n, :, :, :] = torch.FloatTensor(sizeN).normal_(mean=0, std=stdN[n] / 255.)

            # add noise
            imgn_train = img_train # + noise.type(torch.FloatTensor)
            groundtruth = data['image_gt']
            ground_truth = groundtruth.type(torch.cuda.FloatTensor)

            # wrap them in Variable
            if torch.cuda.is_available():
                img_train, ground_truth = Variable(img_train).cuda(), Variable(ground_truth.type(torch.cuda.FloatTensor)).cuda()
            else:
                i_net, ground_truth = Variable(i_net), Variable(ground_truth)

            img_train, imgn_train = img_train, imgn_train.cuda()
            out_train = model(imgn_train)
            loss = criterion(out_train, ground_truth) /(imgn_train.data.shape[0]*2)
            loss.backward()
            optimizer.step()
            # results
            model.eval()
            out_train = torch.clamp(model(imgn_train), 0., 255.)
            psnr_train = batch_PSNR(out_train, ground_truth, 255.)
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
           
            # if you are using older version of PyTorch, you may need to change loss.item() to loss.data[0]
            if step % 10 == 0:
                # Log the scalar values
                print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f" %
                  (epoch + 1, i + 1, len(loader_train), loss, psnr_train))
                print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(loader_train), batch_time=batch_time,
                                                                  data_time=data_time, loss=losses))
                writer.add_scalar('loss', loss, step)
                writer.add_scalar('PSNR on training data', psnr_train, step)
            step += 1

        ## the end of each epoch
        model.eval()
        # validate
        psnr_val = 0
        for k, data_val in enumerate(loader_val, 0):
            imgval = data_val.get("image_input")
            # ground truth : noisy image - clean image(noise)
            ground_truth = data_val.get("image_gt")

            # wrap them in Variable
            if torch.cuda.is_available():
                imgval, ground_truth = Variable(imgval).cuda(), Variable(ground_truth).cuda()
            else:
                imgval, ground_truth = Variable(imgval), Variable(ground_truth)
            img_val = imgval
            noise = torch.cuda.FloatTensor(img_val.size()).normal_(mean=0, std=opt.val_noiseL / 255.)
            imgn_val = img_val.type(torch.cuda.FloatTensor) # + Variable(noise).cuda()
            img_val, imgn_val = img_val, imgn_val
            out_val = torch.clamp(model(imgn_val), 0., 255.)
            psnr_val += batch_PSNR(out_val, ground_truth,  255.)
        psnr_val /= len(loader_val)
        print(len(loader_val))
        print("\n[epoch %d] PSNR_val: %.4f" % (epoch + 1, psnr_val))
        writer.add_scalar('PSNR on validation data', psnr_val, epoch)
        # log the images
        out_train = torch.clamp(imgn_train - model(imgn_train), 0., 1.)
        Img = utils.make_grid(img_train.data, nrow=8, normalize=True, scale_each=True)
        Imgn = utils.make_grid(imgn_train.data, nrow=8, normalize=True, scale_each=True)
        Irecon = utils.make_grid(out_train.data, nrow=8, normalize=True, scale_each=True)
        writer.add_image('clean image', Img, epoch)
        writer.add_image('noisy image', Imgn, epoch)
        writer.add_image('reconstructed image', Irecon, epoch)
        # save model
        torch.save(model.state_dict(), os.path.join(opt.outf, 'net.pth'))


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


import torchvision


def unnormalize(images):
    img = torchvision.utils.make_grid(images)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    return npimg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 16 epochs"""
    lr = opt.lr * (0.5 ** (epoch // 256))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == "__main__":
    main()
