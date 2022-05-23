import cv2
import os
import argparse
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from models import MARNDD, BasicBlock
from utils import *
from data_loaders import *
import scipy.misc
from skimage.measure import compare_ssim as ssim
import os, time, datetime

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="MARN")
parser.add_argument("--num_of_layers", type=int, default=20, help="Number of total layers")
parser.add_argument("--logdir", type=str, default="drive/My Drive/",
                    help='path of log files')
parser.add_argument("--test_data", type=str,
                    default='drive/My Drive/marndd- backup.zip (Unzipped Files)/testdata/',
                    help='test on Set12 or Set68')
parser.add_argument("--test_noiseL", type=float, default=0, help='noise level used on test set')
opt = parser.parse_args()
apply_bilinear = False
i = 1

def log(*args, **kwargs):
     print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)


def normalize(data):
    return data / 255.

def main():
    # Build model
    print('Loading model 0r 1...\n')
    net = MARNDD(BasicBlock, num_of_layers=opt.num_of_layers)
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    model.load_state_dict(torch.load(os.path.join(opt.logdir, 'model_best.pth')))
    model.eval()
    # load data info
    print('Loading data info ba...\n')
    # files_source = glob.glob(os.path.join(opt.test_data, '*.png'))
    # files_source.sort()
    # process data
    psnr_test = 0
    ssm_test = 0
    print(os.path.join(opt.test_data, '*.png'))
    test_batch_size = 1
    demosaic_dataset_kodak = KodakDataset(root_dir=os.path.join(opt.test_data, 'kodak'), apply_bilinear=apply_bilinear,
                                          transform=ToTensor())
    # report performance on MCM test set
    demosaic_dataset_mcmd = MCMDataset(root_dir=os.path.join(opt.test_data, 'McM'), apply_bilinear=apply_bilinear,
                                       transform=ToTensor())
    datasets = [(demosaic_dataset_kodak, 'Kodak'), (demosaic_dataset_mcmd, 'MCM')]
    for demosaic_dataset_test, dataset_name in datasets:
        dataloader_test = DataLoader(demosaic_dataset_test, batch_size=test_batch_size,
                                     shuffle=False, num_workers=1, pin_memory=True)
        for i, sample in enumerate(dataloader_test):
            groundtruth = sample['image_gt']
            mosaic = sample['image_input']
            name = sample['filename']
            M = sample['mask']
            i_net = mosaic.type(torch.cuda.FloatTensor)
            ISource = groundtruth.type(torch.cuda.FloatTensor)
            # noise
            # noise = torch.cuda.FloatTensor(np.random.normal(loc=0.0, scale = opt.test_noiseL, size =  np.array([3,512, 768]).size() ))
            noise = torch.cuda.FloatTensor(i_net.size()).normal_(mean=0, std=opt.test_noiseL / 1.)
            # noisy image
            INoisy = i_net.type(torch.cuda.FloatTensor)  + noise.type(torch.cuda.FloatTensor)
            start_time = time.time()
          
            with torch.no_grad():  # this can save much memory
                Out = torch.clamp((INoisy - model(INoisy)), 0., 255.)
                Outresidual = torch.clamp( model(INoisy), 0., 255.)
            ## if you are using older version of PyTorch, torch.no_grad() may not be supported
            Out1 = tensor2Im((Out).cpu())
            Outresidual1 = tensor2Im((Outresidual).cpu())
            
            elapsed_time = time.time() - start_time
           
           
            print('%10s : %10s : %2.4f second' % (dataset_name, name, elapsed_time))
           
            # bayer_rgb = remosaicv2(groundtruth, 1)
            mosaic = tensor2Im(mosaic.cpu())
            groundtruth = tensor2Im(groundtruth.cpu())
            
            ssm=0
            for i_ in range(Out1.shape[0]):
                name_ = name[i_].replace('/', '_')
                scipy.misc.imsave(opt.test_data + name_ + '_output.png', Out1[i_].clip(0, 255).astype(np.uint8))
                scipy.misc.imsave(opt.test_data + name_ + '_original.png',
                                  groundtruth[i_].clip(0, 255).astype(np.uint8))
                ssm += ssim( groundtruth[i_].clip(0, 255).astype(np.uint8),Out1[i_].clip(0, 255).astype(np.uint8),dynamic_range=255.,multichannel =True)
                scipy.misc.imsave(opt.test_data + name_ + '_mosaic.png', mosaic[i_].clip(0, 255).astype(np.uint8))
                scipy.misc.imsave(opt.test_data + name_ + '_residual.png', Outresidual1[i_].clip(0, 255).astype(np.uint8))
                # 3ch bayer

            i += 1
            psnr = batch_PSNR(ISource,Out, 255.)
            ssm = ssm /Out1.shape[0]
            # ssm = ssims(Out,ISource, 255.0)
            psnr_test += psnr
            ssm_test +=ssm
            print("%s PSNR %f SSIM %f" % (name, psnr, ssm))
        psnr_test /= len(dataloader_test)
        ssm_test /= len(dataloader_test)
        print("\nPSNR on test data %f" % psnr_test)
        print("\nSSIM on test data %f" % ssm_test)
        psnr_test = 0
        ssm_test = 0


if __name__ == "__main__":
    main()
