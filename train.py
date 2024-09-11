# By Yuxiang Sun, Dec. 4, 2019
# Email: sun.yuxiang@outlook.com

import os, argparse, time, datetime
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.utils as vutils
from util.augmentation import RandomFlip, RandomCrop
from model.segformer.builder import EncoderDecoder as segmodel
from util.util import compute_results
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter
from util.VVIG import homographic_transform, euler_to_homography
from util.init_func import prepare_environment, init_optimizer, setup_data_loaders, prepare_directories
from config import config
from util.dice import DiceLoss, DiceHomoLoss, DiceHomoLoss2


def parse_args():
    parser = argparse.ArgumentParser(description='Train with pytorch')
    parser.add_argument('--model_name', '-m', type=str, default='CMX_mit_b2')
    parser.add_argument('--batch_size', '-b', type=int, default=config.batch_size)
    parser.add_argument('--lr_start', '-ls', type=float, default=config.lr)
    parser.add_argument('--gpu', '-g', type=int, default=0)
    parser.add_argument('--epoch_max', '-em', type=int, default=config.nepochs) # please stop training mannully
    parser.add_argument('--epoch_from', '-ef', type=int, default=0) 
    parser.add_argument('--num_workers', '-j', type=int, default=config.num_workers)
    parser.add_argument('--n_class', '-nc', type=int, default=config.num_classes)
    parser.add_argument('--data_dir', '-dr', type=str, default=config.dataset_path)
    parser.add_argument('--pre_weight', '-prw', type=str, default='/pretrained/mit_b2.pth')
    parser.add_argument('--backbone', '-bac', type=str, default='mit_b2')
    parser.add_argument('--loss_strategy', type=str, default='L_dice', choices=['L_dice', 'L_con_acc', 'L_con'],
                    help='Choose the loss strategy: L_dice, L_con_acc, or L_con')
    return parser.parse_args()

augmentation_methods = [
    RandomFlip(prob=0.5),
    RandomCrop(crop_rate=0.1, prob=1.0),
    # RandomCropOut(crop_rate=0.2, prob=1.0),
    # RandomBrightness(bright_range=0.15, prob=0.9),
    # RandomNoise(noise_range=5, prob=0.9),
]

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

def train(epo, model, train_loader, optimizer, writer, accIter, start_datetime, lr_policy):
    loss_meter = AverageMeter()
    loss1_meter = AverageMeter()
    loss2_meter = AverageMeter()
    
    model.train()
    
    for it, (images, labels, names) in enumerate(train_loader):
        n = images.shape[0]
        yaw = torch.randint(-10, 10, (n,), device=images.device).float()
        pitch = torch.randint(-10, 10, (n,), device=images.device).float()
        roll = torch.randint(-5, 5, (n,), device=images.device).float()
        fx = 702.6030497884977
        fy = 703.4541726858521
        cx = 320
        cy = 240
        I = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        K = euler_to_homography(yaw, pitch, roll, I, n , images)

        images = Variable(images).cuda(args.gpu)
        labels = Variable(labels).cuda(args.gpu)

        start_t = time.time()
        optimizer.zero_grad()
        logits = model(images)
        
        images_homo, labels_homo, logits2_homo = homographic_transform(images, labels.unsqueeze(1), logits, K)
        logits_homo = model(images_homo)
        labels_homo = labels_homo.squeeze(1).long()

        # different loss strategies
        if args.loss_strategy == 'L_dice':
            dice_loss = DiceLoss(mode='multiclass', ignore_index=0)
            loss11 = dice_loss(logits, labels)
            loss12 = F.cross_entropy(logits, labels)
            loss1 = loss11 + loss12
            
            dice_loss_homo = DiceLoss(mode='multiclass', ignore_index=0)
            loss21 = dice_loss_homo(logits_homo, labels_homo)
            loss22 = F.cross_entropy(logits_homo, labels_homo)
            loss2 = loss21 + loss22

        elif args.loss_strategy == 'L_con_acc':
            dice_loss = DiceLoss(mode='multiclass', ignore_index=0)
            loss11 = dice_loss(logits, labels)
            loss12 = F.cross_entropy(logits, labels)
            loss1 = loss11 + loss12
            
            dice_loss_homo = DiceHomoLoss(mode='multiclass', ignore_index=0)
            loss21 = dice_loss_homo(logits_homo, labels_homo, logits2_homo)
            loss22 = F.cross_entropy(logits_homo, labels_homo)
            loss2 = loss21 + loss22

        elif args.loss_strategy == 'L_con':
            dice_loss = DiceLoss(mode='multiclass', ignore_index=0)
            loss11 = dice_loss(logits, labels)
            loss12 = F.cross_entropy(logits, labels)
            loss1 = loss11 + loss12
            
            dice_loss_homo = DiceHomoLoss2(mode='multiclass', ignore_index=0)
            loss21 = dice_loss_homo(logits_homo, logits2_homo)
            loss22 = F.cross_entropy(logits_homo, labels_homo)
            loss2 = loss21 + loss22

        
        loss = loss1 + loss2
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), n)
        loss1_meter.update(loss1.item(), n)
        loss2_meter.update(loss2.item(), n)
    
        current_idx = (epo- 0) * config.niters_per_epoch + it
        lr = lr_policy.get_lr(current_idx)

        for i in range(len(optimizer.param_groups)):
            optimizer.param_groups[i]['lr'] = lr

        lr_this_epo=0
        for param_group in optimizer.param_groups:
            lr_this_epo = param_group['lr']

        print('Train: %s, epo %s/%s, iter %s/%s, lr %.8f, %.2f img/sec, loss %.4f, loss_average %.4f, loss1_average %.4f, loss2_average %.4f, time %s' \
            % (args.model_name, epo, args.epoch_max, it+1, len(train_loader), lr_this_epo, len(names)/(time.time()-start_t), float(loss), loss_meter.avg, loss1_meter.avg, loss2_meter.avg, datetime.datetime.now().replace(microsecond=0)-start_datetime))
        if accIter['train'] % 1 == 0:
            writer.add_scalar('Train/loss', loss, accIter['train'])
            writer.add_scalar('Train/loss11', loss11, accIter['train'])
            writer.add_scalar('Train/loss12', loss12, accIter['train'])
            writer.add_scalar('Train/loss21', loss21, accIter['train'])
            writer.add_scalar('Train/loss22', loss22, accIter['train'])
        view_figure = True # note that I have not colorized the GT and predictions here
        if accIter['train'] % 10 == 0:
            if view_figure:
                input_rgb_images = vutils.make_grid(images[:,:3], nrow=8, padding=10) # can only display 3-channel images, so images[:,:3]
                writer.add_image('Train/input_rgb_images', input_rgb_images, accIter['train'])
                input_homo_images = vutils.make_grid(images_homo[:,:3], nrow=8, padding=10) # can only display 3-channel images, so images[:,:3]
                writer.add_image('Train/input_homo_images', input_homo_images, accIter['train'])
                scale = max(1, 255//args.n_class) # label (0,1,2..) is invisable, multiply a constant for visualization
                groundtruth_tensor = labels.unsqueeze(1) * scale  # mini_batch*480*640 -> mini_batch*1*480*640
                groundtruth_tensor = torch.cat((groundtruth_tensor, groundtruth_tensor, groundtruth_tensor), 1)  # change to 3-channel for visualization
                groudtruth_images = vutils.make_grid(groundtruth_tensor, nrow=8, padding=10)
                writer.add_image('Train/groudtruth_images', groudtruth_images, accIter['train'])
                predicted_tensor = logits.argmax(1).unsqueeze(1) * scale # mini_batch*args.n_class*480*640 -> mini_batch*480*640 -> mini_batch*1*480*640
                predicted_tensor = torch.cat((predicted_tensor, predicted_tensor, predicted_tensor),1) # change to 3-channel for visualization, mini_batch*1*480*640
                predicted_images = vutils.make_grid(predicted_tensor, nrow=8, padding=10)
                writer.add_image('Train/predicted_images', predicted_images, accIter['train'])
                predicted_homo_tensor = logits_homo.argmax(1).unsqueeze(1) * scale # mini_batch*args.n_class*480*640 -> mini_batch*480*640 -> mini_batch*1*480*640
                predicted_homo_tensor = torch.cat((predicted_homo_tensor, predicted_homo_tensor, predicted_homo_tensor),1) # change to 3-channel for visualization, mini_batch*1*480*640
                predicted_homo_images = vutils.make_grid(predicted_homo_tensor, nrow=8, padding=10)
                writer.add_image('Train/predicted_homo_images', predicted_homo_images, accIter['train'])
                predicted_tensor2 = labels_homo.unsqueeze(1) * scale # mini_batch*args.n_class*480*640 -> mini_batch*480*640 -> mini_batch*1*480*640
                predicted_tensor2 = torch.cat((predicted_tensor2, predicted_tensor2, predicted_tensor2),1) # change to 3-channel for visualization, mini_batch*1*480*640
                predicted_images2 = vutils.make_grid(predicted_tensor2, nrow=8, padding=10)
                writer.add_image('Train/labels_homo', predicted_images2, accIter['train'])
        accIter['train'] = accIter['train'] + 1

def validation(epo, model, val_loader, writer, accIter, start_datetime):
    losses = AverageMeter()
    model.eval()
    with torch.no_grad():
        for it, (images, labels, names) in enumerate(val_loader):
            images = Variable(images).cuda(args.gpu)
            labels = Variable(labels).cuda(args.gpu)
            start_t = time.time() # time.time() returns the current time
            logits = model(images)
            loss = F.cross_entropy(logits, labels)  # Note that the cross_entropy function has already include the softmax function
            losses.update(loss.item(), images.size(0))

            print('Val: %s, epo %s/%s, iter %s/%s, %.2f img/sec, loss %.4f, time %s' \
                  % (args.model_name, epo, args.epoch_max, it + 1, len(val_loader), len(names)/(time.time()-start_t), float(loss),
                    datetime.datetime.now().replace(microsecond=0)-start_datetime))
            if accIter['val'] % 1 == 0:
                writer.add_scalar('Validation/loss', loss, accIter['val'])
            view_figure = False  # note that I have not colorized the GT and predictions here
            if accIter['val'] % 100 == 0:
                if view_figure:
                    input_rgb_images = vutils.make_grid(images[:, :3], nrow=8, padding=10)  # can only display 3-channel images, so images[:,:3]
                    writer.add_image('Validation/input_rgb_images', input_rgb_images, accIter['val'])
                    scale = max(1, 255 // args.n_class)  # label (0,1,2..) is invisable, multiply a constant for visualization
                    groundtruth_tensor = labels.unsqueeze(1) * scale  # mini_batch*480*640 -> mini_batch*1*480*640
                    groundtruth_tensor = torch.cat((groundtruth_tensor, groundtruth_tensor, groundtruth_tensor), 1)  # change to 3-channel for visualization
                    groudtruth_images = vutils.make_grid(groundtruth_tensor, nrow=8, padding=10)
                    writer.add_image('Validation/groudtruth_images', groudtruth_images, accIter['val'])
                    predicted_tensor = logits.argmax(1).unsqueeze(1)*scale  # mini_batch*args.n_class*480*640 -> mini_batch*480*640 -> mini_batch*1*480*640
                    predicted_tensor = torch.cat((predicted_tensor, predicted_tensor, predicted_tensor), 1)  # change to 3-channel for visualization, mini_batch*1*480*640
                    predicted_images = vutils.make_grid(predicted_tensor, nrow=8, padding=10)
                    writer.add_image('Validation/predicted_images', predicted_images, accIter['val'])
            accIter['val'] += 1
    return {'valid_loss': losses.avg}

def testing(epo, model, test_loader, writer, start_datetime, weight_dir):
    model.eval()
    conf_total = np.zeros((args.n_class, args.n_class))
    # label_list = ["unlabeled", "person", "car"]
    label_list = config.class_names
    testing_results_file = os.path.join(weight_dir, 'testing_results_file.txt')
    with torch.no_grad():
        for it, (images, labels, names) in enumerate(test_loader):
            images = Variable(images).cuda(args.gpu)
            labels = Variable(labels).cuda(args.gpu)
            logits = model(images)
            label = labels.cpu().numpy().squeeze().flatten()
            prediction = logits.argmax(1).cpu().numpy().squeeze().flatten() # prediction and label are both 1-d array, size: minibatch*640*480
            conf = confusion_matrix(y_true=label, y_pred=prediction, labels=[0,1,2,3,4,5,6,7,8]) # conf is args.n_class*args.n_class matrix, vertical axis: groundtruth, horizontal axis: prediction
            conf_total += conf
            print('Test: %s, epo %s/%s, iter %s/%s, time %s' % (args.model_name, epo, args.epoch_max, it+1, len(test_loader),
                 datetime.datetime.now().replace(microsecond=0)-start_datetime))
    precision, recall, IoU, F1 = compute_results(conf_total)
    writer.add_scalar('Test/average_recall', recall.mean(), epo)
    writer.add_scalar('Test/average_IoU', IoU.mean(), epo)
    writer.add_scalar('Test/average_precision',precision.mean(), epo)
    writer.add_scalar('Test/average_F1', F1.mean(), epo)
    for i in range(len(precision)):
        writer.add_scalar("Test(class)/precision_class_%s" % label_list[i], precision[i], epo)
        writer.add_scalar("Test(class)/recall_class_%s"% label_list[i], recall[i], epo)
        writer.add_scalar('Test(class)/Iou_%s'% label_list[i], IoU[i], epo)
        writer.add_scalar('Test(class)/F1_%s'% label_list[i], F1[i], epo)
    if epo==0:
        with open(testing_results_file, 'w') as f:
            f.write("# %s, initial lr: %s, batch size: %s, date: %s \n" %(args.model_name, args.lr_start, args.batch_size, datetime.date.today()))
            f.write("# epoch: unlabeled, car, person, bike, curve, car_stop, guardrail, color_cone, bump,  average(nan_to_num). (Pre %, Acc %, IoU %, F1 %)\n")
    with open(testing_results_file, 'a') as f:
        f.write(str(epo)+': ')
        for i in range(len(precision)):
            f.write('%0.4f, %0.4f, %0.4f, %0.4f ' % (100*precision[i], 100*recall[i], 100*IoU[i], 100*F1[i]))
        f.write('%0.4f, %0.4f, %0.4f, %0.4f\n' % (100*np.mean(np.nan_to_num(precision)), 100*np.mean(np.nan_to_num(recall)), 100*np.mean(np.nan_to_num(IoU)), 100*np.mean(np.nan_to_num(F1)) ))
        #f.write('%0.4f, %0.4f, %0.4f, %0.4f\n' % (100*np.mean(np.nan_to_num(recall)), 100*np.mean(np.nan_to_num(IoU), 100*np.mean(np.nan_to_num(precision)), ))))
    print('saving testing results.')
    with open(testing_results_file, "r") as file:
        writer.add_text('testing_results', file.read().replace('\n', '  \n'), epo)


def main(args):

    prepare_environment(args)

    model = segmodel(cfg=config, encoder_name=config.backbone, decoder_name='MLPDecoder', norm_layer=torch.nn.BatchNorm2d)
    if args.gpu >= 0: 
        model.cuda(args.gpu)

    optimizer, lr_policy = init_optimizer(model, config) 

    train_loader, val_loader, test_loader = setup_data_loaders(config)

    weight_dir, writer = prepare_directories(args)

    start_datetime = datetime.datetime.now().replace(microsecond=0)
    accIter = {'train': 0, 'val': 0}

    for epo in range(args.epoch_from, args.epoch_max):
        print('\ntrain %s, epo #%s begin...' % (args.model_name, epo))

        train(epo, model, train_loader, optimizer, writer, accIter, start_datetime, lr_policy)
        validation(epo, model, val_loader, writer, accIter, start_datetime)

        checkpoint_model_file = os.path.join(weight_dir, str(epo) + '.pth')
        print('saving check point %s: ' % checkpoint_model_file)
        torch.save(model.state_dict(), checkpoint_model_file)

        testing(epo, model, test_loader, writer, start_datetime, weight_dir)

if __name__ == '__main__':
    args = parse_args()
    main(args)
