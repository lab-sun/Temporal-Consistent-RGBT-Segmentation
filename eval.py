import os, argparse, stat, shutil
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from util.MF_dataset import MF_dataset 
from config import config
import cv2
import logging
import numpy as np
from util.util import AverageMeter, intersectionAndUnion, triple_intersectionAndUnion


parser = argparse.ArgumentParser(description='Train with pytorch')
############################################################################################# 
parser.add_argument('--model_name', '-m', type=str, default='CMX_mit_b2')
parser.add_argument('--batch_size', '-b', type=int, default=1)
parser.add_argument('--gpu', '-g', type=int, default=0)
parser.add_argument('--num_workers', '-j', type=int, default=config.num_workers)
parser.add_argument('--n_class', '-nc', type=int, default=config.num_classes)
parser.add_argument('--data_dir', '-dr', type=str, default=config.dataset_path)
parser.add_argument('--pre_weight', '-prw', type=str, default='/pretrained/mit_b2.pth')
parser.add_argument('--backbone', '-bac', type=str, default='mit_b2')
parser.add_argument('--class_name', '-cn', type=str, default=config.class_names)
parser.add_argument('--model_path', '-mp', type=str, default='./TCFuseNet_pth/model_G.pth')
args = parser.parse_args()

def get_corners(K, width, height):
    # Create an array of the four corners in the original image
    corners = np.array([[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]], dtype=np.float32)
    # Add a dimension for homogenous coordinates
    corners = np.hstack((corners, np.ones((4, 1))))
    # Apply the homographic transformation matrix K
    corners = np.dot(K, corners.T).T
    # Normalize the coordinates by dividing by the last element
    corners = corners / corners[:, -1:]
    # Return the coordinates as a numpy array
    return corners

# Define a function to crop and resize the transformed image and logits
def crop_and_resize(image_homo, logits, corners, output_size, width, height):
    # Get the x and y coordinates of the four corners
    x_coords = corners[:, 0]
    y_coords = corners[:, 1]
    # Find the second smallest and second largest values for x and y
    x_min = np.partition(x_coords, 1)[1]
    x_max = np.partition(x_coords, -2)[-2]
    y_min = np.partition(y_coords, 1)[1]
    y_max = np.partition(y_coords, -2)[-2]
    x_min = max(0, int(x_min))
    x_max = min(width - 1, int(x_max))
    y_min = max(0, int(y_min))
    y_max = min(height - 1, int(y_max))
    # Crop the image and logits using the computed values
    image_homo = image_homo[int(y_min):int(y_max), int(x_min):int(x_max), :]
    logits = logits[int(y_min):int(y_max), int(x_min):int(x_max)]
    # Resize the cropped tensors back to original resolution
    image_homo = cv2.resize(image_homo, output_size, interpolation=cv2.INTER_LINEAR)
    logits = cv2.resize(logits, output_size, interpolation=cv2.INTER_NEAREST)
    # Return the cropped and resized tensors
    return image_homo, logits

def homographic_transform(images, logits_homo, K):
    n, _, height, width = images.shape

    # Convert images to numpy array on CPU
    images_np = images.detach().cpu().numpy()
    # Convert logits_homo to numpy array on CPU
    logits_homo_np = logits_homo.detach().cpu().numpy()
    # Convert K to numpy array on CPU
    K_np = K.cpu().numpy()
    
    # Define output size for transformed images
    output_size = (width, height)

    # Initialize output images_homo and logits2
    images_homo = np.zeros_like(images_np)
    logits2 = np.zeros_like(logits_homo_np)
    for i in range(n):
        # Convert PyTorch Tensor to OpenCV image format (uint8 type)
        image_np_uint8 = (images_np[i] * 255).astype(np.uint8).transpose(1, 2, 0)
        logits_homo_np_float32 = logits_homo_np[i].astype(np.float32).transpose(1, 2, 0)
        # Apply homographic transformation using cv2.warpPerspective
        image_homo = cv2.warpPerspective(image_np_uint8, K_np[i], output_size, flags=cv2.INTER_LINEAR)
        logits = cv2.warpPerspective(logits_homo_np_float32, K_np[i], output_size, flags=cv2.INTER_NEAREST)
        
        logits = np.expand_dims(logits, axis=-1)

        # Resize cropped tensors back to original resolution
        image_homo = cv2.resize(image_homo,(width,height),interpolation=cv2.INTER_LINEAR)
        logits = cv2.resize(logits,(width,height),interpolation=cv2.INTER_NEAREST)
    
        # Get the coordinates of the four corners after homographic transformation
        corners = get_corners(K_np[i], width, height)
        # Crop and resize the transformed image and logits
        image_homo, logits = crop_and_resize(image_homo, logits, corners, output_size, width, height)
        
        # Convert back to PyTorch Tensor format (float type in range [0.0 ,1.0])
        images_homo[i] = image_homo.transpose(2, 0, 1) / 255.0
        logits2[i] = logits


    # Convert images_homo and logits2 to PyTorch Tensors and move to GPU
    images_homo = torch.tensor(images_homo).float().cuda()
    logits2 = torch.tensor(logits2).squeeze(1).long().cuda()
    # logits2 = torch.tensor(logits2).squeeze(1).long().cuda()   
    return images_homo, logits2

def euler_to_homography(yaw, pitch, roll, I, n , images):
    R_yaw = torch.zeros((n,3,3), device=images.device)
    R_yaw[:,0,0] = torch.cos(torch.deg2rad(yaw))
    R_yaw[:,0,2] = torch.sin(torch.deg2rad(yaw))
    R_yaw[:,1,1] = 1
    R_yaw[:,2,0] = -torch.sin(torch.deg2rad(yaw))
    R_yaw[:,2,2] = torch.cos(torch.deg2rad(yaw))

    R_pitch = torch.zeros((n,3,3), device=images.device)
    R_pitch[:,0,0] = 1
    R_pitch[:,1,1] = torch.cos(torch.deg2rad(pitch))
    R_pitch[:,1,2] = -torch.sin(torch.deg2rad(pitch))
    R_pitch[:,2,1] = torch.sin(torch.deg2rad(pitch))
    R_pitch[:,2,2] = torch.cos(torch.deg2rad(pitch))

    R_roll = torch.zeros((n,3,3), device=images.device)
    R_roll[:,0,0] = torch.cos(torch.deg2rad(roll))
    R_roll[:,0,1] = -torch.sin(torch.deg2rad(roll))
    R_roll[:,1,0] = torch.sin(torch.deg2rad(roll))
    R_roll[:,1,1] = torch.cos(torch.deg2rad(roll))
    R_roll[:,2,2] = 1

    I = torch.from_numpy(I).to(torch.float32).to(images.device)
    I_inv = torch.inverse(I).to(torch.float32)
    K = I @ (R_roll @ (R_pitch @ R_yaw)) @ I_inv
    return K

def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger

def read_angles(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
        angles = [tuple(map(float, line.strip().split(","))) for line in lines]
    return torch.tensor(angles)

def testing(model, test_loader, angles):
    model.eval()
    TC_intersection_meter = AverageMeter()
    TC_union_meter = AverageMeter()
    TC_target_meter = AverageMeter()
    CA_intersection_meter = AverageMeter()
    CA_union_meter = AverageMeter()
    CA_target_meter = AverageMeter()
    true_intersection_meter = AverageMeter()
    true_union_meter = AverageMeter()
    true_target_meter = AverageMeter()

    with torch.no_grad():
        for it, (images, labels, names) in enumerate(test_loader):
            n = images.shape[0]
            yaw = angles[it*n:(it+1)*n, 0]
            pitch = angles[it*n:(it+1)*n, 1]
            roll = angles[it*n:(it+1)*n, 2]
            fx = 702.6030497884977
            fy = 703.4541726858521
            cx = 320
            cy = 240
            I = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            K = euler_to_homography(yaw, pitch, roll, I, n, images)

            classes = 9
            images = Variable(images).cuda(args.gpu)
            labels = Variable(labels).cuda(args.gpu)
            images_homo, labels_homo = homographic_transform(images, labels.unsqueeze(1), K)
            logits = model(images)
            prediction = logits.data.max(1)[1].squeeze_(1).squeeze_(0).cpu().numpy()

            logits_homo = model(images_homo)
            prediction_homo = logits_homo.data.max(1)[1].squeeze_(1).squeeze_(0).cpu().numpy()
            labels_homo = labels_homo.squeeze_(0).cpu().numpy()

            TC_intersection, TC_union, TC_target = intersectionAndUnion(prediction, prediction_homo, classes)
            TC_intersection_meter.update(TC_intersection)
            TC_union_meter.update(TC_union)
            TC_target_meter.update(TC_target)
            TC_accuracy = sum(TC_intersection_meter.val) / (sum(TC_target_meter.val) + 1e-10)
            logger.info('Evaluating {0}/{1} on image {2}, TC {3:.4f}.'.format(it, len(test_loader)-1, names, TC_accuracy))
                
            CA_intersection, CA_union, CA_target = triple_intersectionAndUnion(prediction, prediction_homo, labels_homo, classes)
            CA_intersection_meter.update(CA_intersection)
            CA_union_meter.update(CA_union)
            CA_target_meter.update(CA_target)
            CA_accuracy = sum(CA_intersection_meter.val) / (sum(CA_target_meter.val) + 1e-10)
            logger.info('Evaluating {0}/{1} on image {2}, CA {3:.4f}.'.format(it, len(test_loader)-1, names, CA_accuracy))

            true_intersection, true_union, true_target = intersectionAndUnion(prediction, labels.squeeze(0).cpu().numpy(), classes)
            true_intersection_meter.update(true_intersection)
            true_union_meter.update(true_union)
            true_target_meter.update(true_target)

    return TC_intersection_meter, TC_union_meter, TC_target_meter, CA_intersection_meter, CA_union_meter, CA_target_meter, true_intersection_meter, true_union_meter, true_target_meter

def calculate_metrics(intersection_meter, union_meter, target_meter):
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    recall_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mRecall = np.mean(recall_class)
    return mIoU, mRecall, iou_class, recall_class

if __name__ == '__main__':
    global logger
    logger = get_logger()

    torch.cuda.set_device(args.gpu)
    print("\nthe pytorch version:", torch.__version__)
    print("the gpu count:", torch.cuda.device_count())
    print("the current used gpu:", torch.cuda.current_device(), '\n')

    config.pretrained_model = config.root_dir + args.pre_weight
    criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=config.background)

    from model.segformer.builder import EncoderDecoder as segmodel
    model = segmodel(cfg=config, encoder_name=args.backbone, decoder_name='MLPDecoder', norm_layer=nn.BatchNorm2d)
    criterion_list = nn.ModuleDict({'ce': criterion})
    model.load_state_dict(torch.load(args.model_path, map_location='cuda:0'))
    if args.gpu >= 0: 
        model.cuda(args.gpu)

    test_dataset = MF_dataset(data_dir=args.data_dir, split='test', input_h=config.image_height, input_w=config.image_width)

    visualize_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    angles_files = ["./angles1.txt", "./angles2.txt", "./angles3.txt"]
    TC_results = []
    CA_results = []
    true_results = []

    for file in angles_files:
        angles = read_angles(file)
        TC_intersection_meter, TC_union_meter, TC_target_meter, CA_intersection_meter, CA_union_meter, CA_target_meter, true_intersection_meter, true_union_meter, true_target_meter = testing(model, visualize_loader, angles)
        
        TC_mIoU, _, TC_iou_class, _ = calculate_metrics(TC_intersection_meter, TC_union_meter, TC_target_meter)
        CA_mIoU, _, CA_iou_class, _ = calculate_metrics(CA_intersection_meter, CA_union_meter, CA_target_meter)
        true_mIoU, true_mRecall, true_iou_class, true_recall_class = calculate_metrics(true_intersection_meter, true_union_meter, true_target_meter)
        
        TC_results.append((TC_mIoU, TC_iou_class))
        CA_results.append((CA_mIoU, CA_iou_class))
        true_results.append((true_mIoU, true_mRecall, true_iou_class, true_recall_class))

    avg_TC_mIoU = np.mean([result[0] for result in TC_results])
    avg_CA_mIoU = np.mean([result[0] for result in CA_results])
    avg_true_mIoU = np.mean([result[0] for result in true_results])
    avg_true_mRecall = np.mean([result[1] for result in true_results])

    class_name = args.class_name
    logger.info('Ave TC result: {:.4f}.'.format(avg_TC_mIoU))
    logger.info('Ave CA result: {:.4f}.'.format(avg_CA_mIoU))
    logger.info('IoU/Recall result: {:.4f}/{:.4f}.'.format(avg_true_mIoU, avg_true_mRecall))

    for i in range(args.n_class):
        avg_TC_iou_class = np.mean([result[1][i] for result in TC_results])
        logger.info('Class_{} TC result: {:.4f}, name: {}.'.format(i, avg_TC_iou_class, class_name[i]))

        avg_CA_iou_class = np.mean([result[1][i] for result in CA_results])
        logger.info('Class_{} CA result: {:.4f}, name: {}.'.format(i, avg_CA_iou_class, class_name[i]))

        avg_true_iou_class = np.mean([result[2][i] for result in true_results])
        avg_true_recall_class = np.mean([result[3][i] for result in true_results])
        logger.info('Class_{} IoU/Recall result: {:.4f}/{:.4f}, name: {}.'.format(i, avg_true_iou_class, avg_true_recall_class, class_name[i]))
