import numpy as np
import torch
import cv2

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
def crop_and_resize(image_homo, labels, logits, corners, output_size, width, height):
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
    labels = labels[int(y_min):int(y_max), int(x_min):int(x_max)]
    logits = logits[int(y_min):int(y_max), int(x_min):int(x_max)]
    # Resize the cropped tensors back to original resolution
    image_homo = cv2.resize(image_homo, output_size, interpolation=cv2.INTER_LINEAR)
    labels = cv2.resize(labels, output_size, interpolation=cv2.INTER_NEAREST)
    logits = cv2.resize(logits, output_size, interpolation=cv2.INTER_LINEAR)
    # Return the cropped and resized tensors
    return image_homo, labels, logits

def homographic_transform(images, labels, logits, K):
    n, _, height, width = images.shape

    # Convert images to numpy array on CPU
    images_np = images.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    logits_np = logits.detach().cpu().numpy()
    # Convert K to numpy array on CPU
    K_np = K.cpu().numpy()
    
    # Define output size for transformed images
    output_size = (width, height)

    # Initialize output images_homo and logits2
    images_homo = np.zeros_like(images_np)
    labels_homo = np.zeros_like(labels_np)
    logits_homo = np.zeros_like(logits_np)
    for i in range(n):
        # Convert PyTorch Tensor to OpenCV image format (uint8 type)
        image_np_uint8 = (images_np[i] * 255).astype(np.uint8).transpose(1, 2, 0)
        # Apply homographic transformation using cv2.warpPerspective
        image_homo = cv2.warpPerspective(image_np_uint8, K_np[i], output_size, flags=cv2.INTER_LINEAR)
        label_homo = cv2.warpPerspective(labels_np[i].astype(np.float32).transpose(1, 2, 0), K_np[i], output_size, flags=cv2.INTER_NEAREST)
        logit_homo = cv2.warpPerspective(logits_np[i].transpose(1, 2, 0), K_np[i], output_size, flags=cv2.INTER_LINEAR)

        label_homo = np.expand_dims(label_homo, axis=-1)

        # Resize cropped tensors back to original resolution
        image_homo = cv2.resize(image_homo,(width,height),interpolation=cv2.INTER_LINEAR)
        label_homo = cv2.resize(label_homo,(width,height),interpolation=cv2.INTER_NEAREST)
        logit_homo = cv2.resize(logit_homo,(width,height),interpolation=cv2.INTER_LINEAR)
    
        # Get the coordinates of the four corners after homographic transformation
        corners = get_corners(K_np[i], width, height)
        # Crop and resize the transformed image and label_homo
        image_homo, label_homo, logit_homo = crop_and_resize(image_homo, label_homo, logit_homo, corners, output_size, width, height)
        
        # Convert back to PyTorch Tensor format (float type in range [0.0 ,1.0])
        images_homo[i] = image_homo.transpose(2, 0, 1) / 255.0
        labels_homo[i] = label_homo
        logits_homo[i] = logit_homo.transpose(2, 0, 1)


    # Convert images_homo and labels_homo to PyTorch Tensors and move to GPU
    images_homo = torch.tensor(images_homo).float().cuda()
    labels_homo = torch.tensor(labels_homo).float().cuda()
    logits_homo = torch.tensor(logits_homo).float().cuda()

    return images_homo, labels_homo, logits_homo

def homographic_transform_eval(images, logits_homo, K):
    n, _, height, width = images.shape

    # Convert images to numpy array on CPU
    images_np = images.detach().cpu().numpy()
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
