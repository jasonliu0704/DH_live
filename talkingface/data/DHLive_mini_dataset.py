import numpy as np
import cv2
import tqdm
import copy
import os
from talkingface.utils import *
import glob
import pickle
import torch
import torch.utils.data as data
import logging
from talkingface.models.DINet_mini import input_height, input_width

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

model_size = (256, 256)

def safe_imread(path):
    """Safe image reading with error handling"""
    if not os.path.exists(path):
        logger.error(f"Image file not found: {path}")
        return np.zeros((256, 256, 3), dtype=np.uint8)
    
    try:
        img = cv2.imread(path)
        if img is None:
            logger.error(f"Failed to read image: {path}")
            return np.zeros((256, 256, 3), dtype=np.uint8)
        return img
    except Exception as e:
        logger.error(f"Error reading image {path}: {e}")
        return np.zeros((256, 256, 3), dtype=np.uint8)

def get_image(A_path, crop_coords, input_type, resize=(256, 256)):
    try:
        (x_min, y_min, x_max, y_max) = crop_coords
        
        # Ensure valid crop coordinates
        if x_min >= x_max or y_min >= y_max:
            logger.warning(f"Invalid crop coordinates: {crop_coords}")
            if input_type == 'mediapipe':
                return np.zeros((A_path.shape[0], 2))
            else:
                return np.zeros((*resize, 3), dtype=np.uint8)
                
        size = (max(1, x_max - x_min), max(1, y_max - y_min))  # Avoid division by zero

        if input_type == 'mediapipe':
            if not isinstance(A_path, np.ndarray):
                logger.error(f"Invalid keypoints data type: {type(A_path)}")
                return np.zeros((486, 2))  # Default MediaPipe shape
                
            pose_pts = (A_path - np.array([x_min, y_min])) * np.array(resize) / np.array(size)
            return pose_pts[:, :2]
        else:
            if not isinstance(A_path, np.ndarray):
                logger.error(f"Invalid image data type: {type(A_path)}")
                return np.zeros((*resize, 3), dtype=np.uint8)
                
            h, w = A_path.shape[:2]
            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = min(w, x_max), min(h, y_max)
            
            img_output = A_path[y_min:y_max, x_min:x_max, :]
            img_output = cv2.resize(img_output, resize)
            return img_output
            
    except Exception as e:
        logger.error(f"Error in get_image: {e}")
        if input_type == 'mediapipe':
            return np.zeros((486, 2))  # Default MediaPipe shape
        else:
            return np.zeros((*resize, 3), dtype=np.uint8)

def generate_input(img, keypoints, is_train=False, mode=["mouth_bias"]):
    try:
        if img is None or keypoints is None or len(img.shape) < 3:
            logger.error(f"Invalid input for generate_input: img shape={None if img is None else img.shape}, keypoints={None if keypoints is None else keypoints.shape}")
            return (np.zeros((*model_size, 3), dtype=np.uint8), 
                    np.zeros((*model_size, 3), dtype=np.uint8),
                    [0, 0, model_size[0], model_size[1]])
        
        # Get crop coordinates from keypoints
        crop_coords = crop_mouth(keypoints, img.shape[1], img.shape[0], is_train=is_train)
        target_keypoints = get_image(keypoints[:,:2], crop_coords, input_type='mediapipe', resize=model_size)
        target_img = get_image(img, crop_coords, input_type='img', resize=model_size)

        source_img = copy.deepcopy(target_img)
        source_keypoints = target_keypoints

        source_face_egde = draw_mouth_maps(source_keypoints, im_edges=source_img)
        return source_img, target_img, crop_coords
        
    except Exception as e:
        logger.error(f"Error in generate_input: {e}")
        return (np.zeros((*model_size, 3), dtype=np.uint8), 
                np.zeros((*model_size, 3), dtype=np.uint8),
                [0, 0, model_size[0], model_size[1]])

def generate_ref(img, keypoints, is_train=False, teeth=False):
    try:
        if img is None or keypoints is None:
            logger.error(f"Invalid input for generate_ref")
            return np.zeros((*model_size, 4), dtype=np.uint8)
            
        crop_coords = crop_mouth(keypoints, img.shape[1], img.shape[0], is_train=is_train)
        ref_keypoints = get_image(keypoints, crop_coords, input_type='mediapipe', resize=model_size)
        ref_img = get_image(img, crop_coords, input_type='img', resize=model_size)

        if teeth:
            teeth_mask = np.zeros((model_size[1], model_size[0], 3), np.uint8)
            try:
                pts = ref_keypoints[INDEX_LIPS_INNER, :2]
                if pts.size > 0:  # Check if points exist
                    pts = pts.reshape((-1, 1, 2)).astype(np.int32)
                    cv2.fillPoly(teeth_mask, [pts], color=(1, 1, 1))
                ref_img = ref_img * teeth_mask
            except Exception as e:
                logger.error(f"Error creating teeth mask: {e}")

        ref_face_edge = draw_mouth_maps(ref_keypoints, size=model_size)
        ref_img = np.concatenate([ref_img, ref_face_edge[:,:,:1]], axis=2)
        return ref_img
        
    except Exception as e:
        logger.error(f"Error in generate_ref: {e}")
        return np.zeros((*model_size, 4), dtype=np.uint8)

def select_ref_index(driven_keypoints, n_ref=5, ratio=1/3., ratio2=1):
    try:
        if driven_keypoints is None or len(driven_keypoints) == 0:
            logger.error("Invalid keypoints for reference selection")
            return list(range(n_ref))  # Return default indices
            
        # Calculate valid indices based on available keypoints
        n_frames = len(driven_keypoints)
        if n_frames == 0:
            return list(range(min(n_ref, 10)))  # Fallback to first 10 frames
            
        # Check if keypoints have valid indices
        lips_inner_idx = INDEX_LIPS_INNER
        if lips_inner_idx is None or len(lips_inner_idx) < 10:
            logger.warning("Invalid INDEX_LIPS_INNER, using fallback")
            # Fallback to random selection
            return random.sample(range(n_frames), min(n_ref, n_frames))
            
        try:
            # Calculate lips distance safely
            lips_distance = np.zeros(n_frames)
            for i in range(n_frames):
                if i < len(driven_keypoints) and 5 < len(lips_inner_idx) and -5 < len(lips_inner_idx):
                    if lips_inner_idx[5] < driven_keypoints[i].shape[0] and lips_inner_idx[-5] < driven_keypoints[i].shape[0]:
                        lips_distance[i] = np.linalg.norm(
                            driven_keypoints[i][lips_inner_idx[5]] - driven_keypoints[i][lips_inner_idx[-5]])
        except Exception as e:
            logger.error(f"Error calculating lips distance: {e}")
            return random.sample(range(n_frames), min(n_ref, n_frames))
            
        # Select indices based on lips distance
        lower_bound = max(0, int(n_frames * ratio))
        upper_bound = min(n_frames, int(n_frames * ratio2))
        
        if lower_bound >= upper_bound or lower_bound >= n_frames:
            logger.warning(f"Invalid selection bounds: {lower_bound}-{upper_bound}, n_frames={n_frames}")
            selected_index_list = list(range(n_frames))
        else:
            selected_index_list = np.argsort(lips_distance).tolist()[lower_bound:upper_bound]
            
        # If we don't have enough indices, use whatever we have
        if len(selected_index_list) < n_ref:
            logger.warning(f"Not enough reference candidates: {len(selected_index_list)} < {n_ref}")
            if len(selected_index_list) == 0:
                selected_index_list = list(range(min(n_frames, n_ref)))
            else:
                # Repeat indices if necessary
                while len(selected_index_list) < n_ref:
                    selected_index_list.append(selected_index_list[len(selected_index_list) % len(selected_index_list)])
                    
        # Sample n_ref indices
        ref_img_index_list = random.sample(selected_index_list, min(n_ref, len(selected_index_list)))
        
        # If we still don't have enough, pad with zeros
        while len(ref_img_index_list) < n_ref:
            ref_img_index_list.append(0)
            
        return ref_img_index_list
        
    except Exception as e:
        logger.error(f"Error in select_ref_index: {e}")
        # Fallback to first n_ref frames or fewer if not enough frames
        return list(range(min(n_ref, 10)))

def get_ref_images_fromVideo(cap, ref_img_index_list, ref_keypoints):
    try:
        if cap is None or not cap.isOpened():
            logger.error("Invalid video capture")
            return np.zeros((*model_size, 3 * len(ref_img_index_list)), dtype=np.uint8)
            
        ref_img_list = []
        for index in ref_img_index_list:
            try:
                # Ensure index is valid
                index = max(0, min(int(index), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1))
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                ret, frame = cap.read()
                
                if not ret or frame is None:
                    logger.warning(f"Failed to read frame {index}")
                    # Create empty frame as fallback
                    frame = np.zeros((256, 256, 3), dtype=np.uint8)
                else:
                    frame = frame[:,:,::-1]  # BGR to RGB
                    
                if index >= len(ref_keypoints):
                    logger.warning(f"Keypoint index {index} out of range {len(ref_keypoints)}")
                    # Use first keypoint as fallback
                    keypoint_index = 0
                else:
                    keypoint_index = index
                    
                ref_img = generate_ref(frame, ref_keypoints[keypoint_index])
                ref_img_list.append(ref_img)
            except Exception as e:
                logger.error(f"Error processing reference frame {index}: {e}")
                # Add empty image as fallback
                ref_img_list.append(np.zeros((*model_size, 4), dtype=np.uint8))
                
        if not ref_img_list:
            logger.error("No reference images generated")
            return np.zeros((*model_size, 4), dtype=np.uint8)
            
        try:
            ref_img = np.concatenate(ref_img_list, axis=2)
            return ref_img
        except Exception as e:
            logger.error(f"Error concatenating reference images: {e}")
            # Return first reference image repeated
            if len(ref_img_list) > 0:
                first_img = ref_img_list[0]
                return np.tile(first_img, (1, 1, len(ref_img_list)))
            else:
                return np.zeros((*model_size, 4), dtype=np.uint8)
                
    except Exception as e:
        logger.error(f"Error in get_ref_images_fromVideo: {e}")
        return np.zeros((*model_size, 4 * max(1, len(ref_img_index_list))), dtype=np.uint8)

class Few_Shot_Dataset(data.Dataset):
    def __init__(self, dict_info, n_ref = 2, is_train = False):
        super(Few_Shot_Dataset, self).__init__()
        self.driven_images = dict_info["driven_images"]
        self.driven_keypoints = dict_info["driven_keypoints"]
        self.driving_keypoints = dict_info["driving_keypoints"]

        self.driven_teeth_images = dict_info["driven_teeth_image"]
        self.driven_teeth_rect = dict_info["driven_teeth_rect"]
        self.is_train = is_train

        assert len(self.driven_images) == len(self.driven_keypoints)
        assert len(self.driven_images) == len(self.driving_keypoints)

        self.out_size = (256, 256)

        self.sample_num = np.sum([len(i) for i in self.driven_images])

        # list: 每个视频序列的视频块个数
        self.clip_count_list = []  # number of frames in each sequence
        for path in self.driven_images:
            self.clip_count_list.append(len(path))
        self.n_ref = n_ref

    def get_ref_images(self, video_index, ref_img_index_list):
        try:
            if video_index < 0 or video_index >= len(self.driven_images):
                logger.error(f"Invalid video index: {video_index}")
                video_index = 0
                
            # Reset reference image list
            self.ref_img_list = []
            
            for index_, ref_img_index in enumerate(ref_img_index_list):
                try:
                    # Validate index
                    if ref_img_index < 0 or ref_img_index >= len(self.driven_images[video_index]):
                        logger.warning(f"Invalid reference index {ref_img_index}, clip length {len(self.driven_images[video_index])}")
                        ref_img_index = min(len(self.driven_images[video_index]) - 1, max(0, ref_img_index))
                    
                    image_path = self.driven_images[video_index][ref_img_index]
                    ref_img = safe_imread(image_path)[:, :, ::-1]  # BGR to RGB
                    
                    if video_index >= len(self.driven_keypoints) or ref_img_index >= len(self.driven_keypoints[video_index]):
                        logger.error(f"Keypoint index out of range: video_index={video_index}, ref_img_index={ref_img_index}")
                        # Create empty keypoints as fallback
                        ref_keypoints = np.zeros((468, 3))
                    else:
                        ref_keypoints = self.driven_keypoints[video_index][ref_img_index]
                        
                    if index_ > 0:
                        ref_img = generate_ref(ref_img, ref_keypoints, self.is_train, teeth=True)
                    else:
                        ref_img = generate_ref(ref_img, ref_keypoints, self.is_train)
                        
                    self.ref_img_list.append(ref_img)
                    
                except Exception as e:
                    logger.error(f"Error processing reference image {index_}: {e}")
                    # Add empty reference as fallback
                    self.ref_img_list.append(np.zeros((*model_size, 4), dtype=np.uint8))
                    
            # Ensure we have at least one reference image
            if not self.ref_img_list:
                logger.error("No reference images loaded")
                self.ref_img_list = [np.zeros((*model_size, 4), dtype=np.uint8)]
                
        except Exception as e:
            logger.error(f"Error in get_ref_images: {e}")
            self.ref_img_list = [np.zeros((*model_size, 4), dtype=np.uint8)]

    def __getitem__(self, index):
        try:
            # Handle training mode
            if self.is_train:
                try:
                    video_index = random.randint(0, max(0, len(self.driven_images) - 1))
                    current_clip = random.randint(0, max(0, self.clip_count_list[video_index] - 1))
                    
                    ref_img_index_list = select_ref_index(self.driven_keypoints[video_index], n_ref=self.n_ref-1, ratio=0.33)
                    ref_img_index_list = [random.randint(0, max(0, self.clip_count_list[video_index] - 1))] + ref_img_index_list
                    
                    self.get_ref_images(video_index, ref_img_index_list)
                except Exception as e:
                    logger.error(f"Error in training data selection: {e}")
                    # Use first video and frame as fallback
                    video_index = 0
                    current_clip = 0
                    self.ref_img_list = [np.zeros((*model_size, 4), dtype=np.uint8)]
            else:
                # Testing mode
                video_index = 0
                current_clip = min(index, max(0, self.clip_count_list[video_index] - 1))
                
                if index == 0:
                    try:
                        ref_img_index_list = select_ref_index(self.driven_keypoints[video_index], n_ref=self.n_ref)
                        self.get_ref_images(video_index, ref_img_index_list)
                    except Exception as e:
                        logger.error(f"Error selecting reference images: {e}")
                        self.ref_img_list = [np.zeros((*model_size, 4), dtype=np.uint8)]
            
            # Load target image
            try:
                if video_index >= len(self.driven_images) or current_clip >= len(self.driven_images[video_index]):
                    logger.error(f"Invalid indices: video_index={video_index}, current_clip={current_clip}")
                    target_img = np.zeros((256, 256, 3), dtype=np.uint8)
                else:
                    target_path = self.driven_images[video_index][current_clip]
                    target_img = safe_imread(target_path)[:, :, ::-1]  # BGR to RGB
            except Exception as e:
                logger.error(f"Error loading target image: {e}")
                target_img = np.zeros((256, 256, 3), dtype=np.uint8)

            # Apply transformations for training
            if self.is_train:
                try:
                    # Generate random parameters for augmentation
                    alpha = np.random.uniform(0.8, 1.2)  # Contrast
                    beta = 0  # Brightness
                    h_shift = np.random.randint(-15, 15)  # Hue shift
                    
                    # Apply to target image
                    target_img = cv2.convertScaleAbs(target_img, alpha=alpha, beta=beta)
                    target_img = cv2.cvtColor(target_img, cv2.COLOR_RGB2HSV)
                    target_img[..., 0] = (target_img[..., 0] + h_shift) % 180
                    target_img = cv2.cvtColor(target_img, cv2.COLOR_HSV2RGB)
                    
                    # Apply to reference images
                    for ii in range(len(self.ref_img_list)):
                        if self.ref_img_list[ii].shape[2] >= 3:  # Make sure we have enough channels
                            ref_img = self.ref_img_list[ii][:,:,:3]
                            ref_img = cv2.convertScaleAbs(ref_img, alpha=alpha, beta=beta)
                            ref_img = cv2.cvtColor(ref_img, cv2.COLOR_RGB2HSV)
                            ref_img[..., 0] = (ref_img[..., 0] + h_shift) % 180
                            ref_img = cv2.cvtColor(ref_img, cv2.COLOR_HSV2RGB)
                            
                            if self.ref_img_list[ii].shape[2] > 3:  # Preserve extra channels
                                self.ref_img_list[ii] = np.concatenate([ref_img, self.ref_img_list[ii][:,:,3:]], axis=2)
                            else:
                                self.ref_img_list[ii] = ref_img
                except Exception as e:
                    logger.error(f"Error applying augmentations: {e}")

            # Concatenate reference images
            try:
                if not self.ref_img_list:
                    self.ref_img = np.zeros((*model_size, 4), dtype=np.uint8)
                else:
                    self.ref_img = np.concatenate(self.ref_img_list, axis=2)
            except Exception as e:
                logger.error(f"Error concatenating reference images: {e}")
                self.ref_img = np.zeros((*model_size, 4 * self.n_ref), dtype=np.uint8)

            # Get target keypoints
            try:
                if (video_index < len(self.driving_keypoints) and 
                    current_clip < len(self.driving_keypoints[video_index])):
                    target_keypoints = self.driving_keypoints[video_index][current_clip]
                else:
                    logger.error(f"Keypoint indices out of range: video_index={video_index}, current_clip={current_clip}")
                    target_keypoints = np.zeros((468, 3))
                    
                source_img, target_img, crop_coords = generate_input(target_img, target_keypoints, self.is_train, mode="mouth")
            except Exception as e:
                logger.error(f"Error generating input: {e}")
                source_img = np.zeros((256, 256, 3), dtype=np.uint8)
                target_img = np.zeros((256, 256, 3), dtype=np.uint8)
                crop_coords = [0, 0, 256, 256]

            # Process teeth image
            try:
                safe_rect = [0, 0, 10, 10]  # Default safe rectangle
                if (video_index < len(self.driven_teeth_rect) and 
                    current_clip < len(self.driven_teeth_rect[video_index])):
                    teeth_rect = self.driven_teeth_rect[video_index][current_clip]
                    if len(teeth_rect) >= 4:
                        safe_rect = teeth_rect
                [x_min, y_min, x_max, y_max] = safe_rect
                teeth_img = safe_imread(self.driven_teeth_images[video_index][current_clip])
                ref_face_edge = np.zeros_like(target_img)
                ref_face_edge[int(y_min):int(y_max), int(x_min):int(x_max), 1][
                    np.where(teeth_img[:, teeth_img.shape[1] // 2:, 0] == 0)] = 255
                ref_face_edge[int(y_min):int(y_max), int(x_min):int(x_max), 2][
                    np.where(teeth_img[:, teeth_img.shape[1] // 2:, 0] == 255)] = 255
                teeth_img = get_image(ref_face_edge, crop_coords, input_type='img', resize=model_size)
                source_img[:,:,1][np.where(source_img[:,:,0] == 0)] = teeth_img[:,:,1][np.where(source_img[:,:,0] == 0)]
                source_img[:, :, 2][np.where(source_img[:, :, 0] == 0)] = teeth_img[:, :, 2][np.where(source_img[:, :, 0] == 0)]
            except Exception as e:
                logger.error(f"Error processing teeth image: {e}")

            target_img = cv2.resize(target_img, (128, 128))
            source_img = cv2.resize(source_img, (128, 128))
            self.ref_img = cv2.resize(self.ref_img, (128, 128))

            w_pad = int((128 - input_width) / 2)
            h_pad = int((128 - input_height) / 2)

            target_img = target_img[h_pad:-h_pad, w_pad:-w_pad]/255.
            source_img = source_img[h_pad:-h_pad, w_pad:-w_pad]/255.
            ref_img = self.ref_img[h_pad:-h_pad, w_pad:-w_pad]/255.

            # tensor
            source_tensor = torch.from_numpy(source_img).float().permute(2, 0, 1)
            ref_tensor = torch.from_numpy(ref_img).float().permute(2, 0, 1)
            target_tensor = torch.from_numpy(target_img).float().permute(2, 0, 1)
            return source_tensor, ref_tensor, target_tensor, self.driven_images[video_index][current_clip]
        except Exception as e:
            logger.error(f"Error in __getitem__: {e}")
            return (torch.zeros(3, input_height, input_width), 
                    torch.zeros(3, input_height, input_width), 
                    torch.zeros(3, input_height, input_width), 
                    "")

    def __len__(self):
        if self.is_train:
            return len(self.driven_images)
        else:
            return len(self.driven_images[0])
        # return self.sample_num

def data_preparation(train_video_list):
    img_all = []
    keypoints_all = []
    teeth_img_all = []
    teeth_rect_all = []
    for i in tqdm.tqdm(train_video_list):
        # for i in ["xiaochangzhang/00004"]:
        model_name = i
        img_filelist = glob.glob("{}/image/*.png".format(model_name))
        img_filelist.sort()
        if len(img_filelist) == 0:
            continue
        img_teeth_filelist = glob.glob("{}/teeth_seg/*.png".format(model_name))
        img_teeth_filelist.sort()

        teeth_rect_array = np.loadtxt("{}/teeth_seg/all.txt".format(model_name))
        Path_output_pkl = "{}/keypoint_rotate.pkl".format(model_name)
        with open(Path_output_pkl, "rb") as f:
            images_info = pickle.load(f)

        # print(len(img_filelist), len(images_info), len(img_teeth_filelist), len(teeth_rect_array))
        # exit(1)
        valid_frame_num = min(len(img_filelist), len(images_info), len(img_teeth_filelist), len(teeth_rect_array))

        img_all.append(img_filelist[:valid_frame_num])
        keypoints_all.append(images_info[:valid_frame_num, main_keypoints_index, :2])
        teeth_img_all.append(img_teeth_filelist[:valid_frame_num])
        teeth_rect_all.append(teeth_rect_array[:valid_frame_num])

    print("train size: ", len(img_all))
    dict_info = {}
    dict_info["driven_images"] = img_all
    dict_info["driven_keypoints"] = keypoints_all
    dict_info["driving_keypoints"] = keypoints_all
    dict_info["driven_teeth_rect"] = teeth_rect_all
    dict_info["driven_teeth_image"] = teeth_img_all
    return dict_info
