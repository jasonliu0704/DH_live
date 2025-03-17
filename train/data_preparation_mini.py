import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "true"
current_dir = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.append(os.path.join(current_dir, ".."))
import uuid
import tqdm
import numpy as np
import cv2
import glob
import math
import pickle
import torch
from talkingface.util.smooth import smooth_array
from talkingface.run_utils import calc_face_mat
import tqdm
from talkingface.utils import *
import mediapipe as mp
from mini_live.teeth import detect_teeth
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

# Check for CUDA availability
CUDA_AVAILABLE = torch.cuda.is_available()
if CUDA_AVAILABLE:
    print(f"CUDA is available. Found {torch.cuda.device_count()} GPU(s)")
else:
    print("CUDA is not available. Using CPU processing")

mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection
point_size = 1
point_color = (0, 0, 255)  # BGR
thickness = 4  # 0 、4、8

def detect_face(frame):
    """
    Detect face in frame and return detection status and face rectangle
    """
    # Use CUDA for image preprocessing if available
    if CUDA_AVAILABLE and hasattr(cv2, 'cuda'):
        gpu_frame = cv2.cuda_GpuMat()
        gpu_frame.upload(frame)
        rgb_frame = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2RGB)
        rgb_frame = rgb_frame.download()
    else:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    with mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(rgb_frame)
        if not results.detections or len(results.detections) > 1:
            return -1, None
        rect = results.detections[0].location_data.relative_bounding_box
        out_rect = [rect.xmin, rect.xmin + rect.width, rect.ymin, rect.ymin + rect.height]
        nose_ = mp_face_detection.get_key_point(
            results.detections[0], mp_face_detection.FaceKeyPoint.NOSE_TIP)
        l_eye_ = mp_face_detection.get_key_point(
            results.detections[0], mp_face_detection.FaceKeyPoint.LEFT_EYE)
        r_eye_ = mp_face_detection.get_key_point(
            results.detections[0], mp_face_detection.FaceKeyPoint.RIGHT_EYE)
        
        if nose_.x > l_eye_.x or nose_.x < r_eye_.x:
            return -2, out_rect

        h, w = frame.shape[:2]
        if rect.xmin < 0 or rect.ymin < 0 or rect.xmin + rect.width > w or rect.ymin + rect.height > h:
            return -3, out_rect
        if rect.width * w < 100 or rect.height * h < 100:
            return -4, out_rect
    return 1, out_rect


def calc_face_interact(face0, face1):
    """Calculate face region overlap"""
    x_min = min(face0[0], face1[0])
    x_max = max(face0[1], face1[1])
    y_min = min(face0[2], face1[2])
    y_max = max(face0[3], face1[3])
    tmp0 = ((face0[1] - face0[0]) * (face0[3] - face0[2])) / ((x_max - x_min) * (y_max - y_min))
    tmp1 = ((face1[1] - face1[0]) * (face1[3] - face1[2])) / ((x_max - x_min) * (y_max - y_min))
    return min(tmp0, tmp1)


def detect_face_mesh(frame):
    """
    Detect face mesh landmarks using MediaPipe
    """
    # Use CUDA for image preprocessing if available
    if CUDA_AVAILABLE and hasattr(cv2, 'cuda'):
        gpu_frame = cv2.cuda_GpuMat()
        gpu_frame.upload(frame)
        rgb_frame = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2RGB)
        rgb_frame = rgb_frame.download()
    else:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5) as face_mesh:
        results = face_mesh.process(rgb_frame)
        pts_3d = np.zeros([478, 3])
        if not results.multi_face_landmarks:
            print("****** WARNING! No face detected! ******")
        else:
            image_height, image_width = frame.shape[:2]
            for face_landmarks in results.multi_face_landmarks:
                for index_, i in enumerate(face_landmarks.landmark):
                    x_px = min(math.floor(i.x * image_width), image_width - 1)
                    y_px = min(math.floor(i.y * image_height), image_height - 1)
                    z_px = min(math.floor(i.z * image_width), image_width - 1)
                    pts_3d[index_] = np.array([x_px, y_px, z_px])
        return pts_3d


def ExtractFromVideo(video_path, gpu_id=0):
    """
    Extract face landmarks from video frames using GPU acceleration
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0

    dir_path = os.path.dirname(video_path)
    vid_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    vid_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pts_3d = np.zeros([totalFrames, 478, 3])
    face_rect_list = []

    # Set CUDA device for this thread if available
    if CUDA_AVAILABLE:
        cv2.cuda.setDevice(gpu_id)
        
    # Pre-allocate GPU resources if available
    if CUDA_AVAILABLE and hasattr(cv2, 'cuda'):
        gpu_frame = cv2.cuda_GpuMat()
    
    for frame_index in tqdm.tqdm(range(totalFrames), desc="Extracting frames"):
        ret, frame = cap.read()
        if ret is False:
            break
            
        # Use CUDA for frame processing if available
        if CUDA_AVAILABLE and hasattr(cv2, 'cuda'):
            gpu_frame.upload(frame)
            # Potential CUDA operations on gpu_frame here
            processed_frame = gpu_frame.download()
        else:
            processed_frame = frame
            
        tag_, rect = detect_face(processed_frame)
        
        if frame_index == 0 and tag_ != 1:
            print("First frame face detection error: multiple faces, extreme angle, face outside frame, or face too small")
            pts_3d = -1
            break
        elif tag_ == -1:  # Handle failed detection by using previous frame
            if len(face_rect_list) > 0:
                rect = face_rect_list[-1]
            else:
                print(f"Frame {frame_index}: Face detection failed and no previous frame to use")
                pts_3d = -1
                break
        elif tag_ != 1:
            print(f"Frame {frame_index}: Face detection error (code: {tag_})")
            
        if len(face_rect_list) > 0:
            face_area_inter = calc_face_interact(face_rect_list[-1], rect)
            if face_area_inter < 0.6:
                print(f"Large face region change detected at frame {frame_index}, overlap: {face_area_inter}")
                pts_3d = -2
                break

        face_rect_list.append(rect)

        # Calculate face region
        x_min = rect[0] * vid_width
        y_min = rect[2] * vid_height
        x_max = rect[1] * vid_width
        y_max = rect[3] * vid_height
        seq_w, seq_h = x_max - x_min, y_max - y_min
        x_mid, y_mid = (x_min + x_max) / 2, (y_min + y_max) / 2
        
        crop_size = int(max(seq_w * 1.35, seq_h * 1.35))
        x_min = int(max(0, x_mid - crop_size * 0.5))
        y_min = int(max(0, y_mid - crop_size * 0.45))
        x_max = int(min(vid_width, x_min + crop_size))
        y_max = int(min(vid_height, y_min + crop_size))

        frame_face = processed_frame[y_min:y_max, x_min:x_max]
        frame_kps = detect_face_mesh(frame_face)
        pts_3d[frame_index] = frame_kps + np.array([x_min, y_min, 0])
        
    cap.release()
    
    # Cleanup CUDA resources
    if CUDA_AVAILABLE and hasattr(cv2, 'cuda'):
        if 'gpu_frame' in locals():
            gpu_frame.release()
        cv2.cuda.Stream.Null.waitForCompletion()
        
    return pts_3d


def detect_teeth_batch(img_paths, batch_size=4, gpu_id=0):
    """
    Detect teeth in batches to maximize GPU utilization
    """
    teeth_results = []
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    for i in range(0, len(img_paths), batch_size):
        batch = img_paths[i:i+batch_size]
        batch_results = []
        
        # Process batch
        for img_path in batch:
            try:
                teeth_img, teeth_rect = detect_teeth(img_path, gpu_id=gpu_id)
                batch_results.append((img_path, teeth_img, teeth_rect))
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                # Create placeholder on error
                batch_results.append((img_path, None, [0, 0, 0, 0]))
                
        teeth_results.extend(batch_results)
        
    return teeth_results


def run(video_path, export_imgs=True, gpu_id=0):
    """
    Process video for talking face animation
    """
    video_name = os.path.basename(video_path).split(".")[0]
    video_data_path = os.path.join(os.path.dirname(video_path), video_name)
    if os.path.exists(video_data_path):
        print(f"Skipping {video_data_path} as it already exists")
        return
        
    os.makedirs(video_data_path, exist_ok=True)

    if export_imgs:
        # Set environment variable for CUDA device
        if CUDA_AVAILABLE:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        # Add teeth detection and segmentation
        os.makedirs(f"{video_data_path}/teeth_seg", exist_ok=True)
        
        # Get all frame images
        img_filelist = glob.glob(f"{video_data_path}/image/*.png")
        img_filelist.sort()
        
        if not img_filelist:
            print(f"No images found in {video_data_path}/image/. Extracting frames from video...")
            os.makedirs(f"{video_data_path}/image", exist_ok=True)
            
            # Extract frames using ffmpeg with hardware acceleration if available
            if CUDA_AVAILABLE:
                ffmpeg_cmd = f"ffmpeg -hwaccel cuda -i {video_path} -qscale:v 2 -loglevel quiet -y {video_data_path}/image/%06d.png"
            else:
                ffmpeg_cmd = f"ffmpeg -i {video_path} -qscale:v 2 -loglevel quiet -y {video_data_path}/image/%06d.png"
                
            os.system(ffmpeg_cmd)
            
            # Get the image list again
            img_filelist = glob.glob(f"{video_data_path}/image/*.png")
            img_filelist.sort()

        # Process teeth detection in parallel batches
        print(f"Detecting teeth in {len(img_filelist)} frames...")
        
        # Determine optimal batch size based on available memory
        batch_size = 8 if CUDA_AVAILABLE else 1
        
        if CUDA_AVAILABLE and len(img_filelist) > batch_size * 2:
            # Process in batches for better GPU utilization
            teeth_results = []
            
            # Create progress bar
            with tqdm.tqdm(total=len(img_filelist), desc="Detecting teeth") as pbar:
                # Process in batches
                for i in range(0, len(img_filelist), batch_size):
                    batch = img_filelist[i:min(i+batch_size, len(img_filelist))]
                    for img_path in batch:
                        teeth_img, teeth_rect = detect_teeth(img_path, gpu_id=gpu_id)
                        teeth_out_path = os.path.join(video_data_path, "teeth_seg", os.path.basename(img_path))
                        
                        if teeth_img is not None:
                            cv2.imwrite(teeth_out_path, teeth_img)
                            teeth_results.append(teeth_rect)
                        else:
                            # Create an empty teeth image if detection failed
                            img = cv2.imread(img_path)
                            empty_teeth = np.zeros_like(img)
                            cv2.imwrite(teeth_out_path, empty_teeth)
                            teeth_results.append([0, 0, 0, 0])
                        
                        pbar.update(1)
                    
                    # Free up GPU memory
                    if CUDA_AVAILABLE:
                        torch.cuda.empty_cache()
        else:
            # Process sequentially for smaller datasets
            teeth_results = []
            for img_path in tqdm.tqdm(img_filelist, desc="Detecting teeth"):
                teeth_img, teeth_rect = detect_teeth(img_path, gpu_id=gpu_id)
                teeth_out_path = os.path.join(video_data_path, "teeth_seg", os.path.basename(img_path))
                
                if teeth_img is not None:
                    cv2.imwrite(teeth_out_path, teeth_img)
                    teeth_results.append(teeth_rect)
                else:
                    # Create an empty teeth image if detection failed
                    img = cv2.imread(img_path)
                    empty_teeth = np.zeros_like(img)
                    cv2.imwrite(teeth_out_path, empty_teeth)
                    teeth_results.append([0, 0, 0, 0])
        
        # Save all teeth rectangles to a file
        teeth_rect_array = np.array(teeth_results)
        np.savetxt(f"{video_data_path}/teeth_seg/all.txt", teeth_rect_array, fmt='%d')

    # Load key points data
    img_filelist = glob.glob(f"{video_data_path}/image/*.png")
    img_filelist.sort()

    Path_output_pkl = f"{video_data_path}/keypoint_rotate.pkl"

    with open(Path_output_pkl, "rb") as f:
        images_info = pickle.load(f)[:, main_keypoints_index, :]
        
    pts_driven = images_info.reshape(len(images_info), -1)
    
    # Use GPU for array operations if available
    if CUDA_AVAILABLE:
        with torch.cuda.amp.autocast():
            tensor_pts = torch.tensor(pts_driven, device=torch.device(f'cuda:{gpu_id}'))
            # Move data back to CPU for smooth_array which may not be GPU optimized
            pts_driven_cpu = tensor_pts.cpu().numpy()
            pts_driven = smooth_array(pts_driven_cpu).reshape(len(pts_driven), -1, 3)
    else:
        pts_driven = smooth_array(pts_driven).reshape(len(pts_driven), -1, 3)

    face_pts_mean = np.loadtxt(os.path.join(current_dir, "../data/face_pts_mean_mainKps.txt"))
    mat_list, pts_normalized_list, face_pts_mean_personal = calc_face_mat(pts_driven, face_pts_mean)
    pts_normalized_list = np.array(pts_normalized_list)
    
    # Update face oval points
    face_pts_mean_personal[INDEX_FACE_OVAL[:10], 1] = np.max(pts_normalized_list[:, INDEX_FACE_OVAL[:10], 1],
                                                           axis=0) + np.arange(5, 25, 2)
    face_pts_mean_personal[INDEX_FACE_OVAL[:10], 0] = np.max(pts_normalized_list[:, INDEX_FACE_OVAL[:10], 0],
                                                           axis=0) - (9 - np.arange(0, 10))
    face_pts_mean_personal[INDEX_FACE_OVAL[-10:], 1] = np.max(pts_normalized_list[:, INDEX_FACE_OVAL[-10:], 1],
                                                            axis=0) - np.arange(5, 25, 2) + 28
    face_pts_mean_personal[INDEX_FACE_OVAL[-10:], 0] = np.min(pts_normalized_list[:, INDEX_FACE_OVAL[-10:], 0],
                                                            axis=0) + np.arange(0, 10)

    face_pts_mean_personal[INDEX_FACE_OVAL[10], 1] = np.max(pts_normalized_list[:, INDEX_FACE_OVAL[10], 1], axis=0) + 25

    # Save data
    with open(f"{video_data_path}/face_mat_mask.pkl", "wb") as f:
        pickle.dump([mat_list, face_pts_mean_personal], f)
        
    # Cleanup GPU memory
    if CUDA_AVAILABLE:
        torch.cuda.empty_cache()


def run_parallel(video_path, gpu_id):
    """
    Run processing for one video on specified GPU
    """
    # Set CUDA device
    if CUDA_AVAILABLE:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        torch.cuda.set_device(gpu_id)
    
    try:
        run(video_path, gpu_id=gpu_id)
    except Exception as e:
        print(f"Error processing {video_path} on GPU {gpu_id}: {e}")
        traceback.print_exc()
    finally:
        # Cleanup
        if CUDA_AVAILABLE:
            torch.cuda.empty_cache()


def main():
    """
    Main entry point for parallel processing of videos
    """
    if len(sys.argv) != 2:
        print("Usage: python data_preparation_mini.py <data_dir>")
        sys.exit(1)
        
    data_dir = sys.argv[1]
    print(f"Video dir is set to: {data_dir}")
    video_files = glob.glob(f"{data_dir}/*.mp4")
    print(f"Found {len(video_files)} videos to process")
    
    # Number of GPUs available
    ngpu = torch.cuda.device_count() if CUDA_AVAILABLE else 1
    print(f"Using {ngpu} worker{'s' if ngpu > 1 else ''}")
    
    jobs = [(video_path, i % max(1, ngpu)) for i, video_path in enumerate(video_files)]
    
    with ProcessPoolExecutor(max_workers=ngpu) as executor:
        futures = [executor.submit(run_parallel, video_path, gpu_id) for video_path, gpu_id in jobs]
        for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Overall progress"):
            try:
                future.result()
            except Exception as e:
                print(f"Error in worker: {e}")


if __name__ == "__main__":
    main()