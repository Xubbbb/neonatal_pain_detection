import os
import cv2 as cv
import torch
import dlib

class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, num_levels = 8, num_frames = 20, transform = None, target_transform = None):
        super().__init__()
        self.data_dir = data_dir
        self.num_levels = num_levels
        self.num_frames = num_frames
        self.transform = transform
        self.target_transform = target_transform
        self.video_filename_list = []
        self.levelsIdx_list = []
        
        DS_Store = False
        if ".DS_Store" in os.listdir(self.data_dir):
            DS_Store = True
        self.level_dict = {level_label : idx for idx, level_label in enumerate(sorted(os.listdir(self.data_dir))) if level_label != ".DS_Store"}
        
        for level_label, level_idx in self.level_dict.items():
            level_dir = os.path.join(self.data_dir, level_label)
            for video_filename in sorted(os.listdir(level_dir)):
                if video_filename != ".DS_Store":
                    self.video_filename_list.append(os.path.join(level_label, video_filename))
                    if DS_Store:
                        self.levelsIdx_list.append(level_idx - 1)
                    else:
                        self.levelsIdx_list.append(level_idx)
                
    def __len__(self):
        return len(self.video_filename_list)
    
    def read_video(self, video_path):
        # We will use dlib's face detector to help us complete face alignment
        detector = dlib.get_frontal_face_detector()
        # face landmark predictor
        # predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        
        frames = []
        cap = cv.VideoCapture(video_path)
        total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        stride = total_frames // self.num_frames
        frame_idxs = range(0, total_frames, stride)
        print(f"Reading video {video_path}...")
        for frame_idx in frame_idxs:
            cap.set(cv.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                # Detect faces in the grayscale frame
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                faces = detector(gray, 1)
                #! Assume each frame has one face for simplicity(Maybe we should handle it later to get only neonate's face)
                for face in faces:
                    # Get face alignment result
                    x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
                    face_frame = frame[y1:y2, x1:x2]
                    # Crop face frame into 5 other parts
                    h, w, _ = face_frame.shape
                    crops = [face_frame]
                    
                    # Top left and right corners
                    crops.append(face_frame[0:int(h/2), 0:int(w/2)])
                    crops.append(face_frame[0:int(h/2), int(w/2):w])
                    
                    # Bottom middle
                    crops.append(face_frame[int(h/2):h, int(w/4):int(w*3/4)])
                    
                    # Center regions at different scales
                    for scale in [0.5, 0.75]:
                        center_h, center_w = int(h * scale), int(w * scale)
                        start_h, start_w = int((h - center_h) / 2), int((w - center_w) / 2)
                        crops.append(face_frame[start_h:start_h+center_h, start_w:start_w+center_w])
                    
                    # Resize all crops to 224x224, according to the paper the input size must be 224x224
                    crops = [cv.resize(crop, (224, 224)) for crop in crops]
                    
                    # Apply transform
                    if self.transform:
                        crops = [self.transform(image=crop)['image'] for crop in crops]
                        
                    frames.append(torch.stack(crops, dim=0))
                    break
            else:
                break
        
        cap.release()
        print(f"Read video {video_path} done!")
        return torch.stack(frames, dim=0)
    
    def __getitem__(self, idx):
        levelIdx = self.levelsIdx_list[idx]
        video_filename = self.video_filename_list[idx]
        video_path = os.path.join(self.data_dir, video_filename)
        frames = self.read_video(video_path)
        return frames, levelIdx