import cv2
import sys
import yaml
import time
import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from collections import deque
from src.exception import CustomException
from src.logger import logging

# Create a logger instance
logger = logging.getLogger("utils")

def load_config(config_path):
    try:
        with open(config_path, 'r') as configfile:
            config = yaml.safe_load(configfile)
        return config
    except Exception as e:
        logger.error(f"Error Occurred During Loading Configuration File: {e}")
        raise CustomException(e, sys)

'''def load_yolo_model(model_path):
    try:
        model = YOLO(model_path)
        logger.info(f"Model Loaded Successfully From, [{model_path}] Path.")
        
        # For custom models to see the class names
        names = model.names
        logger.info(f"Fine Tuned Model Has [{len(names)}] Classes.")
        logger.info(f"Custom Class Names:")
        for index, name in names.items():
            logger.info(f"Class [{index}]: [{name}]")
            
        return model, names
    
    except Exception as e:
        logger.error(f"Failed To Load The Fine Tune Model From, [{model_path}] Path: {e}")
        raise CustomException(e, sys)'''


def load_yolo_model(model_path):
    try:
        # SOLUTION: Set torch serialization to allow unsafe loading
        # This is needed for PyTorch 2.6+ compatibility with YOLO models
        original_weights_only = torch.serialization.get_default_load_endianness()
        
        # Temporarily disable weights_only for YOLO model loading
        torch.serialization.add_safe_globals([
            torch.nn.modules.container.Sequential,
            torch.nn.modules.conv.Conv2d,
            torch.nn.modules.batchnorm.BatchNorm2d,
            torch.nn.modules.activation.ReLU,
            torch.nn.modules.activation.SiLU,
            torch.nn.modules.pooling.MaxPool2d,
            torch.nn.modules.pooling.AdaptiveAvgPool2d,
            torch.nn.modules.linear.Linear,
            torch.nn.modules.dropout.Dropout,
        ])
        
        # Alternative approach: Monkey patch torch.load temporarily
        original_torch_load = torch.load
        
        def patched_torch_load(*args, **kwargs):
            # Force weights_only=False for YOLO model loading
            kwargs['weights_only'] = False
            return original_torch_load(*args, **kwargs)
        
        # Temporarily replace torch.load
        torch.load = patched_torch_load
        
        try:
            # Now load the YOLO model - this should work without the weights_only error
            model = YOLO(model_path)
            logger.info(f"Model Loaded Successfully From, [{model_path}] Path.")
            
            # For custom models to see the class names
            names = model.names
            logger.info(f"Fine Tuned Model Has [{len(names)}] Classes.")
            logger.info(f"Custom Class Names:")
            for index, name in names.items():
                logger.info(f"Class [{index}]: [{name}]")
                
            return model, names
            
        finally:
            # Always restore the original torch.load function
            torch.load = original_torch_load
    
    except Exception as e:
        logger.error(f"Failed To Load The Fine Tune Model From, [{model_path}] Path: {e}")
        
        # Try alternative approach if the first one fails
        try:
            logger.info("Trying alternative loading method...")
            
            # Method 2: Use context manager for safe globals
            with torch.serialization.safe_globals([
                torch.nn.modules.container.Sequential,
                torch.nn.modules.conv.Conv2d,
                torch.nn.modules.batchnorm.BatchNorm2d,
                torch.nn.modules.activation.ReLU,
                torch.nn.modules.activation.SiLU,
                torch.nn.modules.pooling.MaxPool2d,
                torch.nn.modules.pooling.AdaptiveAvgPool2d,
                torch.nn.modules.linear.Linear,
                torch.nn.modules.dropout.Dropout,
            ]):
                model = YOLO(model_path)
                names = model.names
                logger.info(f"Model Loaded Successfully Using Alternative Method From, [{model_path}] Path.")
                return model, names
                
        except Exception as e2:
            logger.error(f"Alternative loading method also failed: {e2}")
            
            # Method 3: Final fallback - load with explicit weights_only=False
            try:
                logger.info("Trying final fallback method...")
                
                # Monkey patch at module level
                import ultralytics.nn.tasks
                original_load = torch.load
                
                def safe_load(*args, **kwargs):
                    kwargs.pop('weights_only', None)  # Remove if present
                    kwargs['weights_only'] = False
                    return original_load(*args, **kwargs)
                
                torch.load = safe_load
                ultralytics.nn.tasks.torch.load = safe_load
                
                model = YOLO(model_path)
                names = model.names
                
                # Restore original
                torch.load = original_load
                ultralytics.nn.tasks.torch.load = original_load
                
                logger.info(f"Model Loaded Successfully Using Final Fallback From, [{model_path}] Path.")
                return model, names
                
            except Exception as e3:
                logger.error(f"All loading methods failed: {e3}")
                raise CustomException(e, sys)

def capture_video_get_properties(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        logger.info(f"Video Capture Initialized Successfully From, [{video_path}] Path.")
        if not cap.isOpened():
            raise CustomException(f"Error opening video file: {video_path}", sys)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"Video Properties Are: FPS - [{fps}], Height - [{height}], Width - [{width}] And Total Frames - [{total_frames}]")
        return cap, fps, height, width, total_frames
    
    except Exception as e:
        logger.error(f"Failed To Capture Video And Get Properties From, [{video_path}] Path: {e}")
        raise CustomException(e, sys)
    
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        print(f"Mouse position: ({x}, {y})")
        
def get_next_frame(next_frame_index, skip_frames, total_frames, cap, frame_width, frame_height):   
    # Calculate The Actual Frame Number To Read
    target_frame = next_frame_index * (skip_frames + 1)
    # If skip_frames = 2, then it will process every 3rd frame like 0, 3, 6, 9....
   
    if target_frame >= total_frames:
        logger.info(f"Reached End Of The Video. Target Frame [{target_frame}] >= Total Frames [{total_frames}]")
        return None, None
    
    current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    if current_pos != target_frame:
        success = cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        if not success:
            logger.error(f"Failed To Set Video Position To Frame [{target_frame}]")
            return None, None
        
    # Read The Frame
    ret, frame = cap.read()
    if not ret:
        logger.error("Failed To Read The Frames Or End Of Video.")
        return None, None
    
    # Check if frame is valid or not
    if frame is None or frame.size == 0:
        logger.error(f"Invalid Frame Data At Position [{target_frame}].")
        return None, None
    
    # Update Frame Count Only If The Frame Is Successfully Read
    frame_count = target_frame  

    # Resize The Frame
    try:
        resize_frame = cv2.resize(frame, (frame_width, frame_height))
        # Frame Size Cant Not Be Changed, As It Will Cause The Tracking To Fail
        # Becase The Tracking Line Is Drawn Based On The Original Frame Size (800, 500)
    except Exception as e:
        logger.error(f"Error Occurred During Resize The Frame [{target_frame}]: {e}")
        return None, None
    
    return  frame_count, resize_frame  
    
def extract_object_results(track_results):
    ids, boxes, class_ids, conf = [], [], [], []
    
    if track_results is None:
        logger.debug("Track Results Is None. Returning empty Arrays.")
        return  ids, boxes, class_ids, conf
    
    if not isinstance(track_results, (list, tuple)):
        logger.warning(f"Track Results Is Not A List Or Tuple, Got [{type(track_results)}]. Returning empty Arrays.")
        return  ids, boxes, class_ids, conf
    
    if len(track_results) == 0:
        logger.debug("Track Results Is An Empty List. Returning empty Arrays.")
        return  ids, boxes, class_ids, conf
    
    if track_results[0] is None:
        logger.debug("First Element In Track Results Is None. Returning empty Arrays.")
        return  ids, boxes, class_ids, conf

    if not hasattr(track_results[0], 'boxes'):
        logger.warning("Track results[0] Does Not Have 'boxes' Attribute, Returning empty Arrays.")
        return  ids, boxes, class_ids, conf
     
    if track_results[0].boxes is None:
        logger.debug("Track results[0].boxes Is None, Returning empty Arrays.")
        return  ids, boxes, class_ids, conf
    
    if not hasattr(track_results[0].boxes, 'id') or track_results[0].boxes.id is None:
        logger.debug("No Tracking IDs Found In Results, Returning empty Arrays.")
        return  ids, boxes, class_ids, conf

    try:
        boxes_obj = track_results[0].boxes
        
        # Validate all required attributes exist
        required_attributes = ['id', 'xyxy', 'cls', 'conf']
        for attr in required_attributes:
            if not hasattr(boxes_obj, attr) or getattr(boxes_obj, attr) is None:
                logger.warning(f"Missing Or None Attributes [{attr}] In Boxes Object, Returning empty Arrays.")
                return  ids, boxes, class_ids, conf
                
        # Safe Tensor Extraction 
        if (boxes_obj.id.numel() > 0 and boxes_obj.xyxy.numel() > 0 and boxes_obj.cls.numel() > 0
            and boxes_obj.conf.numel() > 0): 

            # Validate tensor shapes before conversion
            if (boxes_obj.id.dim() == 1 and boxes_obj.xyxy.dim() == 2 and boxes_obj.cls.dim() == 1 
                and boxes_obj.conf.dim() == 1 and boxes_obj.xyxy.shape[1] == 4):
                
                tensor_lengths = [ boxes_obj.id.shape[0],
                                    boxes_obj.xyxy.shape[0],
                                    boxes_obj.cls.shape[0],
                                    boxes_obj.conf.shape[0]]
                
                if len(set(tensor_lengths)) != 1:
                    logger.warning(f"Tensor Length Mismatch - [{tensor_lengths}], Returning empty Arrays.")
                    return  ids, boxes, class_ids, conf
                            
                ids = boxes_obj.id.cpu().numpy().astype(int)
                boxes = boxes_obj.xyxy.cpu().numpy().astype(int) 
                class_ids = boxes_obj.cls.int().cpu().tolist()
                conf = boxes_obj.conf.cpu().numpy()

                # Validate arrays are not empty and have valid values
                if (len(ids) > 0 and not any(id_val < 0 for id_val in ids) and 
                    len(boxes) > 0 and all(len(box) == 4 for box in boxes) and
                    len(class_ids) > 0 and all(isinstance(cls_id, int) for cls_id in class_ids) and
                    len(conf) > 0 and all(0 <= c <= 1 for c in conf)): # Minimum conf 0 and max conf 1

                    # Validate All Arrays Have Same Length
                    if len(ids) == len(boxes) == len(class_ids) == len(conf):
                        logger.debug(f"Successfully Extracted [{len(ids)}] Tracked Objects.")
                        return  ids, boxes, class_ids, conf
                    else:  
                        logger.warning("Mismatched Array Lengths In Object Results.")
                        return [], [], [], []
                else:
                    logger.warning("Invalid Tensor Data Detected, Skipping Frame.")
                    return [], [], [], []
            else:
                logger.warning("Invalid Tensor Shapes Detected, Skipping Frame.")
                return [], [], [], []
        else:
            logger.debug("Empty Tensors Detected, Skipping Frame.")
            return [], [], [], []
            
    except Exception as e:
        logger.error(f"Error Occurred During Extract Object Information: {e}")
        raise CustomException(e, sys)

    return ids, boxes, class_ids, conf    

def cleanup_inactive_ids(dictionary, current_frame_ids, current_frame_number, max_frames_missing, max_dictionary_size):
    tracks_to_remove = []
    emergency_remove = []
    try:
        dictionary_keys = list(dictionary.keys()) # Create list of keys first to avoid iteration errors
        logger.debug(f"Starting Cleanup Check For [{len(dictionary_keys)}] Tracked Objects.")
        
        # Remove inactive tracks based on frames missing
        for track_id in dictionary_keys:
            if track_id not in current_frame_ids:
                track_data = dictionary.get(track_id) # Safe access 
                if track_id is None:
                    logger.debug("Track Was Already Removed By Another Process")
                    continue
                
                last_seen = track_data.get('last_seen', current_frame_number) 
                frames_missing = current_frame_number - last_seen
                
                # Mark inactive ids and add to list
                if frames_missing > max_frames_missing:
                    tracks_to_remove.append(track_id)
                    logger.debug(f"Track [{track_id}] Marked For Remove, Which Is Missing For [{frames_missing}] Frames.")

        # Remove inactive ids
        removed_count = 0
        for track_id in tracks_to_remove:
            if track_id in dictionary:
                track_data = dictionary[track_id]
                logger.debug(f"Removing Inactive Track [{track_id}] - Class: {track_data.get('class_name', 'Unknown')}")
                del dictionary[track_id]
                removed_count += 1
         
        if removed_count > 0:
            logger.info(f"Successfully Remove [{removed_count}] Inactive Track From Dictionary.") 
                    
        # Emergency cleanup (for large size)
        if len(dictionary) > max_dictionary_size:
            logger.warning(f"Dictionary Size [{len(dictionary)}] Exceeds Maxium [{max_dictionary_size}] - For That Performing Emergency Cleanup.")
            try:
                sorted_trackes_by_last_seen = sorted(dictionary.items(), key = lambda x: x[1].get('last_seen', 0)) 
                excess_track_count = len(dictionary) - max_dictionary_size
                emergency_remove = [track_id for track_id, _ in sorted_trackes_by_last_seen[:excess_track_count]]
                logger.info(f"Emergency Cleanup Will Remove {len(emergency_remove)} Oldest Tracks")
            
            except Exception as e:
                logger.error(f"Error During Emergency Cleanup Sorting: {e}")
                emergency_remove = list(dictionary.key())[:excess_track_count]
                logger.warning(f"Using Fallback Removal For {len(emergency_remove)} Tracks")
                
            # Remove tracks after sorting is complete
            for track_id in emergency_remove:
                if track_id in dictionary:
                    track_data = dictionary[track_id]
                    logger.debug(f"Emergency Removing Track [{track_id}] - Last Seen: {track_data.get('last_seen', 'Unknown')}")
                    del dictionary[track_id]
                    emergency_remove += 1
                
            if emergency_remove > 0:
                logger.warning(f"Emergency Removed {emergency_remove} Oldest Tracks To Maintain Dictionary Size")
            
        total_removed = removed_count + len(emergency_remove)   
        if total_removed:
            logger.info(f"Cleanup Completed: [{total_removed}] Tracks Removed, [{len(dictionary)}] Tracks Remaining")
       
        return True
    
    except Exception as e:
        logger.error(f"Error Occurred In cleanup_inactive_ids Function: {e}")
        logger.error(f"Dictionary State: [{len(dictionary) if dictionary else 'None'}] Tracks.")
        return False  
        
def update_tracking_dictionary(dictionary, track_id, names, detection_data, class_id, frame_width_for_trajectory,
                               frame_height_for_trajectory, max_history_for_save_in_dictionary):
    if not isinstance(max_history_for_save_in_dictionary, int) or max_history_for_save_in_dictionary <= 0:
        logger.warning(f"Invalid max_history_for_save_in_dictionary Value: [{max_history_for_save_in_dictionary}].")
        max_history_for_save_in_dictionary = 1000
    
    if names is None:
        names = {}
    
    try:
        track_exists = track_id in dictionary
        # Initialize new track if needed
        if not track_exists:
            new_track_data = {
                    'class_id':  detection_data['class_id'],
                    'class_name': names.get(class_id, f'class_{class_id}'),
                    'trajectory': deque(maxlen=max_history_for_save_in_dictionary),
                    'count_status': 'not_counted',
                    'last_seen': detection_data['frame_number'],
                    'confidence_history': deque(maxlen=max_history_for_save_in_dictionary),
                    'bbox_history': deque(maxlen=max_history_for_save_in_dictionary),
                    'direction': None,
                    'line_crossed': False,
                    'crossing_direction': None,
                    'valid_trajectory': deque(maxlen=max_history_for_save_in_dictionary) }
            
            dictionary[track_id] = new_track_data  #  Atomic insertion - either succeeds completely or fails
            logger.debug(f"New Track [{track_id}] Initialiez Successfully With Class '{new_track_data['class_name']}'.")
        
        #Get track data with error checking
        track_data = dictionary.get(track_id)  
        if track_data is None:
            logger.error(f"Track [{track_id}] Disappeared During Update, Possible Race Condition.")
            return False
            
        # Valid and store the center point
        center_point = detection_data['center']
        if (isinstance(center_point, (list, tuple)) and len(center_point) >=2 and isinstance(center_point[0], (int, float)) and
            isinstance(center_point[1], (int, float))):
            
            # Pre validate and store clean coordinates for drawing
            clean_x = max(0, min(int(center_point[0]), frame_width_for_trajectory))
            clean_y = max(0, min(int(center_point[1]), frame_height_for_trajectory))
            
            # Update track data atomically
            try:
                # Deque automatically maintains size
                track_data['trajectory'].append(center_point)
                track_data['valid_trajectory'].append((clean_x, clean_y))
                track_data['confidence_history'].append(detection_data['confidence'])
                track_data['bbox_history'].append(detection_data['bbox'])
                logger.debug(f"Track [{track_id}] Updated. Trajectory Length: [{len(track_data['trajectory'])}.]")
            
            except Exception as e:
                logger.error(f"Error Updating Track Data For [{track_id}]: {e}")
                return False
                
        else:
            logger.warning(f"Invalid Center Points For Track [{track_id}]: [{center_point}]")
            return False
        
        try:
            # Update other data 
            track_data['last_seen'] = detection_data['frame_number']
        except Exception as e:
            logger.error(f"Error Updating Track Data For ['last_seen']: {e}")    

        try: 
            # Update direction calculation
            if len(track_data['trajectory']) >= 2:
                start_point = track_data['trajectory'][0]
                end_point = track_data['trajectory'][-1]
                
                dx = end_point[0] - start_point[0]
                dy = end_point[1] - start_point[1]
                
                if abs(dx) > abs(dy):
                    track_data['direction'] = 'right' if dx > 0 else 'left'
                else:
                    track_data['direction'] = 'down' if dy > 0 else 'up'  
                    
        except Exception as e:
            logger.error(f"Error Calculating Direction For Track [{track_id}]: {e}")
        
        return True 
    
    except Exception as e:
        logger.error(f"Critical Error In update_tracking_dictionary For Track [{track_id}]: {e}")
        return False    
            
def draw_detection_on_frame(frame, names, dictionary, track_id, x1, y1, x2, y2, class_id, confi, cx, cy, 
                            bbox_thickness, font_size_bbox_label, font_thickness_bbox_label, text_label_padding_bbox, text_label_offset,
                            trajectory_thickness, center_point_radious, bgr_blue, bgr_black, bgr_white, bgr_green, bgr_red):
    if frame is None or frame.size == 0:
        raise ValueError("Frame Is Empty.")

    if names is None:
        names = {}

    if dictionary is None:
        dictionary = {}
    
    try:    
        frame_height, frame_width = frame.shape[:2] 
    except Exception as e:
        logger.error(f"Frame Is Not A Valid Numpy Array: {e}")
        raise CustomException(e,sys)
        
    # Over ensure ensure coordinates and other are in the standerd formet for cv2
    try:
        track_id = int(track_id)
        class_id = int(class_id)
    except Exception as e:
        logger.warning(f"Invalid ID Values: {e}")
        return frame
        
    try:    
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cx, cy = int(cx), int(cy)
    except Exception as e:
        logger.warning(f"Invalid Coordinates Values: {e}")
        return frame
        
    try:
        confi = float(confi)
        if not (0 <= confi <= 1):
            logger.warning(f"Condidence Value [{confi}] Is Out Of Valid Range [0 - 1].")
            confi = max(0, min(1, confi)) # Clamp to valid range
    except Exception as e:
        logger.warning(f"Invalid Confidence Values. Using Default 0.5.: {e}")
        confi = 0.5
                   
    # Boundary check for coordinates are in frame bounds
    x1 = max(0, min(x1, frame_width - 1))
    y1 = max(0, min(y1, frame_height - 1))
    x2 = max(0, min(x2, frame_width - 1))
    y2 = max(0, min(y2, frame_height - 1))
    cx = max(0, min(cx, frame_width - 1))
    cy = max(0, min(cy, frame_height - 1))
        
    if x1 >= x2 or y1 >= y2:
        logger.warning(f"Invalid Bounding Box Coordinates For [{track_id}] Track ID.")  
        return frame  
    
    try:      
        # Get The Class Label
        class_name = names.get(class_id, f"Class_{class_id}")
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), bgr_blue, bbox_thickness) 
        
        # Bounding box label
        box_label = f"{class_name} ID:{track_id} Conf:{confi:.2f}"
        
        # Get the text size
        (text_width, text_height), _ = cv2.getTextSize(box_label, cv2.FONT_HERSHEY_SIMPLEX, font_size_bbox_label, font_thickness_bbox_label)
        
        # Calculate label background position with boundary checking
        lab_y_start = max(text_height + text_label_padding_bbox, y1)
        lab_x_end = min(x1 + text_width, frame_width)
        
        # Draw label background
        cv2.rectangle(frame, (x1, lab_y_start - text_height - text_label_padding_bbox), (lab_x_end, lab_y_start), bgr_black, -1) 
        # Draw label text
        cv2.putText(frame, box_label, (x1, lab_y_start - text_label_offset), cv2.FONT_HERSHEY_SIMPLEX, font_size_bbox_label, bgr_white, font_thickness_bbox_label) 
    
    except Exception as e:
        logger.error(f"Unexpected Error During BBOX Drawing For Track [{track_id}]: {e}")
        return frame

    # Draw trajectory trail
    if track_id in dictionary:
        try:
            trajectory_data = dictionary[track_id]
            if isinstance(trajectory_data, dict) and 'valid_trajectory' in trajectory_data:
                valid_trajectory = trajectory_data['valid_trajectory']
                
                # Draw the trajectory line                  
                if valid_trajectory and len(valid_trajectory) > 1:
                    for i in range(1, len(valid_trajectory)):
                        try:
                            cv2.line(frame, valid_trajectory[i-1], valid_trajectory[i], bgr_green, trajectory_thickness) 
                        except Exception as e:
                            logger.warning(f"Failed To draw Trajectory Line Segment [{i}] For Track [{track_id}]: {e}")
                            continue
            else:
                logger.warning(f"Invalid Trajectory Data Structure For [{track_id}] Track ID.")
        
        except Exception as e:
            logger.warning(f"Error Occurred During Access Trajectory For [{track_id}] Track ID.")    

    try:
        # Draw bbox center point
        cv2.circle(frame, (cx, cy), center_point_radious, bgr_red, -1) # BGR format color 'red'
    except Exception as e:
        logger.error(f"Unexpected Error During Center Drawing For Track [{track_id}]: {e}")

    return frame

def check_line_intersect( x1, y1, x2, y2, line_coords):
    line_x1, line_y1, line_x2, line_y2 = line_coords
    
    # Movement vector
    move_dx = x2 - x1
    move_dy = y2 - y1
    
    # Line vector
    line_dx = line_x2 - line_x1
    line_dy = line_y2 - line_y1
    
    # Cross product for intersection 
    denom = (move_dx * line_dy) - (move_dy * line_dx)
    logger.debug(f"Movement Vector: [{move_dx},{move_dy}] - Line Vector: [{line_dx},{line_dy}] - Denominator: [{denom}]. ")
    
    # Check if lines are parallel 
    if abs(denom) < 1e-10:
        logger.debug("Counting Lines Are Parallel, So No Intersection And No Count Will Generate.")
        return False
    
    # Calculate intersection 
    dx_to_line = line_x1 - x1
    dy_to_line = line_y1 - y1    
    t = (dx_to_line * line_dy - dy_to_line * line_dx) / denom
    u = (dx_to_line * move_dy - dy_to_line * move_dx) / denom
    
    # Check if intersection occurs within both line
    intersects = 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0
    
    if intersects:
        intersect_x = x1 + t * move_dx
        intersect_y =  y1 + t * move_dy
        logger.debug(f"Intersection Point: [{intersect_x:.1f},{intersect_y:.1f}].")
    
    logger.debug(f"Intersection Results: [{intersects}].")   
    return intersects

def close_database_connection(db_manager, cleanup_status):
    try:
        logger.info("Closing Database Connection...........")
        if db_manager is not None and hasattr(db_manager, 'close'):
            close_start_time = time.time()
            db_manager.close()
            close_database_duration = time.time() - close_start_time
            logger.info(f"Database Connection Closed In [{close_database_duration:.2f}]s.")
            cleanup_status['database_closed'] = True
        else:
            logger.info("No Database Connection To Close.")
            cleanup_status['database_closed'] = True
            
        return True
    except Exception as e:
        logger.error(f"Error Occurred During Close Database Connection: {e}")
        cleanup_status['database_closed'] = False
        raise CustomException(e, sys)

def release_video_capture(cap, cleanup_status):
    try:
        logger.info("Releasing Video Capture...........")
        
        if cap is not None and hasattr(cap, 'release'):
            release_start_time = time.time()
            cap.release()
            release_video_duration = time.time() - release_start_time
            logger.info(f"Video Capture Released In [{release_video_duration:.2f}]s.")
            cleanup_status['video_capture_released'] = True
        else:
            logger.info("No Video Capture To Release.")
            cleanup_status['video_capture_released'] = True
        
        return True
    except Exception as e:
        logger.error(f"Error Occurred During Release Video Capture: {e}")
        cleanup_status['video_capture_released'] = False
        raise CustomException(e, sys)
    
def close_opencv_windows(cleanup_status, wait_key_value, cleanup_delay):
    try:
        logger.info("Closing OpenCV windows...........")
        close_start_time = time.time()
        cv2.destroyAllWindows()
        cv2.waitKey(wait_key_value)   
        time.sleep(cleanup_delay)  # Small Delay To Allow Window Cleanup
        close_openCV_duration = time.time() - close_start_time
        logger.info(f"OpenCV Windows Successfully Closed In [{close_openCV_duration:.2f}]s.")
        cleanup_status['windows_closed'] = True
        
        return True
    except Exception as e:
        logger.error(f"Error Occurred During Close OpenCV Windows: {e}")
        cleanup_status['windows_closed'] = False
        raise CustomException(e, sys)

def log_cleanup_status(cleanup_status):
    logger.info("CLEANUP RESOURCES STATUS")
    for operation, status in cleanup_status.items():
        status_text = "SUCCESS" if status else "FAILED"
        logger.info(f"{operation}: {status_text}")
    
    # Calculate success rate
    successful_op = sum(cleanup_status.values())
    total_op = len(cleanup_status)
    success_rate = 0
    
    if total_op > 0 :
        success_rate = (successful_op / total_op) * 100
        logger.info(f"Cleanup Success Rate: [{success_rate:.1f}]% ({successful_op}/{total_op})")

        if success_rate < 100:
            logger.warning("Some Cleanup Operations Failed. Check Logs For Details.")
        else:
            logger.info("All Cleanup Operations Completed Successfully.")
    else:
        logger.warning("No Cleanup Operations To Report.")
    
def force_cleanup(cap, db_manager):
    try:
        if cap is not None and hasattr(cap, 'release'):
            cap.release()
            logger.info("Force Release Video Capture.")
        else:
            logger.info("No Video Capture To Release.")
        
    except Exception as e:
        logger.error(f"Failed To Force Release Video Capture: {e}")

    try:
        cv2.destroyAllWindows()
        logger.info("Force Closed OpenCV Window.")
    
    except Exception as e:
        logger.error(f"Failed To Force Closed OpenCV Window: {e}")

    try:
        if db_manager is not None and hasattr(db_manager, 'close'):
            db_manager.close()
            logger.info("Force Closed Database Connection.")
        else:
           logger.info("No Database Connection To Close")  
    
    except Exception as e:
        logger.error(f"Failed To Force Closed Database Connection: {e}")
    
    logger.info("Force Cleanup Completed Successfully.")
    
