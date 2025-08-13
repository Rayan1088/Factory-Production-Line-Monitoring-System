# N.B. To run on the local system please "Uncomment" some following codes
# N.B. Comment some line of code of "Streamlit App".
import cv2
import os
import sys
import time
from src.exception import CustomException
from src.logger import logging
from src.databaseManager import DataBaseManagerClass
from src.utils import *

# Create a logger instance
logger = logging.getLogger("tracker_1") 

class TrackerClass1:
     
    def __init__(self, config_path="config.yaml"):
        
        # Load configuration from YAML file
        self.config = load_config(config_path)
      
        if not os.path.exists(self.config['MODEL_PATH']):
            raise FileNotFoundError(f"Model File [{self.config['MODEL_PATH']}] Not Found From [TrackerClass1].")
    
        if not os.path.exists(self.config['VIDEO_SOURCE_1']):
            raise FileNotFoundError(f"Video File [{self.config['VIDEO_SOURCE_1']}] Not Found From [TrackerClass1].")

        if self.config['SKIP_FRAMES_FOR_TRACKER_1'] < 0:
            raise ValueError("skip_frames Must Be Greater Than Or Equal To '0' From [TrackerClass1].")
        
        # Initialize model and classes
        self.model = None   
        self.names = {}
        
        # Initialize frame processing
        self.cap = None
        self.total_frames = 0
        self.current_frame_number = 0 
        self.frames_processed_count = 0
        self.next_frame_index = 0    
        
        # Initialize tracking 
        self.frames_since_last_save = 0 
        self.save_every_n_frames = 0
        
        # Initialize cleanup status 
        self.status = {
            'database_save_completed': False,
            'database_closed': False,
            'video_capture_released': False,
            'windows_closed': False,
            'database_ready': False,
            'database_save': False,
            'database_close': False,
            'video_capture_released': False,
            'windows_closed': False,
        }
         
        # Initialize dictionary and count
        self.dictionary = {}
        self.box_in_count = 0
        self.box_out_count = 0
        
        # Initialize time and database 
        self.last_save_time = time.time() 
        try:
            self.db_manager = DataBaseManagerClass(self.config['DB_PATH_1'], self.config['DB_PATH_2'], self.config['LIMIT'])
            self.status['database_ready'] = True
        except Exception as e:
            logger.error(f"Error Initializing Database Manager From [TrackerClass1]: {e}")
            self.status['database_ready'] = False
            raise CustomException(e, sys)
        
        # Get model and cap the frames
        try:
            self.model, self.names = load_yolo_model(self.config['MODEL_PATH'] )
            self.cap, self.fps, self.height, self.width, self.total_frames = capture_video_get_properties(self.config['VIDEO_SOURCE_1']) 
            
            if self.cap is None:
                raise ValueError("Failed To Open Video Capture From From [TrackerClass1].")
            if self.total_frames <= 0:
                raise ValueError("No Frames Detected - 'Invalid Video' From [TrackerClass1].")
            if self.fps <= 0:
                self.fps = self.config['DEFAULT_FPS']  # Default FPS from config
                logger.warning(f"Invalid FPS Detected Using Default Value: [{self.fps}] From [TrackerClass1].")
                     
            # Calculate save interval in frames
            if self.fps and self.fps > 0:
                self.save_every_n_frames = int(self.fps * self.config['DATABASE_SAVE_TIME_INTERVAL']) 
                logger.info(f"Auto Save Will Trigger Every [{self.save_every_n_frames}] Frames Every [{self.config['DATABASE_SAVE_TIME_INTERVAL']}] Sec From [TrackerClass1].")
            else:
                raise ValueError(f"Invalid FPS Value: [{self.fps}] For [TrackerClass1]. FPS Value Must Be Greater Then 0 To Proceed.")
                
        except Exception as e:
            logger.error(f"Error Initializing Model Or Video Capture From [TrackerClass1]: {e}")
            raise CustomException(e, sys)
        
        # self.mouse_callback = mouse_callback
        # cv2.namedWindow("Camera 01/Video 01")
        # cv2.setMouseCallback("Camera 01/Video 01", self.mouse_callback)
         
    def check_line_crossing(self, track_id):
        if track_id not in self.dictionary:
            logger.debug(f"Track ID [{track_id}] Not In Dictionary For [TrackerClass1].")
            return
        track_data = self.dictionary[track_id]
        
        # Count if not already counted
        if track_data.get('count_status') == 'counted':
            logger.debug(f"Track ID[{track_id}] Already Counted From [TrackerClass1].")
            return
        
        # Get the trajectory point
        trajectory = track_data.get('trajectory', [])
        logger.debug(f"Track ID [{track_id}] Trajectory Length: [{len(trajectory)}] From [TrackerClass1].")
        
        if len(trajectory) < 2:
            logger.debug(f"Track ID [{track_id}] Insufficient Trajectory: [{trajectory}] From [TrackerClass1].")
            return
        
        # Get previous and current positions
        prev_x, prev_y = trajectory[-2] 
        curr_x, curr_y = trajectory[-1]
        
        # Calculate movement distance and direction
        dx = curr_x - prev_x
        dy = curr_y - prev_y
        movement_distance = (dx**2 + dy**2)**0.5
        logger.debug(f"Track ID [{track_id}] Movement: [{prev_x},{prev_y}] -> [{curr_x},{curr_y}] From [TrackerClass1]")
        logger.debug(f"Track ID {track_id} Delta: dx = [{dx}], dy = [{dy}], Distance=[{movement_distance:.2f}] From [TrackerClass1]")
        
        # skip if movement is too small 
        if movement_distance < self.config['MINIMUM_MOVEMENT_THRESHOLD']:
            logger.debug(f"Track ID [{track_id}] Movement Too Small: [{movement_distance:.2f}] From [TrackerClass1].")
            return
        lines = [("Counting Line 1", self.config['HORIZONTAL_LINE_1_FOR_TRACKER_1']),
                 ("Counting Line 2", self.config['HORIZONTAL_LINE_2_FOR_TRACKER_1']), 
                 ("Counting Line 3", self.config['HORIZONTAL_LINE_3_FOR_TRACKER_1'])]
        
        # Check line crossing
        for line_name, line_coords in lines:
            x1, y1, x2, y2 = line_coords
            logger.debug(f"Checking [{line_name}]: [{x1},{y1}] -> [{x2},{y2}.] From [TrackerClass1]")
            
            # Check if line intersects with movement 
            intersects = check_line_intersect(prev_x, prev_y , curr_x, curr_y , line_coords)
            
            if intersects:
                logger.info(f"Intersection Found For Track ID: [{track_id}] Crossed [{line_name}] From [TrackerClass1].")
                
                # For horizontal lines, determine vertical movement direction
                for_horizontal_line = abs(x2 - x1) > abs(y2 - y1)
                if for_horizontal_line:
                    # Vertcal movement across horiozontal line
                    if dy > 0:
                        direction = 'down'
                    elif dy < 0:
                        direction = 'up' 
                    else:
                        logger.warning(f"Track ID [{track_id}] Has No Vertical Movement Detected From [TrackerClass1].")
                else:
                    # Horiozontal movement across vertcal line
                    if dx > 0:
                        direction = 'right'
                    elif dx < 0:
                        direction = 'left' 
                    else:
                        logger.warning(f"Track ID [{track_id}] Has No Horiozontal Movement Detected From [TrackerClass1].")
                    
                logger.info(f"Track ID [{track_id}] And Direction [{direction}] From [TrackerClass1].")    
                    
                # Count based on direction
                if direction == 'up':
                    self.box_in_count +=1
                    track_data['count_status'] = 'counted'
                    track_data['crossing_direction'] = 'up'
                    track_data['crossing_line'] = line_name
                    logger.info(f"Track ID [{track_id}] Crossed Up On [{line_name}]. Total Up: [{self.box_in_count}] From [TrackerClass1].")
                    break
                elif direction == 'down':
                    self.box_out_count += 1
                    track_data['count_status'] = 'counted'
                    track_data['crossing_direction'] = 'down'
                    track_data['crossing_line'] = line_name
                    logger.info(f"Track ID [{track_id}] Crossed Down On [{line_name}]. Total Down: [{self.box_out_count}] From [TrackerClass1].")
                    break
                else:
                    logger.warning(f"Track ID [{track_id}] And Direction [{direction}] Not Handled For Counting From [TrackerClass1].")
            else:
                logger.debug(f"Track ID [{track_id}] Did Not Intersect [{line_name}] From [TrackerClass1].")        
                    
    def processing_tracking_results(self, frame, names, dictionary, frame_count, ids, boxes, class_ids, conf):
        current_frame_ids = set()
        logger.debug(f"Processing [{len(boxes)}] Detections In Frame [{frame_count}] From [TrackerClass1].")
        
        for track_id, box, class_id, confi in zip(ids, boxes, class_ids, conf):
            try:
                x1, y1, x2, y2 = box
                current_frame_ids.add(track_id)
                
                # Convert numpy types
                if hasattr(x1, 'item'):
                    x1, y1, x2, y2 = x1.item(), y1.item(), x2.item(), y2.item()
                
                if x2 > x1 and y2 > y1:
                    # Format: [x1, y1, x2, y2]
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                else:
                    # Format: [x, y, width, height]
                    cx = int(x1 + x2/2)
                    cy = int(y1 + y2/2)
                
                # Update the tracking dictionary
                update_tracking_dictionary(self.dictionary,
                                           track_id, 
                                           self.names,
                                           {'bbox': (x1, y1, x2, y2),'center': (cx, cy),'confidence': confi, 'class_id': class_id,'frame_number': frame_count},
                                           class_id,
                                           self.config['FRAME_WIDTH_FOR_TRAJECTORY'],
                                           self.config['FRAME_HEIGHT_FOR_TRAJECTORY'],
                                           self.config['MAX_HISTORY_FOR_SAVE_IN_DICTIONARY'])
                
                # Check line crossing or not 
                self.check_line_crossing(track_id)
                
                # Draw the detection on the each frame
                draw_detection_on_frame(frame, names, dictionary, track_id, x1, y1, x2, y2, class_id, confi, cx, cy,
                                        self.config['BBOX_THICKNESS'], self.config['FONT_SIZE_BBOX_LABEL'], self.config['FONT_THICKNESS_BBOX_LABEL'], 
                                        self.config['TEXT_LABEL_PADDING_BBOX'], self.config['TEXT_LABEL_OFFSET'], self.config['TRAJECTORY_THICKNESS'], 
                                        self.config['CENTER_POINT_RADIUS'], self.config['BGR_BLUE'], self.config['BGR_BLACK'], self.config['BGR_WHITE'], 
                                        self.config['BGR_GREEN'], self.config['BGR_RED'])
                              
            except Exception as e:
                logger.error(f"Error Occurred During Process Detection For [{track_id}] Track ID: {e} From [TrackerClass1].")    
        
        logger.debug(f"Processed [{len(current_frame_ids )}] Tracks: [{current_frame_ids}] From [TrackerClass1].")  
        return current_frame_ids       
    
    def draw_counting_line(self, frame):
        lines = [self.config['HORIZONTAL_LINE_1_FOR_TRACKER_1'],
                 self.config['HORIZONTAL_LINE_2_FOR_TRACKER_1'],
                 self.config['HORIZONTAL_LINE_3_FOR_TRACKER_1']]
        
        for idx, line in enumerate(lines, start=1): 
            cv2.line(frame, (line[0], line[1]), (line[2], line[3]), self.config['BGR_RED'], self.config['COUNTING_LINE_THICKNESS']) 
            cv2.putText(frame, f"Counting Line {idx:02d}", (line[0] + self.config['X_TEXT_LABEL_PADDING_COUNTING_LINE'], 
                                                            line[1] - self.config['Y_TEXT_LABEL_PADDING_COUNTING_LINE']), 
                                                            cv2.FONT_HERSHEY_SIMPLEX, self.config['FONT_SIZE_COUNTING_LINE'],
                                                            self.config['BGR_RED'], self.config['FONT_THICKNESS_COUNTING_LINE']) 

    def add_text_on_frame(self, frame):
        # Count info
        cv2.putText(frame, f"Up Count: [{self.box_in_count}]", self.config['TEXT_POSITION_IN_COUNT'], cv2.FONT_HERSHEY_SIMPLEX,  
                    self.config['FONT_SIZE_IN_OUT_TOTAL'], self.config['BGR_RED'], self.config['FONT_THICKNESS_UP_DOWN_TOTAL']) 
        
        cv2.putText(frame, f"Down Count: [{self.box_out_count}]", self.config['TEXT_POSITION_OUT_COUNT'], cv2.FONT_HERSHEY_SIMPLEX, 
                    self.config['FONT_SIZE_IN_OUT_TOTAL'], self.config['BGR_RED'], self.config['FONT_THICKNESS_UP_DOWN_TOTAL'])
        
        cv2.putText(frame, f"Total Count: [{self.box_in_count + self.box_out_count}]", self.config['TEXT_POSITION_TOTAL_COUNT'], 
                    cv2.FONT_HERSHEY_SIMPLEX,  self.config['FONT_SIZE_IN_OUT_TOTAL'], self.config['BGR_RED'], self.config['FONT_THICKNESS_UP_DOWN_TOTAL'])
        # Frame info
        cv2.putText(frame, f"Frame: [{self.current_frame_number} / {self.total_frames}]", self.config['TEXT_POSITION_FRAME'], cv2.FONT_HERSHEY_SIMPLEX,
                    self.config['FONT_SIZE_FRAME_PRO_TRACK'], self.config['BGR_GREEN'], self.config['FONT_THICKNESS_FRAME_PRO_TRACK']) 
        
        cv2.putText(frame, f"Frame Processed: [{self.frames_processed_count}]", self.config['TEXT_POSITION_FRAME_PROCESSED'], cv2.FONT_HERSHEY_SIMPLEX,
                    self.config['FONT_SIZE_FRAME_PRO_TRACK'], self.config['BGR_GREEN'], self.config['FONT_THICKNESS_FRAME_PRO_TRACK'])
        
        cv2.putText(frame, f"Tracked Objects: [{len(self.dictionary)}]", self.config['TEXT_POSITION_TRACKED_OB'], cv2.FONT_HERSHEY_SIMPLEX, 
                    self.config['FONT_SIZE_FRAME_PRO_TRACK'], self.config['BGR_GREEN'], self.config['FONT_THICKNESS_FRAME_PRO_TRACK'])
        
        # Key info
        cv2.putText(frame, "ESC: Exit | SPACE: Pause | R: Reset | S: Save", (self.config['TEXT_LABEL_PADDING_INPUT_KEY'], 
                    frame.shape[0] - self.config['TEXT_LABEL_PADDING_INPUT_KEY']), cv2.FONT_HERSHEY_SIMPLEX, self.config['FONT_SIZE_INPUT_KEY'], 
                    self.config['BGR_GREEN'], self.config['FONT_THICKNESS_INPUT_KEY'])
    
    def process_each_frame(self, model, frame, names, dictionary, frame_count):
        current_frame_ids = set()
        try:
            # Track Objects (For Box Class = 0 And Cement Bag Class =1)
            track_results = model.track(frame, persist=True, classes=[0])
            # Extract tracking information
            ids, boxes, class_ids, conf = extract_object_results(track_results) 
        
        except Exception as e:
            logger.error(f"Error Occurred During Tracking Objects From [TrackerClass1]: {e}")
            cleanup_inactive_ids(self.dictionary,
                                 current_frame_ids,
                                 self.current_frame_number,
                                 self.config['MAX_FRAMES_MISSING'],
                                 self.config['MAX_DICTIONARY_SIZE'])
            return
        
        # Processing track results 
        try:
            current_frame_ids = self.processing_tracking_results(frame, names, dictionary, frame_count, ids, boxes, class_ids, conf)
        except Exception as e:
            logger.error(f"Error Occurred During Process Tracking Results From [TrackerClass1]: {e}")
        finally:
            cleanup_inactive_ids(self.dictionary,
                                 current_frame_ids,
                                 self.current_frame_number,
                                 self.config['MAX_FRAMES_MISSING'],
                                 self.config['MAX_DICTIONARY_SIZE'])

        # Draw counting line
        try:
            self.draw_counting_line(frame)
        except Exception as e:
            logger.error(f"Error Occurred During Draw The Counting Line From [TrackerClass1]: {e}")
        
        # Add counting info over the frame
        try:
            self.add_text_on_frame(frame) 
        except Exception as e:
            logger.error(f"Error Occurred During Add The Text On The Frame From [TrackerClass1]: {e}")
        
        # Show The Frame
        # try:
            # cv2.imshow("Camera 01/Video 01", frame)
        # except Exception as e:
            # logger.error(f"Error Displaying Frame From [TrackerClass1]: {e}")
            # raise CustomException(e, sys) 

        return current_frame_ids 
     
    def save_and_reset_counts(self):
        objects_removed = 0 
        try:
            if self.box_in_count <= 0 and self.box_out_count <= 0:
                logger.info("No Counts To Save During Save And Reset Counts From [TrackerClass1].")
                return
            
            # Check databse is ready or not
            if not self.status['database_ready']:
                try:
                    if (self.db_manager is not None and 
                        (hasattr(self.db_manager, 'is_connected') and 
                         self.db_manager.is_connected())):
                        self.status['database_ready'] = True
                    else:
                        self.status['database_ready'] = False
                        
                except Exception as e:
                    logger.error(f"Database Manager Not Ready For Save And Reset Counts From [TrackerClass1]: {e}")
                    self.status['database_ready'] = False
                
            if not self.status['database_ready']:
                logger.error("Database Manager Not Ready, Skipping Save And Reset Counts [TrackerClass1].")
                return
            
            # Save Counts To Database If Ready
            in_count_to_save = self.box_in_count
            out_count_to_save = self.box_out_count
            
            # Collect counted object ids
            counted_objects = []
            for track_id, track_data in self.dictionary.items():
                if track_data['count_status'] == 'counted':
                    counted_objects.append(track_id)
            logger.info(f"Saving Counts: In: [{in_count_to_save}], Out: [{out_count_to_save}] From [TrackerClass1].")

            # Save counts to database
            try:
                save_start_time = time.time()
                self.db_manager.save_box_counts(in_count=in_count_to_save, out_count=out_count_to_save)
                save_duration = time.time() - save_start_time 
                logger.info(f"Database Save Completed In [{save_duration:.2f}]s From [TrackerClass1].")
                
            except Exception as e:
                logger.error(f"Error Occurred During Database Save And Reset Counts From [TrackerClass1]: {e}")
                raise CustomException(e, sys)
            
            # Reset count after successfully save 
            self.box_in_count = 0
            self.box_out_count = 0
            
            # Remove counted objecrs from dictionary
            for track_id in counted_objects:
                if track_id in self.dictionary:
                    del self.dictionary[track_id]
                    objects_removed += 1

            self.last_save_time = time.time()
            logger.info(f"Count Save And Reset Successfully And Removed [{objects_removed}] Objects From [TrackerClass1] Dictionary.")
                            
        except Exception as e:
            logger.error(f"Error In save_and_reset_counts() Function From [TrackerClass1]: {e}")
            raise CustomException(e, sys)
    
    def reset_counts_manually(self):
        try:
            old_in_count = self.box_in_count
            old_out_count = self.box_out_count
            old_dict_size = len(self.dictionary)
            
            self.box_in_count = 0
            self.box_out_count = 0
            self.dictionary.clear()
            logger.info(f"Manual Reset Completed And Cleared: In={old_in_count}, Out={old_out_count}, Tracked={old_dict_size} From [TrackerClass1].")
        
        except Exception as e:
            logger.error(f"Error Occurred During Manually Reset Counts From [TrackerClass1]: {e}")
            try:
                # Force reset
                self.box_in_count = 0
                self.box_out_count = 0
                self.dictionary = {} # Create new empty dictionary
                logger.warning("Force Reset Completed After Error Occurred From [TrackerClass1].")
            except Exception as e:
                logger.critical(f"Force Reset Also Failed From [TrackerClass1]: {e}")
                raise CustomException(e, sys) 

    def keyboard_input(self):
        try:
            key = cv2.waitKey(self.config['OPENCV_WAIT_KEY_DELAY']) & 0xFF
            
            if key == 27: # ESC
                logger.info("ESC Key Pressed, Exiting Video Processing For [TrackerClass1].")
                return False
            
            elif key == 32: # Space
                logger.info("SPACE Key Pressed, Pausing Video Processing For [TrackerClass1].")
                cv2.waitKey(0)
                return True
            
            elif key == ord("s") or key == ord("S"):
                logger.info("S Key Pressed, Performing Manual Save From [TrackerClass1].")
                try:
                    self.save_and_reset_counts() 
                    logger.info("Manual Save Completed Successfully From [TrackerClass1].")
                except Exception as e:
                    logger.error(f"Manual Save Failed From [TrackerClass1]: {e}")
                return True
            
            elif key == ord("r") or key == ord("R"):
                logger.info("R Key Pressed, Performing Manual Reset From [TrackerClass1].")
                try:
                    self.reset_counts_manually()
                    logger.info("Manual Reset Completed Successfully From [TrackerClass1].")
                except Exception as e:
                    logger.error(f"Manual Reset Failed From [TrackerClass1]: {e}")
                return True
            
            return True
        except Exception as e:
            logger.error(f"Error In Keyboard Input Handling From [TrackerClass1]: {e}")
            return True
        
    def database_save_perform(self):
        logger.info("Performing Final Save Operation From [TrackerClass1].")
        try:
            if self.box_in_count > 0 or self.box_out_count > 0:
                save_start_time = time.time()
                self.db_manager.save_box_counts(in_count=self.box_in_count, out_count=self.box_out_count)
                save_duration = time.time() - save_start_time
                logger.info(f"Final Save Completed In [{save_duration:.2f}]s Total In: [{self.box_in_count}], Total Out: [{self.box_out_count}] From [TrackerClass1].")
                self.box_in_count = 0
                self.box_out_count = 0
                self.status['database_save_completed'] = True
            else:
                logger.info("No Counts Save During Final Save From [TrackerClass1].")
                self.status['database_save_completed'] = True
                
        except Exception as e:
            logger.error(f"Error Occurred During Final Save From [TrackerClass1].: {e}")
            self.status['database_save_completed'] = False
            raise CustomException(e, sys)
        
    def cleanup_for_tracker1(self):
        # Perform Final Save
        try:
            self.database_save_perform()
            self.status['database_save'] = True
            logger.info("Database Save Completed Successfully From [TrackerClass1].")
        except Exception as e:
            logger.error(f"Error Occurred During Database Save From [TrackerClass1]: {e}")
            self.status['database_save'] = False
            
        # Close Database Connection With Timeout
        try:
            close_database_connection(self.db_manager, self.status)
            self.status['database_close'] = True
            logger.info("Database Closed Successfully From [TrackerClass1].")
        except Exception as e:
            logger.error(f"Error Occurred During Close Database Connection From [TrackerClass1]: {e}")
            self.status['database_close'] = False
            
        # Release Video Capture Resources
        try:
            release_video_capture(self.cap, self.status)
            self.status['video_capture_released'] = True
            logger.info("Release Video Capture Successfully From [TrackerClass1].")
        except Exception as e:
            logger.error(f"Error Occurred During Release Video Capture From [TrackerClass1]: {e}")
            self.status['video_capture_released'] = False

        # Close OpenCV windows
        # try:
            # close_opencv_windows(self.status, self.config['OPENCV_WAIT_KEY_DELAY'], self.config['WINDOW_CLEANUP_DELAY'])
            # self.status['windows_closed'] = True
            # logger.info("Close Opencv Windows Successfully From [TrackerClass1].")
        # except Exception as e:
            # logger.error(f"Error Occurred During Close OpenCV Windows From [TrackerClass1]: {e}")
            # self.status['windows_closed'] = False
        
        # No windows to close in Streamlit mode
        self.status['windows_closed'] = True # Comment this line when run the code in local system
            
        try:
            log_cleanup_status(self.status)
        except Exception as e:
            logger.warning(f"Failed To Log Cleanup Status From [TrackerClass1]: {e}")
            
        logger.info("Cleanup Completed Successfully From [TrackerClass1]")
        
    # process The Video Frame By Frame For Object Detection And Tracking
    def detection_and_tracking_1_for_local_system(self):
        current_frame_ids = set()
        cleanup_performed = False
        try:
            logger.info(f"Starting Video Processing. Total Frames: [{self.total_frames}] From [TrackerClass1]")
            while True:
                frame_number, frame = get_next_frame(self.next_frame_index,
                                                     self.config['SKIP_FRAMES_FOR_TRACKER_1'],
                                                     self.total_frames, 
                                                     self.cap, 
                                                     self.config['FRAME_WIDTH'],
                                                     self.config['FRAME_HEIGHT'])
                
                if frame_number is None:
                    logger.info("Finished Processing All Frames From [TrackerClass1].")
                    break
                
                if frame is None:
                    logger.warning(f"Received None Frame At Position [{frame_number}], Skipping From [TrackerClass1].")
                    self.next_frame_index += 1
                    continue
                
                # Update tracking variables
                self.current_frame_number = frame_number 
            
                # Process each frame
                current_frame_ids = self.process_each_frame(self.model, frame, self.names, self.dictionary,  self.current_frame_number)
                
                self.frames_processed_count += 1
                self.next_frame_index += 1
                self.frames_since_last_save += 1
                
                # Check for auto save and reset count
                if self.frames_since_last_save >= self.save_every_n_frames:
                    logger.info(f"Auto Save Triggered After [{self.frames_since_last_save}] Frames From [TrackerClass1].")
                    self.save_and_reset_counts()
                    self.frames_since_last_save = 0
    
                # Handle keyboard input
                if not self.keyboard_input():
                    logger.info("Video Processing Stopped By User Input From [TrackerClass1].")
                    break
                    
                if  self.current_frame_number % self.config['LOG_INTERVAL_FRAMES'] == 0:
                    progress_per = (self.current_frame_number / self.total_frames) * 100
                    logger.info(f"Progress [{progress_per:.1f}] - Frame [{self.current_frame_number} / {self.total_frames}] - Processed" 
                                f"[{self.frames_processed_count}] frames - Tracked Objects [{len(self.dictionary)}] From [TrackerClass1].")

        except KeyboardInterrupt:
            logger.info("Video Processing Interrupted By User From [TrackerClass1].")
            try:
                self.cleanup_for_tracker1()
                cleanup_performed = True
                logger.info("Cleanup Completed After Keyboard Interrupt By User From [TrackerClass1].")
            except Exception as e:
                logger.error(f"Cleanup Failed During Keyboaed Interrupt From [TrackerClass1]: {e}")
            
            return
        
        except Exception as e:
            logger.error(f"Error Occurred During Video Processing From [TrackerClass1]: {e}")
            try:
                logger.info("Performing Emergency Cleanup Due To Exception From [TrackerClass1].")
                cleanup_inactive_ids(self.dictionary,
                                     current_frame_ids,
                                     self.current_frame_number,
                                     self.config['MAX_FRAMES_MISSING'],
                                     self.config['MAX_DICTIONARY_SIZE'])
                
                self.cleanup_for_tracker1()
                cleanup_performed = True
                logger.info("Emergency Cleanup Completed Successfully From [TrackerClass1].")
            except Exception as e:
                logger.error(f"Emergency Cleanup Faild Attempting Force Cleanup From [TrackerClass1]: {e}")
                try:
                    force_cleanup(self.cap, self.db_manager)
                    logger.info("Force Cleanup Completed Successfully From [TrackerClass1].")
                except Exception as e:
                    logger.error(f"Force Cleanup Also Faild From [TrackerClass1]: {e}")
            
            raise CustomException(e, sys)    
                
        finally:
            if not cleanup_performed:
                try:
                    logger.info(f"Video Processing Successfully Completed, Total Frames Processed: [{self.frames_processed_count}]."
                                f"Now Performing Final Cleanup From [TrackerClass1].")
                    cleanup_inactive_ids(self.dictionary,
                                        current_frame_ids,
                                        self.current_frame_number,
                                        self.config['MAX_FRAMES_MISSING'],
                                        self.config['MAX_DICTIONARY_SIZE'])
                    
                    self.cleanup_for_tracker1()
                    logger.info("Normal Cleanup Successfully Perforem From [TrackerClass1]")
                except Exception as e:
                    logger.error(f"Error Occurred During Normal Cleanup Performed From [TrackerClass1], Attempting Force Cleanup As Final Fallback: {e}")
                    try:
                        force_cleanup(self.cap, self.db_manager)
                        logger.info("Force Cleanup Completed Successfully As Fallback From [TrackerClass1].")
                    except Exception as e:
                        logger.critical(f"All Cleanup Attempts Failed From [TrackerClass1], System Resources May Not Be Properly Released: {e}")

    def detection_and_tracking_1_for_streamlit_app(self):
        try:
            frame_number, frame = get_next_frame(self.next_frame_index,
                                                     self.config['SKIP_FRAMES_FOR_TRACKER_1'],
                                                     self.total_frames, 
                                                     self.cap, 
                                                     self.config['FRAME_WIDTH'],
                                                     self.config['FRAME_HEIGHT'])
            
            if frame_number is None:
                return None, True
                
            if frame is None:
                logger.warning(f"Received None Frame At Position [{frame_number}], Skipping From [TrackerClass1].")
                self.next_frame_index += 1
                return None, False
                
            # Update tracking variables
            self.current_frame_number = frame_number 
            
            # Process each frame
            self.process_each_frame(self.model, frame, self.names, self.dictionary,  self.current_frame_number)
            
            self.frames_processed_count += 1
            self.next_frame_index += 1
            self.frames_since_last_save += 1
            
            # Check for auto save and reset count
            if self.frames_since_last_save >= self.save_every_n_frames:
                logger.info(f"Auto Save Triggered After [{self.frames_since_last_save}] Frames From [TrackerClass1].")
                self.save_and_reset_counts()
                self.frames_since_last_save = 0
    
            if  self.current_frame_number % self.config['LOG_INTERVAL_FRAMES'] == 0:
                progress_per = (self.current_frame_number / self.total_frames) * 100
                logger.info(f"Progress [{progress_per:.1f}] - Frame [{self.current_frame_number} / {self.total_frames}] - Processed" 
                            f"[{self.frames_processed_count}] frames - Tracked Objects [{len(self.dictionary)}] From [TrackerClass1].")

            return frame, False
        
        except Exception as e:
            logger.error(f"Error Occurred In detection_and_tracking_1_for_streamlit_app() Function From [TrackerClass1]: {e}")
            return None, False