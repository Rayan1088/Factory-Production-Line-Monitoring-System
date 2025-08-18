# Factory-Production-Line-Multiprocessing-Tracking-System

## Overview
This project is a **Factory Production Line Monitoring System** designed to track and monitor the movement of **parcel boxes** and **cement bags** on a production line using
**computer vision** and **object tracking**. The system is build on **Yolo11m** model for **object detection and tracking** and **SQLite** for data storage. The system supports both **local execution** and **web-based** interface using **Streamlit** for real-time video processing and data visualization.

The system consists of two main trackers:
- **TrackerClass1**: Monitors boxes **entering** and **exiting** the production line, counting **"in"** and **"out"** movements.
- **TrackerClass2**: Monitors cement bags **exiting** the production line, counting **"out"** movements.

**Both trackers process video feeds, detect objects, track their movement acress predefine counting lines, and store counts in **SQLite database**.

## Features
- **Object Detection and Tracking**: Uses **Yolo11m** for detection and tracking **boxes (class 0)** and **cementbags (class 1)**, where we use YOLO's pre-build **Bot-SHORT** tracker algorithm.    
- **Database Management**: Uses **3 kind of data saveing system** to each tracker class and save those data to SQLite database (`tracking_boxes_data.db` and `tracking_cement_bags_data.db`). 
- **Real-Time Visualization**: Displays video feeds with **bounding boxes**, **trajectories** and **counting line** via **Streamlit** and **OpenCV (for local execution)**.
- **Multiprocessing**: Supports **parallel processing** pf two video feeds for efficient tracking.
- **Error Handling**: Comprehensive **logging** and **custom exception handling** for robust operation.
- **Configuration**: Uses a `config.yaml` file for custmizable parameters like - **(model path, video sources, frame processing parameters, tracking parameters, database paths and save interval parameters, counting lines parameters and drawing parameters)**.  

## Project Structure
```
|
|--src/
|   |-- databaseManager.py  # Manages SQLite databases creation and operations 
|   |-- exception.py        # Custom exception handling
|   |-- logger.py           # Custom logging setup
|   |-- tracker_1.py        # Tracks boxes (in/out counts) 
|   |-- tracker_2.py        # Tracks cement bags (out counts)
|   |-- utils.py            # Utility functions for model load, video processing
|                              and tracking
|--- app.py                 # Streamlt app for web-base interface
|--- config.yaml            # Configuration file for paths and parameters
|--- main.py                # Main script for multiprocessing and run trackers
|--- README.md              # Project documentation
```

## Requirements
- Python 3.8+
- Libraries:
    - `pandas`
    - `numpy`
    - `opencv-python`
    - `ultralytics`
    - `sqlite3`
    - `pyyaml`
- A trained YOLO model file (specified in `config.yaml`)
- Video files or camera feeds fro processing

## Installation 
1. Clone the repository:
    ```
    git clone https://github.com/Rayan1088/Factory-Production-Line-Objects-Detect-Track-And-Count.git
    ```
2. Install dependencies:
    ```
    pip install -r requirements.txt
    ```
3. Prepare the `config.yaml` file with appropriate paths and setting.

## Usage
### Local Execution
run the main script to process videos using multiprocessing:
```
python main.py
``` 
- This starts two processes (`TrackerClass1` and `TrackerClass2`) to track boxes and cement bags.
- Use keyboard controls:
    - `ESC`: Exit processing
    - `SPACE`: Pause/resume
    - `S`: Save counts to database
    - `R`: Reset counts
- Results are stored in SQLite database and also can be viewed at the end of execution.

### Streamlit App
Run the Streamlit app for web-based interface:
```
streamlit run app.py
```
- Access the app at `http://localhost:8501`
- Select **Video Processing** to view real-time tracking for both trackers.
- Select **Database Records** to view stored counts in tables for both trackers.
- The app displays video feeds with overlaid tracking information and supports database save and visualization.

## **Configuration**
Edit `config.yaml` to customize:
- **MODEL_PATH**: Path to the YOLO custom train model path.
- **VIDEO_SOURCE_1** and **VIDEO_SOURCE_2**: Paths to video files. 
- **DB_PATH_1** and **DB_PATH_2**: Paths to SQLite database for save the counts.
- **DATABASE_SAVE_TIME_INTERVAL**: Interval value for auto-save counts.
- **HORIZONTAL_LINE_1_FOR_TRACKER_1**, **HORIZONTAL_LINE_2_FOR_TRACKER_1**, **HORIZONTAL_LINE_3_FOR_TRACKER_1**, **COUNTING_LINE_1_FOR_TRACKER_2**: Is coordinates value for counting lines.
- **SKIP_FRAMES_FOR_TRACKER_1/2**: Number of frames to skip for performance.
- Other parameters for **frame processing**, **thresholds** and **visualization** settings.

## Database Schema
- **tracking_box_data.db (DB_PATH_1) - Table - 'tracking_box_counts'**: 
    - `id`: Auto incremented primary key
    - `total_box_in`: Counts of boxes entering
    - `total_box_out`: Counts of boxes exiting
    - `date_time`: Timestamp of records
- **tracking_cement_bags_data.db (DB_PATH_2) - Table - 'tracking_cement_bag_counts'**: 
    - `id`: Auto incremented primary key
    - `total_cement_bag_out`: Counts of cement bags exiting
    - `date_time`: Timestamp of records

## Logging 
- The system uses a custom logger (`src.logger`) to log events, error and debugging information.
- Logs are generated for database operations, video processing and tracking events.

## Notes
- Ensure video sources and model paths are valid in `config.yaml` befor running.
- For local execution, **uncomment OpenCV Display and other relevent code** in `tracker_1.py` and `tracker_2.py` as noted in the source files.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details. 



