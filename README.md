# Factory-Production-Line-Multiprocessing-Tracking-System

- App Link - https://factory-appuction-line-multiprocessing-tracking-system.streamlit.app/

## Overview
This **Factory-Production-Line-Multiprocessing-Tracking-System** is a robust solution for real-time tracking and monitoring of **parcel boxes** and **cement bags** on a factory production line using **computer vision** and **object tracking**. The system is built on a **YOLOv11m** model for **object detection** and the **Bot-SORT Algorithm** for **tracking**, combined with **SQLite** or **Turso (cloud-hosted SQLite)** for data storage. The system supports both **local execution** and **web-based** interface using **Streamlit** for real-time video processing and data visualisation.

The system consists of two main trackers:
- **TrackerClass1**: Monitors **parcel boxes** (class 0) **entering** and **exiting** the production line across **multiple counting lines**, recording **"in"** and **"out"** counts.
- **TrackerClass2**: Monitors **cement bags** (class 1) **exiting** the production line across a **single counting lines**, recording **"out"** counts.

Both trackers process video feeds, detect objects, track their movement, and store counts in a **selective database (either local or cloud-based)**. The system uses **multiprocessing** to run trackers in parallel, ensuring efficient handling of **multiple video streams**. It includes **custom exception handling**, **logging** and a **configurable setup** via a `config.yaml` file, making it adaptable to various production line environments. 

## Features
- **Object Detection and Tracking**: Detects and tracks parcel boxes and cement bags using YOLO11m and Bot-SORT, with bounding boxes, trajectories and confidence scores.
- **Database Management**: Uses **three kinds of data saving systems** for each tracker class and saves that data in local SQLite or a Turso cloud-hosted SQLite database with automatic fallback to local database if cloud connectivity fails.
- **Real-Time Visualisation**: Displays video feeds with bounding boxes, trajectories and counting lines via Streamlit (web) and OpenCV (for local execution).
- **Multiprocessing**: Runs TrackerClass1 and TrackerClass2 in parallel using Python's `multiprocessing` module for performance optimisation.
- **Web Interface**: Streamlit-based dashboard for real-time video monitoring, database visualisation and data export as CSV.
- **Configurability**: Uses a `config.yaml` file for customizable parameters (model path, video sources, database settings, frame processing parameters, tracking parameters, counting lines parameters and drawing parameters)**. 
- **Error Handling And Logging**: Comprehensive **logging** and **custom exception handling** for robust operation.
- **User Controls**: Supports keyboard inputs (`ESC`, `SPACE`, `S`, `R`) for local execution and interactive controls in the streamlit app.
 

## Project Structure
```
|
|--CVvenv/                  # Conda virtual environment 
|   |--
|   |--
|
|--notebooks/
|   |-- fine-tune-yolov11m-for-real-time-object-detection.ipynb 
|                           # Custom model training file (fine-tune-yolov11m)
|--src/
|   |-- databaseManager.py  # SQLite/Turso databases creation and operations 
|   |-- exception.py        # Custom exception handling
|   |-- logger.py           # Custom logging setup
|   |-- multiprocessing.py  # Manages parallel local execution of trackers 
|   |-- tracker_1.py        # Tracks boxes (in/out counts) 
|   |-- tracker_2.py        # Tracks cement bags (out counts)
|   |-- utils.py            # Utility functions for model load, video processing
|                              and tracking
|--training results/
|   |
|   '------                 # Custom fine-tune model training results
| 
|--video data/
|   |-- box.mp4             # Parcel box data
|   |-- pac.mp4             # Cement bags data
|
|--weights/
|   |-- best.pt             # Custom fine tune model
|   |-- yolo11m.pt          # Yolo pre-train model
|
|--- LICENSE                # License file (MIT License)
|--- README.md              # Project documentation
|--- app.py                 # Streamlit app for web-based interface
|--- config.yaml            # Configuration file for paths and parameters
|--- main.py                # Main script for multiprocessing and running trackers
|--- requirements.txt       # Python dependencies
|--- tracking_data.db       # Local SQLite database

```

## Requirements
- **Python 3.11**
- **Dependencies (listed in `requirements.txt`)**:
    - `ultralytics==8.3.179` # Streamlit cloud app deploy support

    - `lap==0.5.12`             
    
    - `pandas`
    
    - `numpy<2` # Streamlit cloud app deploy support
    
    - `libsql-client`
    
    - `streamlit`
    
    - `opencv-python-headless` # if you don’t need GUI, then use headless of the `opencv-python` 
    
    - `pyyaml`

- **YOLO Model**: A trained YOLOv11m model file (specified in `config.yaml`)
- **Video Sources**:Video files or camera feeds for processing
- **Turso Database**: Turso database URL and token for cloud-hosted SQLite (specified in `config.yaml`)

## Installation 
1. Clone the repository:
 ```
 git clone https://github.com/Rayan1088/Factory-Production-Line-Multiprocessing-Tracking-System.git
 ```
2. Install dependencies:
 ```
 pip install -r requirements.txt
 ```
3. Configure `config.yaml` file:
    - Specify paths for the YOLO model (`MODEL_PATH`), video sources (`VIDEO_SOURCES_1`, `VIDEO_SOURCES_2`), and database (`LOCAL_DATABASE_PATH`) 

    - Provide `TURSO_DB_URL` and `TURSO_DB_TOKEN` for cloud database usage.
   
    - Adjust tracking parameters (like counting line coordinates, frame skipping, and visualisation settings).

## Usage
### Local Execution
Run the main script to process videos using multiprocessing:
```
python main.py
``` 
- This starts two processes (`TrackerClass1` and `TrackerClass2`) to track boxes and cement bags in parallel.
- Displays video feeds with bounding boxes, trajectories and counts using OpenCV windows.
- **Use keyboard controls**:
    - `ESC`: Exit processing
    - `SPACE`: Pause/resume
    - `S`: Save counts to database
    - `R`: Reset counts
- Results are stored in a configured database (SQLite or Turso) and can also be viewed at the end of execution using the `view_database_records` function.

### Streamlit App
Run the Streamlit app for a web-based interface:
```
streamlit run app.py
```
- Access the app at `http://localhost:8501`
- **Pages**:
    - **Video Processing**: Shows real-time video feeds for both trackers, with bounding boxes, trajectories, counting lines and live counts.
    - **Database Records** Displays stored counts in (Local SQLite or Turso) database tables for both trackers, with options to filter by date and download as CSV.
- The app supports interactive controls for monitoring and database operations.

### Database Support
- **Local SQLite**: Stores data in a single SQLite database file specified by `LOCAL_DB_PATH` in `config.yaml` file.
- **Turso (cloud-hosted SQLite Database)**: Connects to a cloud-hosted SQLite Database using `TURSO_DB_URL` and `TURSO_DB_TOKEN`, with **automatic fallback** to local SQLite if the connection fails.
- The `DataBaseManagerClass` handles database initialisation, table creation, data save and fetch operations. 

## Database Schema
The database contains two tables:
- **tracking_box_counts (TrackerClass1)**: 
    - `id`: Auto-incremented primary key
    - `total_box_in`: Counts of boxes entering
    - `total_box_out`: Counts of boxes exiting
    - `date_time`: Timestamp of records
- **tracking_cement_bag_counts (TrackerClass2)**: 
    - `id`: Auto-incremented primary key
    - `total_cement_bag_out`: Counts of cement bags exiting
    - `date_time`: Timestamp of records

## Configuration
The `config.yaml` file allows customisation of the following key parameters,
- **Model And Video**:
    - `MODEL_PATH`: Path to the YOLO custom train model.
    - `VIDEO_SOURCE_1`: Video file path or camera index for TrackerClass1. 
    - `VIDEO_SOURCE_2`: Video file path or camera index for TrackerClass1. 
- **Database**:    
    - `LOCAL_DB_PATH`: Path to the local SQLite database file.
    - `TURSO_DB_URL`: URL for Turso database.
    - `TURSO_DB_TOKEN`: Authentication token for Turso.
    - `DATABASE_SAVE_TIME_INTERVAL`: Interval value for auto-save counts.
- **Tracking**:
    - `HORIZONTAL_LINE_1_FOR_TRACKER_1`: Is the coordinates value for the counting line (TrackerClass1).
    - `HORIZONTAL_LINE_2_FOR_TRACKER_1`: Is the coordinates value for the counting line (TrackerClass1).
    - `HORIZONTAL_LINE_3_FOR_TRACKER_1`: Is the coordinates value for the counting line (TrackerClass1).
    - `COUNTING_LINE_1_FOR_TRACKER_2`: Is the coordinates value for the counting line (TrackerClass2).
    - `SKIP_FRAMES_FOR_TRACKER_1`: Number of frames to skip for performance (TrackerClass1).
    - `SKIP_FRAMES_FOR_TRACKER_2`: Number of frames to skip for performance (TrackerClass2).
    - `MAX_FRAMES_MISSING`: Maximum frames an object can be missing before **track cleanup** for that particular object.
    - `MAX_DICTIONARY_SIZE`: **Maximum number of tracked objects** to prevent memory overload.
- **Visualization**:
    - `FRAME_WIDTH`, `FRAME_HEIGHT`: Dimensions for resizing video frames.
    - `BBOX_THICKNESS`, `CENTER_POINT_RADIUS`, `TRAJECTORY_THICKNESS`, `FONT_SIZE`, `FONT_THICKNESS`, `TEXT_LABEL`, `TEXT_POSITION`: Those parameters for bounding boxes, text, and trajectories visualisation settings.

## Technical Details
### Object Detection and Tracking
- **YOLOv11m**: Detects parcel boxes and cement bags with high accuracy.
- **Bot-SORT**: Tracks objects across frames, maintaining consistent IDs and recording trajectories.
- **TrackerClass1**: Monitors boxes crossing three horizontal lines, counting 'in (upward)' and 'out (downward )' movements.
- **TrackerClass2**: Monitors cement bags crossing a single line, counting 'out (downward) movements.
- Both tracker use **dictionaries** to store tracking data **(trajectories, bounding boxes, confidence scores)** and perform **periodic cleamup** based on `MAX_FRAMES_MISSING` and `MAX_DICTIONARY_SIZE`.

### Multiprocessing
- The `MultiprocessingForTrackers` class manages parallel execution of TrackerClass1 and TrackerClass2 in separate processes.
- Includes monitoring and cleanup to ensure proper resource release, like video capture, database connections, and closing OpenCV windows. 
- Supports configurable timeout to terminate long-running processes.

### Logging 
- Custom logging (`src.logger`) records: 
    - Initialisation of trackers, database, and multiprocessing.
    - Frame-by-frame tracking events (detections, counts, trajectory updates).
    - Database operations (connections, save, errors).
    - Multiprocessing events (process start, termination, errors).
- Logs are written to files and the console, aiding debugging and monitoring.

### Error Handling
- Custom exception (`src.exception`) handles errors in video processing, database operations and tracking.
- Ensure graceful recovery or termination with detailed error messages and stack traces.

## Notes
- **Video Sources and Model**: Ensure video sources and model paths are valid in `config.yaml` before running. Invalid sources will raise exceptions.
- **Turso Setup**: Provide valid `TURSO_DB_URL` and `TURSO_DB_TOKEN` in `config.yaml` for cloud database usage. Without these, the system defaults to local SQLite.
- **Resource Cleanup**: The system automatically releases resources (video capture, database connections, OpenCV windows) on termination or errors.
- **Performance**: Adjust `SKIP_FRAMES_FOR_TRACKER_1/2` and `FRAME_WIDTH/HEIGHT` in `config.yaml` to optimise performance on resource-constrained systems.

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -m 'Add your feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a Pull Request.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details. 

## Contact 
For questions or support, please open an issue on GitHub or contact [mrrayan27@gmail.com].



