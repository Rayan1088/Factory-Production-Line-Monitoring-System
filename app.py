import os
import streamlit as st
import time
import pandas as pd
import numpy as np
import sqlite3
from src.utils import load_config
from src.tracker_1 import TrackerClass1
from src.tracker_2 import TrackerClass2

config = load_config(config_path="config.yaml")

st.title("Factory Production Line Monitoring")

st.sidebar.markdown("<div style='font-size:34px; font-weight:bold; color:red;'>Select A Page</div>", unsafe_allow_html=True)
page = st.sidebar.selectbox(label="Select A Page", options=["Video Processing","Database Records"], label_visibility="collapsed")
# Original label="Select A Page" is hide for label_visibility="collapsed" 

if page == "Video Processing":
    col_1, col_2 = st.columns(2) 
    with col_1:
        st.header("Tracker 1")
        image_placeholder_1 = st.empty()
    with col_2:
        st.header("Tracker 2")
        image_placeholder_2 = st.empty()
        
    if st.button("Start Tracking Processing"):
        Tracker_1 = None
        Tracker_2 = None
        try:
            if not os.path.exists("config.yaml"):
                st.error("Config File Not Found.")
                st.stop()          
                  
            Tracker_1 = TrackerClass1(config_path="config.yaml")
            Tracker_2 = TrackerClass2(config_path="config.yaml")
            
            done_process_1 = False
            done_process_2 = False
            
            fps_Tracker_1 = Tracker_1.fps if hasattr(Tracker_1, 'fps') and Tracker_1.fps and Tracker_1.fps > 0 else 30
            fps_Tracker_2 = Tracker_2.fps if hasattr(Tracker_2, 'fps') and Tracker_2.fps and Tracker_2.fps > 0 else 30
            
            max_iterations = 10000
            iteration_count = 0
            
            while not (done_process_1 and done_process_2) and iteration_count < max_iterations:
                iteration_count += 1
                if not done_process_1:
                    try:
                        frame_1, is_done_process_1 = Tracker_1.detection_and_tracking_1_for_streamlit_app()
                        if frame_1 is not None and isinstance(frame_1, np.ndarray):
                            image_placeholder_1.image(frame_1, channels="BGR", use_container_width=True)
                        
                        if is_done_process_1:
                            done_process_1 = True
                        time.sleep(1/fps_Tracker_1)

                    except Exception as e:
                        st.error(f"Error In Tracker 1: {str(e)}")
                        done_process_1 = True
                        
                if not done_process_2:
                    try:
                        frame_2, is_done_process_2 = Tracker_2.detection_and_tracking_2_for_streamlit_app()
                        if frame_2 is not None:
                            image_placeholder_2.image(frame_2, channels="BGR", use_container_width=True)
                    
                        if is_done_process_2:
                            done_process_2 = True
                        time.sleep(1/fps_Tracker_2)
                    
                    except Exception as e:
                        st.error(f"Error In Tracker 2: {str(e)}")
                        done_process_2 = True
            
            if iteration_count >= max_iterations:
                st.warning("Processing Stopped Due To Timeout.")
            else:
                st.success("Both Video Processing Is Completed Successfully!! To See The Count Records, Please Go To 'Database Records' Page In Sidebar. Thank You!!!")
        
        except FileNotFoundError as e:
            st.error(f"File Not Found: {str(e)}")
        except sqlite3.OperationalError as e:
            st.error(f"Database Error: {str(e)}")
        except Exception as e:
            st.error(f"Error Occurred During Processing The Videos: {str(e)}")
        finally:
            if Tracker_1 is not None and hasattr(Tracker_1, 'cleanup_for_tracker1'):
                Tracker_1.cleanup_for_tracker1()
            
            if Tracker_2 is not None and hasattr(Tracker_2, 'cleanup_for_tracker2'):
                Tracker_2.cleanup_for_tracker2()
    
elif page == "Database Records":
    st.header("Database Records")
    st.subheader("Box Count Database")
    connection_1 = None
    
    try:
        connection_1 = sqlite3.connect(config['DB_PATH_1'])
        df_box = pd.read_sql_query("SELECT * FROM tracking_box_counts ORDER BY date_time DESC LIMIT 1000", connection_1)
        if not df_box.empty:
            st.dataframe(df_box)
        else:
            st.info("No Data Found In tracking_box_counts Table.")
            
    except Exception as e:
        st.error(f"Could Not Read From tracking_box_counts Table: {e}")
    finally:
        if connection_1:
            connection_1.close() 
        
    st.subheader("Cement Beg Count Database")
    connection_2 = None
    try:
        connection_2 = sqlite3.connect(config['DB_PATH_2'])
        df_cement = pd.read_sql_query("SELECT * FROM tracking_cement_bag_counts ORDER BY date_time DESC LIMIT 1000", connection_2)
        if not df_cement.empty:
            st.dataframe(df_cement)
        else:
            st.info("No Data Found In tracking_cement_bag_counts Table.")
    
    except Exception as e:
        st.error(f"Could Not Read From tracking_cement_bag_counts Table: {e}")
    finally:
        if connection_2:
            connection_2.close() 