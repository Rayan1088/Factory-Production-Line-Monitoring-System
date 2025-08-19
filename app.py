import os
import streamlit as st
import time
import pandas as pd
import numpy as np
import sqlite3
from src.utils import load_config
from src.databaseManager import DataBaseManagerClass
from src.tracker_1 import TrackerClass1
from src.tracker_2 import TrackerClass2

config = load_config(config_path="config.yaml")

def display_database_page(table_name, title, use_turso_db=False):
    try:
        if use_turso_db:
            db_manager = DataBaseManagerClass(local_db_path= config['LOCAL_DB_PATH'], # Fallback
                                              turso_db_url = config['TURSO_DB_URL'], 
                                              turso_db_token = config['TURSO_DB_TOKEN'],
                                              limit = config['LIMIT'])
        else:
            db_manager = DataBaseManagerClass(local_db_path = config['LOCAL_DB_PATH'], limit = config['LIMIT'])            
    except Exception as e:
        st.error(f"Error Initializing Database Connection: {e}")
        return None

    if not db_manager.is_db_connected():
        st.error("Failed To Connect To The Database.")
        return None
    
    st.subheader(title)
    with st.spinner(f"Loading {title.lower()} Data..! Please Wait."):
        df = db_manager.fetch_database_records(table_name)
    
    if not df.empty:
        if not df.columns.is_unique:
            df.columns = pd.io.common.dedup_names(df.columns, is_potential_multiindex=False)

        st.dataframe(df, use_container_width=True)
        
        download_csv = df.to_csv(index=False)
        st.download_button(
            label=f"Download {title} As CSV.",
            data=download_csv,
            file_name=f"{table_name}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv")
    else:
        st.info(f"No Data Found In [{table_name}] Table.") 

def main_app(use_turso_db, streamlit):
    st.title("Factory Production Line Multiprocessing Tracking-System")

    st.sidebar.markdown("<div style='font-size:34px; font-weight:bold; color:red;'>Select A Page</div>", unsafe_allow_html=True)
    page = st.sidebar.selectbox(label="Select A Page", options=["Video Processing","Database Records"], label_visibility="collapsed")
    # Original label="Select A Page" is hide for label_visibility="collapsed" 

    if page == "Video Processing":
        st.header("Live Video Processing")
 
        col_1, col_2 = st.columns(2) 
        with col_1:
            st.header("Tracker 1")
            image_placeholder_1 = st.empty()
            status_1 = st.empty()
            
        with col_2:
            st.header("Tracker 2")
            image_placeholder_2 = st.empty()
            status_2 = st.empty()
            
        col_btn_1, col_btn_2, col_btn_3 = st.columns([1,1,2]) 
        
        with col_btn_1:
            start_button = st.button("Start Tracking", type="primary", use_container_width=True)
        with col_btn_2:
            if st.button("Stop Processing", use_container_width=True):
                st.rerun()    
            
        if start_button:
            if not os.path.exists("config.yaml"):
                st.error("Config File Not Found.")
                st.stop() 
            
            Tracker_1 = None
            Tracker_2 = None
            
            try:
                with st.spinner("Initializing Trackers...! Please Wait !"):
                    Tracker_1 = TrackerClass1(config_path="config.yaml", use_turso_db=True, streamlit=streamlit)
                    Tracker_2 = TrackerClass2(config_path="config.yaml", use_turso_db=True, streamlit=streamlit)
                
                st.success("Initialized Trackers Successfully!")
                
                done_process_1 = False
                done_process_2 = False
                
                fps_Tracker_1 = Tracker_1.fps if hasattr(Tracker_1, 'fps') and Tracker_1.fps and Tracker_1.fps > 0 else 30
                fps_Tracker_2 = Tracker_2.fps if hasattr(Tracker_2, 'fps') and Tracker_2.fps and Tracker_2.fps > 0 else 30
                
                max_iterations = 100
                iteration_count = 0
                
                progress_bar = st.progress(0)
                iteration_text = st.empty()
                
                while not (done_process_1 and done_process_2) and iteration_count < max_iterations:
                    iteration_count += 1
                    
                    progress = min(iteration_count / max_iterations, 1.0)
                    progress_bar.progress(progress)
                    iteration_text.text(f"Processing Iteration: [{iteration_count}/{max_iterations}].")
                    
                    if not done_process_1:
                        try:
                            frame_1, is_done_process_1 = Tracker_1.detection_and_tracking_1_for_streamlit_app()
                            if frame_1 is not None and isinstance(frame_1, np.ndarray):
                                image_placeholder_1.image(frame_1, channels="BGR", use_container_width=True)
                                status_1.success("Tracker 1 Activate !!")
                                
                            if is_done_process_1:
                                done_process_1 = True
                                status_1.success("Tracker 1 Completed !!")
                            time.sleep(1/fps_Tracker_1)

                        except Exception as e:
                            st.error(f"Error In Tracker 1: {str(e)}")
                            status_1.error("Tracker 1 Error !!")
                            done_process_1 = True
                            
                    if not done_process_2:
                        try:
                            frame_2, is_done_process_2 = Tracker_2.detection_and_tracking_2_for_streamlit_app()
                            if frame_2 is not None:
                                image_placeholder_2.image(frame_2, channels="BGR", use_container_width=True)
                                status_2.success("Tracker 2 Activate !!")
                                
                            if is_done_process_2:
                                done_process_2 = True
                                status_2.success("Tracker 2 Completed !!")
                            time.sleep(1/fps_Tracker_2)
                        
                        except Exception as e:
                            st.error(f"Error In Tracker 2: {str(e)}")
                            status_2.error("Tracker 2 Error !!")
                            done_process_2 = True
                
                progress_bar.progress(1.0)
                
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
                    try:
                        Tracker_1.cleanup_for_tracker1()
                    except Exception as e:
                        st.warning(f"Warning: Tracker 1 Cleanup Failed: {e}")
                
                if Tracker_2 is not None and hasattr(Tracker_2, 'cleanup_for_tracker2'):
                    try:
                        Tracker_2.cleanup_for_tracker2()
                    except Exception as e:
                        st.warning(f"Warning: Tracker 2 Cleanup Failed: {e}")
        
    elif page == "Database Records":
        if use_turso_db:
            st.header("Turso Database Records")
        else:
            st.header("Local SQLite Database Records")

        if st.button("Refresh Data"):
            st.cache_data.clear()
            st.rerun()
        
        table_name = config['DB_TABLE_NAMES']
        
        display_database_page(table_name=table_name[0], title=table_name[0], use_turso_db=use_turso_db)
        st.divider()
        display_database_page(table_name=table_name[1], title=table_name[1], use_turso_db=use_turso_db)
                          
if __name__ == "__main__":
    main_app(use_turso_db=True, streamlit=True)
    
