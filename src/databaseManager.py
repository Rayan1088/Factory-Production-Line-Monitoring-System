import sqlite3
import sys
import os
from src.exception import CustomException
from src.logger import logging

logger = logging.getLogger("database") 

class DataBaseManagerClass:
    def __init__(self, db_path_1, db_path_2, limit = 10):
        self.db_path_1 = db_path_1
        self.db_path_2 = db_path_2
        self.limit = limit
        self.create_dbs()
        self.init_dbs()
    
    def create_dbs(self):
        try:
            # Check DB Exist Or Not And Create
            # For Box
            if not os.path.exists(self.db_path_1):
                connection_1 = sqlite3.connect(self.db_path_1)
                connection_1.close()
                logger.info(f"[{self.db_path_1}] - Database Create Successfully.")
            else:
                logger.info(f"[{self.db_path_1}] - Database Already Exists.")
                
            # For Cement Bags
            if not os.path.exists(self.db_path_2):
                connection_2 = sqlite3.connect(self.db_path_2)
                connection_2.close()
                logger.info(f"[{self.db_path_2}] - Database Create Successfully.")
            else:
                logger.info(f"[{self.db_path_2}] - Database Already Exists.")
        
        except Exception as e:
            logger.error(f"Error Occurred During Create Database: {e}")
            raise CustomException(e, sys)        
       
    def init_dbs(self):
        try:
            # Initialize 1st Database And Create Tables
            connection_1 = sqlite3.connect(self.db_path_1)
            cursor_1 =  connection_1.cursor()
            
            # Initialize 1st Database And Create Tables
            connection_2 = sqlite3.connect(self.db_path_2)
            cursor_2 =  connection_2.cursor()
            
            # Check If Table Exists Or Not For 'db_path_1'
            cursor_1.execute(''' SELECT name FROM sqlite_master WHERE
                                 type = 'table' AND name = 'tracking_box_counts' ''')
        
            # Check If Table Exists Or Not For 'db_path_2'
            cursor_2.execute(''' SELECT name FROM sqlite_master WHERE
                                 type = 'table' AND name = 'tracking_cement_bag_counts' ''')
            
            # For Boxs 
            if not cursor_1.fetchone():
                cursor_1.execute(''' CREATE TABLE tracking_box_counts (
                                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                                        total_box_in INTEGER DEFAULT 0,
                                        total_box_out INTEGER DEFAULT 0,
                                        date_time DATETIME DEFAULT CURRENT_TIMESTAMP) ''')  
                logger.info("The Table [tracking_box_counts] Created Successfully.")
            else:
                logger.info("The Table [tracking_box_counts] Already Exists.") 
            
            # For Cements Bags 
            if not cursor_2.fetchone():
                cursor_2.execute(''' CREATE TABLE tracking_cement_bag_counts (
                                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                                        total_cement_bag_out INTEGER DEFAULT 0,
                                        date_time DATETIME DEFAULT CURRENT_TIMESTAMP) ''')  
                logger.info("The Table [tracking_cement_bag_counts] Created Successfully.")
            else:
                logger.info("The Table [tracking_cement_bag_counts] Already Exists.")     
              
            connection_1.commit()
            connection_1.close()
            connection_2.commit()
            connection_2.close()
            
            logger.info(f"Database [{self.db_path_1}] And [{self.db_path_2}] Initialization Completed Successfully.")

        except Exception as e:
            logger.error(f"Error Occurred During Initializing Database [{self.db_path_1}] And [{self.db_path_2}]: {e}")
            raise CustomException(e, sys)
    
    def save_box_counts(self, in_count, out_count):
        try:
            connection = sqlite3.connect(self.db_path_1)
            cursor = connection.cursor()
            
            cursor.execute(''' INSERT INTO tracking_box_counts (total_box_in, total_box_out)
                               VALUES (?, ?) ''', (in_count, out_count))
            
            connection.commit()
            connection.close()
            
            logger.info(f"Save To Database [{self.db_path_1}] - Box In: [{in_count}], Box Out: [{out_count}].")
         
        except Exception as e:
            logger.error(f"Error Occurred During Saving Data To Database [{self.db_path_1}]: {e}")
            raise CustomException(e, sys)    
        
    def save_cement_bag_counts(self, out_count):
        try:
            connection = sqlite3.connect(self.db_path_2)
            cursor = connection.cursor()
            
            cursor.execute(''' INSERT INTO tracking_cement_bag_counts (total_cement_bag_out)
                               VALUES (?) ''', (out_count,))
            
            connection.commit()
            connection.close()
            
            logger.info(f"Save To Database [{self.db_path_2}] - Total Cement Bags Out: [{out_count}].")
         
        except Exception as e:
            logger.error(f"Error Occurred During Saving Data To Database [{self.db_path_2}]: {e}")
            raise CustomException(e, sys)  
            
    def get_recent_counts_for_boxs(self):
        try:
            connection = sqlite3.connect(self.db_path_1)
            cursor = connection.cursor()
            
            cursor.execute(''' SELECT total_box_in, total_box_out, date_time
                               FROM tracking_box_counts ORDER BY date_time DESC LIMIT ? ''', (self.limit,))
            
            records_1 = cursor.fetchall()
            logger.info(f"Successfully Retrieve Recent Tracking Records From Database [{self.db_path_1}].")
            
            connection.close()
            return records_1           
        
        except Exception as e:
            logger.error(f"Error Occurred During Retrieve Recent Tracking Records From Database [{self.db_path_1}]: {e}")
            raise CustomException(e, sys)
        
    def get_recent_counts_for_cement_bags(self):
        try:
            connection = sqlite3.connect(self.db_path_2)
            cursor = connection.cursor()
            
            cursor.execute(''' SELECT total_cement_bag_out, date_time
                               FROM tracking_cement_bag_counts ORDER BY date_time DESC LIMIT ? ''', (self.limit,))
            
            records_2 = cursor.fetchall()
            logger.info(f"Successfully Retrieve Recent Tracking Records From Database [{self.db_path_2}].")
            
            connection.close()
            return records_2           
        
        except Exception as e:
            logger.error(f"Error Occurred During Retrieve Recent Tracking Records From Database [{self.db_path_2}]: {e}")
            raise CustomException(e, sys)
    
    def view_database_records(self):
        try:
            # For Boxs
            records_1 = self.get_recent_counts_for_boxs()
            print("\n" + "="*80)
            print(f"{'Total Box In':<15} {'Total Box Out':<15} {'Date_time':<20}")
            print("="*80)
            for record_1 in records_1:
                total_box_in, total_box_out, date_time = record_1
                print(f"{total_box_in:<15} {total_box_out:<15} {date_time:<20}")
            print("="*80)
            
            # For Cements Bags
            records_2 = self.get_recent_counts_for_cement_bags()
            print("\n" + "="*80)
            print(f"{'Total Cement Bag Out':<18} {'Date_time':<20}")
            print("="*80)
            for record_2 in records_2:
                total_cement_bag_out, date_time = record_2
                print(f"{total_cement_bag_out:<18} {date_time:<20}")
            print("="*80)
            
        except Exception as e:
            logger.error(f"Error Occurred During Viewing Database Records: {e}")
            raise CustomException(e, sys)
            
