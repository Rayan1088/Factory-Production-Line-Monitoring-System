import sys
import os
import sqlite3
from libsql_client import create_client
from src.exception import CustomException
from src.logger import logging

logger = logging.getLogger("database") 

class DataBaseManagerClass:
    def __init__(self, db_path, limit):
        self.limit = limit
        self.use_turso = bool(os.getenv("TURSO_DB_URL") and os.getenv("TURSO_DB_TOKEN"))

        if self.use_turso:
            logger.info("Using Turso Cloud-hosted SQLite Database.")
            try:
                self.client = create_client(
                    url = os.getenv("TURSO_DB_URL"),
                    auth_token = os.getenv("TURSO_DB_TOKEN"))
                self.init_turso_dbs()
            except Exception as e:
                logger.error(f"Error Occurred During Initialize Turso Client: {e}")
                raise CustomException(e, sys)   
        else:
            self.db_path = db_path
            self.limit = limit
            self.create_dbs()
            self.init_dbs()
    
    def create_dbs(self):
        try:
            if not os.path.exists(self.db_path):
                connection = sqlite3.connect(self.db_path)
                connection.close()
                logger.info(f"[{self.db_path}] - Database Create Successfully.")
            else:
                logger.info(f"[{self.db_path}] - Database Already Exists.")
        except Exception as e:
            logger.error(f"Error Occurred During Create Database: {e}")
            raise CustomException(e, sys)        

    def init_turso_dbs(self):
        try:
            self.client.execute(''' CREATE TABLE IF NOT EXISTS tracking_box_counts (
                                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                                        total_box_in INTEGER DEFAULT 0,
                                        total_box_out INTEGER DEFAULT 0,
                                        date_time DATETIME DEFAULT CURRENT_TIMESTAMP) ''')  
            
            self.client.execute(''' CREATE TABLE IF NOT EXISTS tracking_cement_bag_counts (
                                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                                        total_cement_bag_out INTEGER DEFAULT 0,
                                        date_time DATETIME DEFAULT CURRENT_TIMESTAMP) ''') 
            logger.info("Turso Database Tables Created Successfully.")
        except Exception as e:
            logger.error(f"Error Occurred During Create Turso Database Tables: {e}")
            raise CustomException(e, sys)   

    def init_dbs(self):
        try:
            connection = sqlite3.connect(self.db_path)
            cursor =  connection.cursor()
            
            # Check If Table Exists Or Not 
            cursor.execute(''' SELECT name FROM sqlite_master WHERE
                                 type = 'table' AND name = 'tracking_box_counts' ''')
            # Check If Table Exists Or Not
            cursor.execute(''' SELECT name FROM sqlite_master WHERE
                                 type = 'table' AND name = 'tracking_cement_bag_counts' ''')
            # For Boxs 
            if not cursor.fetchone():
                cursor.execute(''' CREATE TABLE tracking_box_counts (
                                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                                    total_box_in INTEGER DEFAULT 0,
                                    total_box_out INTEGER DEFAULT 0,
                                    date_time DATETIME DEFAULT CURRENT_TIMESTAMP) ''')  
                logger.info("The Table [tracking_box_counts] Created Successfully.")
            else:
                logger.info("The Table [tracking_box_counts] Already Exists.") 
            
            # For Cements Bags 
            if not cursor.fetchone():
                cursor.execute(''' CREATE TABLE tracking_cement_bag_counts (
                                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                                    total_cement_bag_out INTEGER DEFAULT 0,
                                    date_time DATETIME DEFAULT CURRENT_TIMESTAMP) ''')  
                logger.info("The Table [tracking_cement_bag_counts] Created Successfully.")
            else:
                logger.info("The Table [tracking_cement_bag_counts] Already Exists.")     
              
            connection.commit()
            connection.close()
            logger.info(f"Database [{self.db_path}] Initialization Completed Successfully.")
        except Exception as e:
            logger.error(f"Error Occurred During Initializing Database [{self.db_path}]: {e}")
            raise CustomException(e, sys)
    
    def save_box_counts(self, in_count, out_count):
        try:
            if self.use_turso:
                self.client.execute(
                ''' INSERT INTO tracking_box_counts (total_box_in, total_box_out)
                    VALUES (?, ?) ''', (in_count, out_count))
                logger.info(f"Save To Turso Database - Box In: [{in_count}], Box Out: [{out_count}].")
            else:
                connection = sqlite3.connect(self.db_path)
                cursor = connection.cursor()
                cursor.execute(''' INSERT INTO tracking_box_counts (total_box_in, total_box_out)
                                   VALUES (?, ?) ''', (in_count, out_count))
                connection.commit()
                connection.close()
                logger.info(f"Save To Database [{self.db_path}] - Box In: [{in_count}], Box Out: [{out_count}].")
        except Exception as e:
            logger.error(f"Error Occurred During Saving Data To Database For Boxes Data: {e}")
            raise CustomException(e, sys)    
        
    def save_cement_bag_counts(self, out_count):
        try:
            if self.use_turso:
                self.client.execute(
                    ''' INSERT INTO tracking_cement_bag_counts (total_cement_bag_out)
                        VALUES (?) ''', (out_count,))
                logger.info(f"Save To Turso Database - Total Cement Bags Out: [{out_count}].")
            else:
                connection = sqlite3.connect(self.db_path)
                cursor = connection.cursor()
                cursor.execute(''' INSERT INTO tracking_cement_bag_counts (total_cement_bag_out)
                                   VALUES (?) ''', (out_count,))
                connection.commit()
                connection.close()
                logger.info(f"Save To Database [{self.db_path}] - Total Cement Bags Out: [{out_count}].")
        except Exception as e:
            logger.error(f"Error Occurred During Saving Data To Database For Cement Bags Data: {e}")
            raise CustomException(e, sys)  
            
    def get_recent_counts_for_boxs(self):
        try:
            if self.use_turso:
                records_turso = self.client.execute(
                    ''' SELECT total_box_in, total_box_out, date_time
                        FROM tracking_box_counts ORDER BY date_time DESC LIMIT ? ''', (self.limit,))
                logger.info(f"Successfully Retrieve Recent Boxes Tracking Records From Turso Database.")
                return records_turso.rows
            else:
                connection = sqlite3.connect(self.db_path)
                cursor = connection.cursor()
                cursor.execute(''' SELECT total_box_in, total_box_out, date_time
                                   FROM tracking_box_counts ORDER BY date_time DESC LIMIT ? ''', (self.limit,))
                records = cursor.fetchall()
                logger.info(f"Successfully Retrieve Recent Boxes Tracking Records From Database [{self.db_path}].")
                connection.close()
                return records          
        except Exception as e:
            logger.error(f"Error Occurred During Retrieve Recent Boxes Tracking Records From Database: {e}")
            raise CustomException(e, sys)
        
    def get_recent_counts_for_cement_bags(self):
        try:
            if self.use_turso:
                records_turso = self.client.execute(
                    ''' SELECT total_cement_bag_out, date_time
                        FROM tracking_cement_bag_counts ORDER BY date_time DESC LIMIT ? ''', (self.limit,))               
                logger.info(f"Successfully Retrieve Recent Cement Bags Tracking Records From Turso Database.")
                return records_turso.rows
            else:
                connection = sqlite3.connect(self.db_path)
                cursor = connection.cursor()
                cursor.execute(''' SELECT total_cement_bag_out, date_time
                                   FROM tracking_cement_bag_counts ORDER BY date_time DESC LIMIT ? ''', (self.limit,))
                records = cursor.fetchall()
                logger.info(f"Successfully Retrieve Recent Cement Bags Tracking Records From Database [{self.db_path_2}].")
                connection.close()
                return records          
        except Exception as e:
            logger.error(f"Error Occurred During Retrieve Recent Cement Bags Tracking Records From Database [{self.db_path_2}]: {e}")
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
            
