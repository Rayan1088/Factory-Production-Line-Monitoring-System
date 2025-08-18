import sys
import os
import pandas as pd
import sqlite3
import libsql_client
from src.exception import CustomException
from src.logger import logging

logger = logging.getLogger("database") 

class DataBaseManagerClass:
    def __init__(self, local_db_path=None, turso_db_url=None, turso_db_token=None, limit=None):
       
        self.limit = limit
        self.client = None
        self.local_db_path = local_db_path
        self.connection_url = turso_db_url
        self.auth_token = turso_db_token
        self.use_turso = bool(turso_db_url and turso_db_token)
        self.libsql_client = libsql_client
        
        if self.use_turso:
            self.init_turso_client(turso_db_url, turso_db_token)
        else:
            self.init_local_db()
          
    def init_turso_client(self, url, token):
        try:
            if not self.use_turso:
                logger.info("Turso Disabled, Skipping Initialization")
                return
        
            logger.info("Using Turso Cloud-hosted SQLite Database.")
            
            if url.startswith('libsql://'):
                self.connection_url = url
                self.auth_token = token
            else:
                if url.startswith('https://'):
                    self.connection_url = url.replace('https://', 'libsql://')
                elif url.startswith('http://'):
                    self.connection_url = url.replace('http://', 'libsql://') 
                else:
                    self.connection_url = f"libsql://{url}"
                    
                self.auth_token = token
            
            logger.info(f"Attempting To Connect To: {self.connection_url}")
 
            self.test_connection_and_create_turso_db_tables()
            
            logger.info("Turso Database Initialize Successfully.")
        except Exception as e:
            logger.error(f"Error Occurred During Initialize Turso Client: {e}")
            raise CustomException(e, sys) 

    def test_connection_and_create_turso_db_tables(self):
        conn_methods = [
            self.try_conn_method_1,
            self.try_conn_method_2,
            self.try_conn_method_3
        ]

        for i, method in enumerate(conn_methods, 1):
            try:
                logger.info(f"Trying Connection Method {i}")
                method()
                return
            except Exception as e:
                logger.warning(f"Connection Method {i} Failed: {e}")
                if i == len(conn_methods):
                    raise e

    def try_conn_method_1(self):
        with self.libsql_client.create_client_sync(self.connection_url, auth_token=self.auth_token) as client:
            self.create_turso_db_tables(client) 

    def try_conn_method_2(self):
        url_with_token = f"{self.connection_url}?authToken={self.auth_token}"
        with self.libsql_client.create_client_sync(url_with_token ) as client:
            self.create_turso_db_tables(client) 

    def try_conn_method_3(self):
        if self.connection_url.startswith('libsql://'):
            http_url = self.connection_url.replace('libsql://', 'https://')
            url_with_token = f"{http_url}?authToken={self.auth_token}"
            with self.libsql_client.create_client_sync(url_with_token ) as client:
                self.create_turso_db_tables(client)
    
    def is_db_connected(self):
        try:
            if self.use_turso:
                test_query = "SELECT 1"
                self.execute_turso_query(test_query)
                logger.info("Turso Database Connection Check Successfully.")
                return True
            else:
                if self.local_db_path and os.path.exists(self.local_db_path):
                    Connection = sqlite3.connect(self.local_db_path)
                    Connection.close()
                    logger.info("Local Database Connection Check Successfully.")
                    return True
        except Exception as e:
            logger.warning(f"Database Connection Check Faild: {e}")
            return False

    def create_turso_db_tables(self, client):
        try:
            client.execute(''' CREATE TABLE IF NOT EXISTS tracking_box_counts (
                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                total_box_in INTEGER DEFAULT 0,
                                total_box_out INTEGER DEFAULT 0,
                                date_time DATETIME DEFAULT CURRENT_TIMESTAMP) ''')  
            
            client.execute(''' CREATE TABLE IF NOT EXISTS tracking_cement_bag_counts (
                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                total_cement_bag_out INTEGER DEFAULT 0,
                                date_time DATETIME DEFAULT CURRENT_TIMESTAMP) ''') 
            logger.info("Turso Database Tables Created Successfully.")
        except Exception as e:
            logger.error(f"Error Occurred During Create Turso Database Tables: {e}")
            raise CustomException(e, sys)   
        
    def execute_turso_query(self, query, params=None):  
        conn_methods = [
            lambda: self.libsql_client.create_client_sync(self.connection_url, auth_token=self.auth_token),
            lambda: self.libsql_client.create_client_sync(f"{self.connection_url}?authtoken={self.auth_token}"),
        ]  
          
        if self.connection_url.startswith('libsql://'):
            http_url = self.connection_url.replace('libsql://', 'https://')
            conn_methods.append(lambda: self.libsql_client.create_client_sync(f"{http_url}?authToken={self.auth_token}"))
        
        last_error = None
        for method in conn_methods:
            try:
                with method() as client:
                    if params:
                        return client.execute(query, params)
                    else:
                        return client.execute(query)
            except Exception as e:
                last_error = e
                continue
        raise last_error

    def init_local_db(self):
        try:
            logger.info("Using Local SQLite Database.")
            self.create_local_db()
            self.create_local_db_tables()
            logger.info(f"Local Database [{self.local_db_path}] Initialize Successfully.")
        except Exception as e:
            logger.error(f"Error Occurred During Initialize Local Database: {e}")
            raise CustomException(e, sys) 

    def create_local_db(self):
        try:
            if not os.path.exists(self.local_db_path):
                os.makedirs(os.path.dirname(self.local_db_path), exist_ok=True)
                connection = sqlite3.connect(self.local_db_path)
                connection.close()
                logger.info(f"[{self.local_db_path}] - Database Create Successfully.")
            else:
                logger.info(f"[{self.local_db_path}] - Database Already Exists.")
        except Exception as e:
            logger.error(f"Error Occurred During Create Local Database: {e}")
            raise CustomException(e, sys)   

    def create_local_db_tables(self):
        try:
            connection = sqlite3.connect(self.local_db_path)
            cursor =  connection.cursor()
            cursor.execute(''' CREATE TABLE IF NOT EXISTS tracking_box_counts (
                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                total_box_in INTEGER DEFAULT 0,
                                total_box_out INTEGER DEFAULT 0,
                                date_time DATETIME DEFAULT CURRENT_TIMESTAMP) ''')
              
            cursor.execute(''' CREATE TABLE IF NOT EXISTS tracking_cement_bag_counts (
                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                total_cement_bag_out INTEGER DEFAULT 0,
                                date_time DATETIME DEFAULT CURRENT_TIMESTAMP) ''') 
     
            connection.commit()
            connection.close()
            logger.info("Local Database Tables Created Successfully.")
        except Exception as e:
            logger.error(f"Error Occurred During Create Local Database Tables: {e}")
            raise CustomException(e, sys)
        
    def save_box_counts(self, in_count, out_count):
        try:
            if self.use_turso and self.libsql_client:
                self.execute_turso_query(''' INSERT INTO tracking_box_counts (total_box_in, total_box_out)
                                             VALUES (?, ?) ''', [in_count, out_count])
                logger.info(f"Save To Turso Database - Box In: [{in_count}], Box Out: [{out_count}].")
            else:
                connection = sqlite3.connect(self.local_db_path)
                cursor = connection.cursor()
                cursor.execute(''' INSERT INTO tracking_box_counts (total_box_in, total_box_out)
                                   VALUES (?, ?) ''', (in_count, out_count))
                connection.commit()
                connection.close()
                logger.info(f"Save To Database [{self.local_db_path}] - Box In: [{in_count}], Box Out: [{out_count}].")
        except Exception as e:
            logger.error(f"Error Occurred During Saving Data To Database For Boxes Data: {e}")
            if self.use_turso and self.local_db_path:
                logger.info("Attempting Fallback To Local Database.")
                orig_use_turso = self.use_turso 
                self.use_turso =False
                try:
                    self.save_box_counts(in_count, out_count)
                except Exception as e:
                    logger.error(f"Fallback To Local Database Also Failed: {e}")    
                    self.use_turso = orig_use_turso
                    raise CustomException(e, sys) 
            else:
                raise CustomException(e, sys)    
        
    def save_cement_bag_counts(self, out_count):
        try:
            if self.use_turso and self.libsql_client:
                self.execute_turso_query(''' INSERT INTO tracking_cement_bag_counts (total_cement_bag_out)
                                             VALUES (?) ''', [out_count])
                logger.info(f"Save To Turso Database - Total Cement Bags Out: [{out_count}].")
            else:
                connection = sqlite3.connect(self.local_db_path)
                cursor = connection.cursor()
                cursor.execute(''' INSERT INTO tracking_cement_bag_counts (total_cement_bag_out)
                                   VALUES (?) ''', (out_count,))
                connection.commit()
                connection.close()
                logger.info(f"Save To Database [{self.local_db_path}] - Total Cement Bags Out: [{out_count}].")
        except Exception as e:
            logger.error(f"Error Occurred During Saving Data To Database For Cement Bags Data: {e}")
            if self.use_turso and self.local_db_path:
                logger.info("Attempting Fallback To Local Database.")
                orig_use_turso = self.use_turso 
                self.use_turso =False
                try:
                    self.save_cement_bag_counts(out_count)
                except Exception as e:
                    logger.error(f"Fallback To Local Database Also Failed: {e}")    
                    self.use_turso = orig_use_turso
                    raise CustomException(e, sys)
            else:
                raise CustomException(e, sys)  
    
    def fetch_database_records(self, table_name):
        if self.use_turso and self.libsql_client:
            try:
                query = f"SELECT * FROM {table_name} ORDER BY date_time DESC LIMIT {self.limit}"
                results = self.execute_turso_query(query)
                if results.rows:
                    columns = [col[0] for col in results.columns] if results.columns else []
                    df = pd.DataFrame(results.rows, columns=columns)
                    logger.info(f"Successfully Retrieve Recent Tracking Records From Turso Database And Table {table_name}.")
                    return df
                else:
                    return pd.DataFrame()
            except Exception as e:
                logger.error(f"Error occurred During Fetching Data From Turso Database Table [{table_name}]: {e}")
                return pd.DataFrame()
        else:
            try:
                connection = sqlite3.connect(self.local_db_path)
                query = f"SELECT * FROM {table_name} ORDER BY date_time DESC LIMIT {self.limit}"
                df = pd.read_sql_query(query, connection)
                logger.info(f"Successfully Retrieve Recent Tracking Records From Local Database [{self.local_db_path}] And Table [{table_name}].")
                connection.close()
                return df   
            except Exception as e:
                logger.error(f"Error occurred During Fetching Data From Local Database [{self.local_db_path}] And Table [{table_name}]: {e}")
                return pd.DataFrame()
                
    def view_database_records(self, table_names):
        try:
            for table_name in table_names:
                print("\n" + "="*80)
                print(f"Table: [{table_name}]")
                print("="*80)
                
                df = self.fetch_database_records(table_name)
                
                if df is None:
                    print(f"Error Occurred Fetching Database Records Return None For Table [{table_name}].")
                
                if df.empty:
                    print(f"No Records Found For Table: [{table_name}]")
                else:
                    print(df)
        
        except Exception as e:
            logger.error(f"Error Occurred During Viewing Database Records: {e}")
            raise CustomException(e, sys)

                    
