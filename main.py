import sys
import multiprocessing
from src.logger import logging
from src.utils import load_config
from src.multiprocessing import MultiprocessingForTrackers
from src.databaseManager import DataBaseManagerClass

logger = logging.getLogger("multiprocessing")

config = load_config(config_path='config.yaml')

def init_database_for_multiprocessing(use_turso_db=False):
    if use_turso_db:
        # Turso Database Initialization
        logger.info("Initializing Turso Database Connection For Multiprocessing.")
        db = DataBaseManagerClass(local_db_path=config['LOCAL_DB_PATH'], # Fallback
                                    turso_db_url=config['TURSO_DB_URL'], 
                                    turso_db_token=config['TURSO_DB_TOKEN'],
                                    limit=config['LIMIT'])
    else:
        # Local Database Initialization
        logger.info("Initializing Local Database Connection For Multiprocessing.")
        db = DataBaseManagerClass(local_db_path=config['LOCAL_DB_PATH'], limit=config['LIMIT'])
        
    if db.is_db_connected():
        logger.info("Database Connection Test Sucessfully.")
    else:
        logger.warning("Database Connection Test Failed.")
        raise RuntimeError("Failed To Establish Database Connection")

    return db

def main_run(use_turso_db=False, timeout=None):
    try:
        db = init_database_for_multiprocessing(use_turso_db=use_turso_db)
        
        multiprocessing.set_start_method('spawn', force=True)
        
        run = MultiprocessingForTrackers(use_turso_db=use_turso_db)
        success = run.run_tasks(timeout=timeout)
        
        if success:
            logger.info("All Process Completed Successfully.")
            db.view_database_records(config['DB_TABLE_NAMES'])
            return 0
        else:
            logger.error("Some Tracker Process Failed.")
    
    except KeyboardInterrupt:
        logger.warning("Process Interrupted By User.")
        return 130
    except Exception as e:
        logger.error(f"Unexpected Error In Main: {e}")
        return 130
                                    
if __name__ == '__main__':
    exit_code = main_run(use_turso_db = True, timeout = None)
    sys.exit(exit_code)
                    
