import sys
import time
import multiprocessing
from src.logger import logging
from src.utils import load_config
from src.exception import CustomException
from src.tracker_1 import TrackerClass1
from src.tracker_2 import TrackerClass2
from src.databaseManager import DataBaseManagerClass

# Create a logger instance
logger = logging.getLogger("main")

class MultiprocessingForTrackers:
    
    def __init__(self, config_path="config.yaml"):
        
        # Load configuration from YAML file
        self.config = load_config(config_path)

    def task_1(self):
        try:
            logger.info("Task 1 Starting.")
            db = DataBaseManagerClass(self.config['DB_PATH'], self.config['LIMIT'])
            db.create_dbs()
            tracker = TrackerClass1()
            tracker.detection_and_tracking_1_for_local_system()
            time.sleep(3)
            logger.info("Task 1 Completed Successfully.") 
            return True
        
        except Exception as e:
            logger.error(f"Error Occurred In Task 1: {e}")
            raise CustomException(e, sys)
        
    def task_2(self):
        try:
            logger.info("Task 2 Starting.")
            db = DataBaseManagerClass(self.config['DB_PATH'], self.config['LIMIT'])
            db.create_dbs()
            tracker = TrackerClass2()
            tracker.detection_and_tracking_2_for_local_system()
            time.sleep(4)
            logger.info("Task 2 Completed Successfully.") 
            return True
        
        except Exception as e:
            logger.error(f"Error Occurred In Task 2: {e}")
            raise CustomException(e, sys)

    def monitor_all_processes(self, processes, timeout=None):
        process_start_time = time.time()
        try:
            while any(process.is_alive() for process in processes):
                if timeout and (time.time() - process_start_time) > timeout:
                    logger.warning(f"The Processes Exceeded Timeout Of [{timeout}]s.")
                    for process in processes:
                        if process.is_alive():
                            logger.warning(f"Terminating The [{process.name}].")
                            process.terminate()                    
                    break
                time.sleep(1)
            
            # Wait for all processes to complete
            still_alive = [p for p in processes if p.is_alive()]
            if still_alive:    
                for p in still_alive:        
                    logger.warning(f"[{p.name}] Still Running After Monitoring.")
        
        except Exception as e:
            logger.error(f"Error Occurred In monitor_all_processes() Function: {e}")
            raise CustomException(e, sys)

    def clean_up_processes(self, processes):
        try:
            for process in processes:
                if process.is_alive():
                    logger.warning(f"Terminating The [{process.name}].")
                    process.terminate() 
                    process.join(timeout=5) # This timeout for cleanup 
                
                    if process.is_alive():
                        logger.warning(f"Forcefully Killing [{process.name}].")
                        process.kill()
                        process.join()

        except Exception as e:
            logger.error(f"Error Occurred In clean_up_processes() Function: {e}")
            raise CustomException(e, sys)

    def processing_unit(self):
        logger.info("Starting The Main Process!!!!!!!!!!!!!!!")
        processes = []
        
        try:
            process_1 = multiprocessing.Process(target=self.task_1, name="Task_01")
            process_2 = multiprocessing.Process(target=self.task_2, name="Task_02")
            processes = [process_1,process_2]
            
            # Start processing
            logger.info("Starting All Processes!!!!!!!!!!!!!!!!!!")
            for process in processes:
                process.start()
                logger.info(f"[{process.name}] Has Started.")
            
            self.monitor_all_processes(processes, timeout=None) # Monitor without timeout       
            
            # Wait for completion
            success_count = 0
            for process in processes:
                process.join()
                if process.exitcode == 0:
                    logger.info(f"[{process.name}] Completed Successfully.")
                    success_count += 1
                else: 
                    logger.error(f"[{process.name}] Failed With Exit Code [{process.exitcode}].")

            # Report results
            if success_count == len(processes):
                logger.info("All Processes Completed Successfully.")
                return True
            else:
                logger.error(f"Only [{success_count}/{len(processes)}] Processes Completed Successfully.")
                return False
        
        except Exception as e:
            logger.error(f"Error Occurred In processing_unit() Function: {e}")
            raise CustomException(e, sys)
        finally:
            if processes:
                self.clean_up_processes(processes)
                logger.info("Process Cleanup Successfully Completed.")
                
if __name__ == '__main__':
    try:
        # Load configuration from YAML file
        config = load_config(config_path='config.yaml')
        multiprocessing.set_start_method('spawn', force=True)
        run = MultiprocessingForTrackers().processing_unit()
        
        if run:
            logger.info("processing_unit() Function Run Completed Successfully From [MultiprocessingForTrackers()] Class.")
            db = DataBaseManagerClass(config['DB_PATH'] config['LIMIT'])
            db.view_database_records()
            sys.exit(0)
        else:
            logger.info("processing_unit() Function Run Completed With Errors From [MultiprocessingForTrackers()] Class.")
            sys.exit(1)
        
    except KeyboardInterrupt:
        logger.info("processing_unit() Function Run Interrupted By The User")
        sys.exit(130)

    except Exception as e:
        logger.error(f"Unexpected Error Occurred In if __name__ == '__main__': {e}")
        sys.exit(130)

                    
