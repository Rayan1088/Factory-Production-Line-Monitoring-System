import sys
import time
from src.logger import logging
from src.exception import CustomException
import multiprocessing
from contextlib import contextmanager
from src.tracker_1 import TrackerClass1
from src.tracker_2 import TrackerClass2

logger = logging.getLogger("multiprocessing")

class MultiprocessingForTrackers:
    
    def __init__(self, use_turso_db=False):
        self.use_turso_db = use_turso_db
    
    def task_1(self):
        try:
            logger.info("Task 1 Starting.")
            tracker = TrackerClass1(use_turso_db=self.use_turso_db)
            tracker.detection_and_tracking_1_for_local_system()
            time.sleep(3)
            logger.info("Task 1 Completed Successfully.") 
            return True
        
        except Exception as e:
            logger.error(f"Error Occurred In Task_1(): {e}")
            raise CustomException(e, sys)
        
    def task_2(self):
        try:
            logger.info("Task 2 Starting.")
            tracker = TrackerClass2(use_turso_db=self.use_turso_db)  
            tracker.detection_and_tracking_2_for_local_system()
            time.sleep(4)
            logger.info("Task 2 Completed Successfully.") 
            return True
        
        except Exception as e:
            logger.error(f"Error Occurred In Task_2(): {e}")
            raise CustomException(e, sys)

    def monitor_all_processes(self, processes, timeout=None):
        process_start_time = time.time()
        try:
            while any(process.is_alive() for process in processes):
                if timeout and (time.time() - process_start_time) > timeout:
                    logger.warning(f"The Processes Exceeded Timeout Of [{timeout}]s.")
                    for process in processes:
                        if process.is_alive():
                            logger.warning(f"Terminating The Process: [{process.name}].")
                            process.terminate()
                            process.join(timeout=5)  # Wait time
                            
                            if process.is_alive():
                                logger.warning(f"Force Killing Process: [{process.name}]")                  
                                process.kill()
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
                        
                    logger.info(f"Process {process.name} Cleaned Up Successfully")
                    
        except Exception as e:
            logger.error(f"Error Occurred In clean_up_processes() Function: {e}")
            raise CustomException(e, sys)

    @contextmanager
    def process_manager(self, processes):
        try:
            yield processes
        finally:
            if processes:
                self.clean_up_processes(processes)
                logger.info("Process Cleanup Completed Successfully.")
      
    def check_process_results(self, processes):
        success_count = 0
        results = {'success_count':0, 'failed_processes':[]}
        
        for process in processes:
            process.join()
            
            if process.exitcode == 0:
                logger.info(f"[{process.name}] Completed Successfully.")
                success_count += 1
            else: 
                logger.error(f"[{process.name}] Failed With Exit Code [{process.exitcode}].")
                results['failed_processes'].append({
                    'name': process.name,
                    'exit_code': process.exitcode
                })

        results['success_count'] = success_count

        # Report results
        if success_count == len(processes):
            logger.info("All Processes Completed Successfully.")
        else:
            logger.error(f"Only [{success_count}/{len(processes)}] Processes Completed Successfully.")
        
        return results            
                
    def run_tasks(self, timeout=None):
        logger.info("Starting The Multiprocessing Tracker System")
        processes = [
            multiprocessing.Process(target=self.task_1, name="Tracker_01"),
            multiprocessing.Process(target=self.task_2, name="Tracker_02")                        
        ]
        
        try:
            with self.process_manager(processes):
                logger.info("Starting All Tracker Processes.")
                for process in processes:
                    process.start()
                    logger.info(f"Stsrted Process: [{process.name}] - PID: [{process.pid}]")
                    
                self.monitor_all_processes(processes=processes, timeout=timeout)  
                
                results = self.check_process_results(processes)  

                return results['success_count'] == len(processes)
          
        except Exception as e:
            logger.error(f"Error Occurred In run_tasks() Function: {e}")
            raise CustomException(e, sys)
  