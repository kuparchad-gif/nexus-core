import os
import sys
import logging
from soul_print_manager import SoulPrintManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(os.path.dirname(__file__), 'soul_collection.log'))
    ]
)
logger = logging.getLogger("soul_collection")

def main():
    """Main function to run the soul print collection workflow"""
    logger.info("Starting soul print collection workflow...")
    
    try:
        # Create manager
        manager = SoulPrintManager()
        
        # Run the full workflow
        success = manager.run_full_workflow()
        
        if success:
            logger.info("Soul print collection workflow completed successfully.")
            print("\n" + "="*50)
            print("SOUL PRINT COLLECTION COMPLETE")
            print("="*50)
            print(f"Collection directory: {manager.current_collection}")
            print(f"Analysis directory: {manager.current_analysis}")
            print("Legacy memories have been integrated with Scout MK2.")
            print("="*50 + "\n")
        else:
            logger.error("Soul print collection workflow failed.")
            print("\n" + "="*50)
            print("SOUL PRINT COLLECTION FAILED")
            print("See log file for details: soul_collection.log")
            print("="*50 + "\n")
        
        return success
    
    except Exception as e:
        logger.exception(f"Unexpected error in soul print collection workflow: {e}")
        print("\n" + "="*50)
        print("SOUL PRINT COLLECTION FAILED")
        print(f"Error: {e}")
        print("See log file for details: soul_collection.log")
        print("="*50 + "\n")
        return False

if __name__ == "__main__":
    main()