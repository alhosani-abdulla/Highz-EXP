from highz_exp import file_load
from datetime import datetime
import sys, os, logging

pbase = os.path.basename
pjoin = os.path.join

def setup_logging(level=logging.INFO, output_file=False):
    """Setup logging configuration"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    if output_file:
        file_name = f'file_compressor_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        logging.basicConfig(filename=file_name)

def __main__():
    setup_logging(level=logging.DEBUG, output_file=True)
    logging.info("Starting file compression process...")

    help_use = """
    This script condenses multiple .npy files in a specified directory into a single file.
    Please specify the directory containing the .npy files saved in one day.
    
    Usage:
        python file_compressor.py /home/peterson/Data/INDURANCE/20251105/ /home/peterson/Data/INDURANCE/compressed/
        
    This then compressses files in /home/peterson/Data/INDURANCE/20251105/
    And saves the compressed files in /home/peterson/Data/INDURANCE/compressed/20251105/
    
    Notice that this must be run in a date directory that contains multiple subdirectories, each representing an observation cycle."""
        
    args = sys.argv[1:]
    if len(args) < 2:
        print("Insufficient arguments provided.")
        print(help_use)
        sys.exit(1)
    
    input_dir = args[0]

    if not os.path.isdir(input_dir):
        logging.error(f"The specified directory does not exist: {input_dir}")
        sys.exit(1)

    date = pbase(args[0])
    output_dir = os.path.join(args[1], date)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Created output directory: {output_dir}")

    for hour in os.listdir(input_dir):
        hour_path = os.path.join(input_dir, hour)
        for i in range(0, 8):
            try:
                _ = file_load.condense_npy_by_timestamp(
                    dir_path=hour_path,
                    output_dir=pjoin(output_dir, hour),
                    pattern=f'*state{i}*.npy',
                    time_regex=r'_(\d{6})(?:_|$)',
                    use_pickle=False
                )
                logging.info(f"Successfully condensed state{i} files in {hour_path} into {output_dir}")
            except Exception as e:
                logging.error(f"Error during file condensation for state{i} in {hour_path}: {e}")
#! This section is commented out because we've redesigned our relay board and calibration system in Dec 2025. 
#! And we no longer have the OC states.
        # try:
        #     _ = file_load.condense_npy_by_timestamp(
        #         dir_path=hour_path,
        #         output_dir=output_dir,
        #         pattern='*stateOC*.npy',
        #         time_regex=r'_(\d{6})(?:_|$)',
        #         use_pickle=False
        #     )
        #     logging.info(f"Successfully condensed stateOC files in {hour_path} into {output_dir}")
        # except Exception as e:
        #     logging.error(f"Error during file condensation for stateOC in {hour_path}: {e}")

if __name__ == "__main__":
    __main__()
    logging.info("File compression process completed.")
