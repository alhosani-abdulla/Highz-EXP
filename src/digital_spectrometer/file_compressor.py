from datetime import datetime
import sys, os, logging, argparse

from highz_exp.file_load import LegacyDSFileLoader
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

def main():
    # 1. Initialize Logging
    setup_logging(level=logging.WARNING, output_file=True)
    logging.info("Starting file compression process...")

    # 2. Setup Argparse
    parser = argparse.ArgumentParser(
        description="Condense multiple .npy files from observation cycles into a single file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Usage Example:
            python file_compressor.py /data/20251105/ /data/compressed/

            Note: The script uses the basename of the input directory as the date folder 
            inside the output directory."""
    )

    # Positional Arguments
    parser.add_argument("input_dir", type=str, 
                        help="Directory containing the .npy files in legacy format for one day.")
    parser.add_argument("output_root", type=str, 
                        help="Root directory where compressed results are saved.")

    # Optional Arguments
    parser.add_argument("--pickle", action="store_true",
                        help="Save using pickle instead of numpy format.")

    args = parser.parse_args()

    # 3. Path Logic & Validation
    input_path = os.path.abspath(args.input_dir)
    
    if not os.path.isdir(input_path):
        logging.error(f"The specified directory does not exist: {input_path}")
        sys.exit(1)

    # Extract date from input directory (e.g., /.../20251105/ -> 20251105)
    # Using os.path.basename(os.path.normpath()) handles trailing slashes correctly
    date_folder = os.path.basename(os.path.normpath(input_path))
    final_output_dir = os.path.join(args.output_root, date_folder)

    if not os.path.exists(final_output_dir):
        os.makedirs(final_output_dir)
        logging.info(f"Created output directory: {final_output_dir}")

    # 4. Execution
    for hour in os.listdir(input_path):
        hour_path = os.path.join(input_path, hour)
        for i in range(0, 8):
            loader = LegacyDSFileLoader(hour_path)
            try:
                loader.condense_npy_by_timestamp(output_dir=pjoin(final_output_dir, hour), 
                    pattern=f'*state{i}*.npy', use_pickle=args.pickle)
            except Exception as e:
                logging.error(f"Error during file condensation for state{i} in {hour_path}: {e}")

if __name__ == "__main__":
    main()
                