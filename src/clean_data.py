import os
import logging
import pathlib
import re
import hydra

import review_classifier as review_clf

FILE_NAME = "reviews_Clothing_Shoes_and_Jewelry_5.json.gz"

@hydra.main(config_path="../conf/base", config_name="pipelines.yml")
def main(args):
    """Main programme to read in raw data files and process them."""

    logger = logging.getLogger(__name__)
    logger.info("Setting up logging configuration.")
    logger_config_path = os.path.join(
        hydra.utils.get_original_cwd(), "conf/base/logging.yml"
    )
    review_clf.general_utils.setup_logging(logger_config_path)

    raw_data_dirs_list = args["data_prep"]["raw_dirs_paths"]
    processed_data_path = args["data_prep"]["processed_data_path"]

    logging.info("THIS IS FROM clean_data.py...")

    logging.info(f"CURRENT PWD: {os.getcwd()}")

    logging.info("PRINTING RAW DATA PATH BELOW...")
    logging.info(raw_data_dirs_list)
    logging.info("PRINTING RAW DATA PATH ABOVE...")

    logging.info("PRINTING PROCESSED DATA PATH BELOW...")
    logging.info(processed_data_path)
    logging.info("PRINTING PROCESSED DATA PATH ABOVE...")

    for raw_data_dir in raw_data_dirs_list:
        raw_data_dir = os.path.join(hydra.utils.get_original_cwd(), raw_data_dir)
        processed_data_path = os.path.join(hydra.utils.get_original_cwd(), processed_data_path)
        logging.info(f"Processing raw text files from: {raw_data_dir}")
        logging.info(f"file name is... {FILE_NAME}")
        # try:
        logging.info(f"Processing text file: {raw_data_dir}/{FILE_NAME}")
        df_clean = review_clf.data_prep.process_text.process_file(raw_data_dir)
        logging.info("lemmatizing completed...")
        logging.info(f"lemmatized shape: {df_clean.shape}")
        df_clean.to_csv(os.path.join(processed_data_path, "cleaned_review.csv"))
        logging.info("saving to polyaxon worksapce completed...")

            # out_filename = re.sub(raw_data_dir, processed_data_path, str("processed.csv"))
            # os.makedirs(os.path.dirname(out_filename), exist_ok=True)
            # tf.io.write_file(out_filename, df_clean)
            # logger.debug("Processed text file exported: {}".
            #                 format(out_filename))
        # except:
            # logger.error(f"Error encountered while processing file: {raw_data_dir}{FILE_NAME}")

    logging.info("Data preparation pipeline has completed.")


if __name__ == "__main__":
    main()
