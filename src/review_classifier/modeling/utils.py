"""Any miscellaneous utilities/functions to assist the model
training workflow are to be contained here."""

import os
import hydra
import logging
# import pickle
import joblib
# import tensorflow as tf

logging.getLogger(__name__)

def export_model(model):
    model_file_path = os.path.\
        join(hydra.utils.get_original_cwd(),
            "models/amazon-review-classification-model")
    logging.info(f"showing model_file_path...{model_file_path}")
    # model.save(model_file_path)
    model_filename = "tfidf_log_rep.joblib"
    # current_pickle_path = os.getcwd()
    # logging.info(f"show current pickle path... {current_pickle_path}")
    # with open(pkl_filename, 'wb') as file:
    #     pickle.dump(model, file)
    joblib.dump(model, f"/Users/chekwei/Documents/Personal/AIAP/review-classification/src/models/{model_filename}")
    logging.info(f"model saved successfully...")


def export_vectorizer(vectorizer):
    vectorizer_filename = "vectorizer.joblib"
    joblib.dump(vectorizer, f"/Users/chekwei/Documents/Personal/AIAP/review-classification/src/models/{vectorizer_filename}")
    logging.info(f"vectorizer saved successfully...")

    # path_temp = current_working_dir + "/models/vectorizer.pickle"
    # pickle.dump(tfidf_vectorizer, open(path_temp, "wb"))

def load_model(path_to_model):
    # model_filename = "tfidf_log_rep.joblib"
    loaded_model = joblib.load(path_to_model)
    return loaded_model
