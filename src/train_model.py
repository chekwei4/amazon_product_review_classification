import os
import logging
import hydra
import mlflow
from collections import Counter
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import review_classifier as review_clf

@hydra.main(config_path="../conf/base", config_name="pipelines.yml")
def main(args):
    """This main function does the following:
    - load logging config
    - initialise experiment tracking (MLflow)
    - loads training, validation and test data
    - initialises model layers and compile
    - trains, evaluates, and then exports the model
    """

    logger = logging.getLogger(__name__)

    logger.info("running...2aaf")

    logger.info("Setting up logging configuration.")
    logger_config_path = os.path.\
        join(hydra.utils.get_original_cwd(),
            "conf/base/logging.yml")
    review_clf.general_utils.setup_logging(logger_config_path)

    mlflow_init_status, mlflow_run = review_clf.general_utils.\
        mlflow_init(
            args, setup_mlflow=args["train"]["setup_mlflow"],
            autolog=args["train"]["mlflow_autolog"])
    review_clf.general_utils.\
        mlflow_log(
            mlflow_init_status, "log_params",
            params=args["train"])

    X_res, test_tfidf_matrix, y_res, y_test = review_clf.modeling.data_loaders.\
        load_datasets(hydra.utils.get_original_cwd(), args)

    logging.info("back to train model.py...")
    logging.info(f'Resampled dataset shape {Counter(y_res)}')
    logging.info(test_tfidf_matrix.shape)

    model = review_clf.modeling.models.logistic_regression_model(args)

    logger.info("Training the model...")
    model.fit(X_res, y_res)
    logger.info("Model training done...")

    logger.info("Evaluating the model...")
    y_pred = model.predict(test_tfidf_matrix)
    logger.info("Model prediction done...")

    accu_score = accuracy_score(y_test, y_pred)
    logging.info(f"showing model accuracy score:...{accu_score}")
    logging.info("showing classification report:..")
    logging.info(classification_report(y_test, y_pred))
    # logger.info("Evaluating the model...")
    # test_loss, test_acc = model.evaluate(datasets["test"])

    # logger.info("Test Loss: {}, Test Accuracy: {}".\
    #     format(test_loss, test_acc))

    logger.info("Exporting the model...")
    review_clf.modeling.utils.export_model(model)

    if mlflow_init_status:
        artifact_uri = mlflow.get_artifact_uri()
        # artifact_uri = (f"gs://review-classification-artifacts/mlflow-tracking-server/{mlflow_run.info.run_id}/artifacts")
        logger.info("Artifact URI: {}".format(artifact_uri))
        review_clf.general_utils.\
            mlflow_log(
                mlflow_init_status, "log_params",
                params={"artifact_uri": artifact_uri})
        logger.info("Model training with MLflow run ID {} has completed.".
            format(mlflow_run.info.run_id))
        mlflow.end_run()
    else:
        logger.info("Model training has completed.")


if __name__ == "__main__":
    main()
