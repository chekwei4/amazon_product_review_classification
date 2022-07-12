import review_classifier as review_clf
import review_classifier_fastapi as review_clf_fapi


PRED_MODEL = review_clf.modeling.utils.load_model(
    review_clf_fapi.config.SETTINGS.PRED_MODEL_PATH)
