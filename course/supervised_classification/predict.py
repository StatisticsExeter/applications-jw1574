import joblib
import pandas as pd
from course.utils import find_project_root


def predict(model_path, X_test_path, y_pred_path, y_pred_prob_path):
    # Load model and test data
    model = joblib.load(model_path)
    X_test = pd.read_csv(X_test_path)

    # Predict classes
    y_pred = model.predict(X_test)
    pd.DataFrame({"predicted_built_age": y_pred}).to_csv(y_pred_path, index=False)

    # Predict probabilities
    y_pred_proba = model.predict_proba(X_test)
    pd.DataFrame({"predicted_built_age": y_pred_proba[:, 1]}).to_csv(y_pred_prob_path, index=False)


def pred_lda():
    base_dir = find_project_root()
    model_path = base_dir / 'data_cache' / 'models' / 'lda_model.joblib'
    X_test_path = base_dir / 'data_cache' / 'energy_X_test.csv'
    y_pred_path = base_dir / 'data_cache' / 'models' / 'lda_y_pred.csv'
    y_pred_prob_path = base_dir / 'data_cache' / 'models' / 'lda_y_pred_prob.csv'  
    predict(model_path, X_test_path, y_pred_path, y_pred_prob_path)


def pred_qda():
    base_dir = find_project_root()
    model_path = base_dir / 'data_cache' / 'models' / 'qda_model.joblib'
    X_test_path = base_dir / 'data_cache' / 'energy_X_test.csv'
    y_pred_path = base_dir / 'data_cache' / 'models' / 'qda_y_pred.csv'
    y_pred_prob_path = base_dir / 'data_cache' / 'models' / 'qda_y_pred_prob.csv'  
    predict(model_path, X_test_path, y_pred_path, y_pred_prob_path)
