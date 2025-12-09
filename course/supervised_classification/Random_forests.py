import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from course.utils import find_project_root
from course.supervised_classification.classify import fit_classifier
from course.supervised_classification.metrics import metric_report


def fit_rf():
    base_dir = find_project_root()

    X_train_path = base_dir / 'data_cache' / 'energy_X_train.csv'
    y_train_path = base_dir / 'data_cache' / 'energy_y_train.csv'
    model_path = base_dir / 'data_cache' / 'models' / 'rf_model.joblib'

    classifier = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42)

    fit_classifier(X_train_path, y_train_path, model_path, classifier)


fit_rf()


def predict_rf():
    base_dir = find_project_root()

    model_path = base_dir / 'data_cache' / 'models' / 'rf_model.joblib'
    X_test_path = base_dir / 'data_cache' / 'energy_X_test.csv'
    y_pred_path = base_dir / 'data_cache' / 'models' / 'rf_y_pred.csv'

    model = joblib.load(model_path)
    X_test = pd.read_csv(X_test_path)

    preds = model.predict(X_test)
    pd.DataFrame({"built_age": preds}).to_csv(y_pred_path, index=False)


predict_rf()


def metric_report_rf():
    base_dir = find_project_root()

    y_test_path = base_dir / 'data_cache' / 'energy_y_test.csv'
    y_pred_path = base_dir / 'data_cache' / 'models' / 'rf_y_pred.csv'
    report_path = base_dir / 'data_cache' / 'vignettes' / 'supervised_classification' / 'rf.csv'

    metric_report(y_test_path, y_pred_path, report_path)


metric_report_rf()
