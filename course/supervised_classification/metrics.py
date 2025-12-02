import pandas as pd
from course.utils import find_project_root
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def metric_report(y_test_path, y_pred_path, report_path):
    y_test = pd.read_csv(y_test_path)
    y_pred = pd.read_csv(y_pred_path)
    """Create a pandas data frame called report which contains your classifier results"""
    report = pd.DataFrame({
        "accuracy": [accuracy_score(y_test, y_pred)],
        "precision": [precision_score(y_test, y_pred, average='weighted')],
        "recall": [recall_score(y_test, y_pred, average='weighted')],
        "f1-score": [f1_score(y_test, y_pred, average='weighted')],
    })
    report.to_csv(report_path, index=False)


def metric_report_lda():
    base_dir = find_project_root()
    y_test_path = base_dir / 'data_cache' / 'energy_y_test.csv'
    y_pred_path = base_dir / 'data_cache' / 'models' / 'lda_y_pred.csv'
    report_path = base_dir / 'data_cache' / 'vignettes' / 'supervised_classification' / 'lda.csv'
    metric_report(y_test_path, y_pred_path, report_path)


def metric_report_qda():
    base_dir = find_project_root()
    y_test_path = base_dir / 'data_cache' / 'energy_y_test.csv'
    y_pred_path = base_dir / 'data_cache' / 'models' / 'qda_y_pred.csv'
    report_path = base_dir / 'data_cache' / 'vignettes' / 'supervised_classification' / 'qda.csv'
    metric_report(y_test_path, y_pred_path, report_path)
