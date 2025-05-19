import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, balanced_accuracy_score
from sklearn.linear_model import LinearRegression
from itertools import product
import warnings
from sklearn.base import clone
import pickle as pkl
import os
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import f1_score
import numpy as np
from scipy.optimize import dual_annealing
import copy
import logging


def configure_logging(logger, output_path):
    """Configures logging for a new dataset by creating a new log file."""
    # Remove old handlers to prevent duplicate logs
    while logger.hasHandlers():
        logger.handlers.clear()

    # Create new log file per dataset
    log_file = os.path.join(output_path, "gpt_training_submit.log")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

    # Console handler (optional, for debugging)
    # console_handler = logging.StreamHandler()
    # console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    # Attach handlers
    logger.addHandler(file_handler)
    # logger.addHandler(console_handler)  # Comment this if you don't want console output

    # Set log level
    logger.setLevel(logging.INFO)

    logger.info(f"Logging configured for: {log_file}")
    print(f"Logging to {log_file}")  # Ensures visibility in notebooks

    return logger

def calculate_f1(y_true, y_pred):
    """
    Calculate F1 score for the given true and predicted labels.
    """
    tp = 0
    fp = 0
    fn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt > 0 and yp > 0:
            tp += 1
        elif yt == 0 and yp > 0:
            fp += 1
        elif yt > 0 and yp == 0:
            fn += 1

    if tp + fp + fn == 0:
        return 0.0

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0

    return 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
def calculate_balanced_accuracy(y_true, y_pred):
    """
    Calculate balanced accuracy for the given true and predicted labels.
    """
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt > 0 and yp > 0:
            tp += 1
        elif yt == 0 and yp > 0:
            fp += 1
        elif yt > 0 and yp == 0:
            fn += 1
        else:
            tn += 1

    sensitivity = tp / (tp + fn) if tp + fn > 0 else 0
    specificity = tn / (tn + fp) if tn + fp > 0 else 0

    return (sensitivity + specificity) / 2 if sensitivity + specificity > 0 else 0

def extract_features(volumes, variant='A', last_n=5):
    x_all = np.arange(len(volumes)).reshape(-1, 1)
    y_all = np.array(volumes).reshape(-1, 1)

    slope_all = LinearRegression().fit(x_all, y_all).coef_[0][0]
    slope_last = LinearRegression().fit(x_all[-last_n:], y_all[-last_n:]).coef_[0][0]
    mean_last_two = np.mean(volumes[-2:])

    features = {
        'A': [mean_last_two, slope_last, slope_all],
        'B': [mean_last_two, slope_last],
        'C': [mean_last_two, slope_all],
        'D': [mean_last_two],
    }
    return features[variant]

class ThresholdModel(BaseEstimator, ClassifierMixin):
    def __init__(self, custom_loss_fn, search_grid=None, min_t=5, max_t=50,
                 alpha_late=2.0, beta_fn=20.0, gamma_fp=1.0):
        self.thresholds = None
        self.search_grid = search_grid if search_grid is not None else np.linspace(-6, 0, 50)
        self.min_t = min_t
        self.max_t = max_t
        self.alpha_late = alpha_late
        self.beta_fn = beta_fn
        self.gamma_fp = gamma_fp
        self.custom_loss_fn = custom_loss_fn

    def fit(self, X_seqs, y_true):
        """
        X_seqs: List of sequences (one per patient), each sequence is a list of feature vectors (t=5 to 30)
        y_true: True replanning labels (0 if no replanning, or fraction of replanning)
        """
        
        bounds = [(-6, 0) for _ in range(len(X_seqs[0][0]))]
        best_thresh = dual_annealing(lambda x: self.custom_loss_fn(y_true, [self._predict_single(x_seq, x) for x_seq in X_seqs], self.alpha_late, self.beta_fn, self.gamma_fp),
                                            bounds=bounds, maxiter=500).x


        self.thresholds = np.array(best_thresh)
        return self

    def _predict_single(self, x_seq, thresholds):
        """
        Returns the earliest time t (as fraction index) where all features < thresholds.
        """
        for t, x in enumerate(x_seq, start=self.min_t):
            if (np.array(x) < thresholds).all():
                return t
        return 0

    def predict(self, X_seqs):
        return [self._predict_single(x_seq, self.thresholds) for x_seq in X_seqs]


def custom_loss(y_true, y_pred, alpha_late, beta_fn, gamma_fp):
    total = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 0:
            if yp > 0:
                total += beta_fn # False positive
        else:
            if yp == 0:
                total += gamma_fp # False negative
            elif yp > yt:
                total += alpha_late * (yp - yt)**2 # Late replanning
            else:
                total += (1-alpha_late)*(yt - yp)**2 # Early replanning
    return total # / len(y_true)


def main():
    # Load the data
    DATA_PATH = '/storage/homefs/tf24s166/code/Volumetric_ART/data'
    all_data = pkl.load(open(os.path.join(DATA_PATH, 'all_data.pkl'), 'rb'))
    for patient_id in all_data.keys():
        all_data[patient_id]['data'] = all_data[patient_id]['data'].drop([index for index in all_data[patient_id]['data'].index if 'REPLAN' in index or 'RT' in index or 'PT' in index])
    # print(np.array(all_data['001']['data']['volume difference to pCT [cm^3]'].values))

    patients = [{'patient_id': int(patient_id.lstrip('0'))-1,
                'volume': all_data[patient_id]['data']['volume difference to pCT [%]'].values,
                'label': all_data[patient_id]['label'][0]
                }
                for patient_id in all_data.keys()]
    # Setup
    feature_variants = ['A', 'B', 'C', 'D']

    model_types = {
        'THRESH': ThresholdModel(custom_loss_fn=custom_loss),
        # 'RF': RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42),
        # 'LR': LogisticRegression(max_iter=1000)
    }
    # loss_params_grid = list(product([1, 2], [2, 5], [0.5, 1]))  # (alpha_late, beta_fn, gamma_fp)
    loss_params_grid = list(product([0.5, 0.7, 0.8, 0.9], [300, 600, 900], [300, 600, 900]))  # (alpha_late, beta_fn, gamma_fp)

    outer_cv = StratifiedKFold(n_splits=7, shuffle=True, random_state=1)
    y_all    = [p['label'] for p in patients]

    summary_rows = []

    for train_idx, test_idx in outer_cv.split(np.zeros(len(y_all)), y_all):
        logger.info(f'Outer CV fold: {train_idx} / {test_idx}')
        train_patients = [patients[i] for i in train_idx]
        test_patients  = [patients[i] for i in test_idx]

        # best-of-grid trackers
        best_f1, best_ba = -np.inf, -np.inf
        best_conf, best_model = None, None   # (variant, α,β,γ)

        for variant in feature_variants:
            logger.info(f'  Feature variant: {variant}')
            for (alpha_late, beta_fn, gamma_fp) in loss_params_grid:
                if beta_fn <= gamma_fp:
                    continue

                # ----- INNER  LOOCV -----
                tp = fp = fn = tn = 0
                for loo_idx, val_patient in enumerate(train_patients):
                    # print(f'  Inner CV fold: {loo_idx} / {len(train_patients)}')

                    # build inner-training set  = all except val_patient
                    inner_train   = [p for k, p in enumerate(train_patients) if k != loo_idx]
                    y_inner_train = [p['label'] for p in inner_train]

                    X_inner_train = [[extract_features(p['volume'][:t], variant)
                                    for t in range(5, 36)] for p in inner_train]
                    X_val         = [[extract_features(val_patient['volume'][:t], variant)
                                    for t in range(5, 36)]]

                    model = ThresholdModel(custom_loss_fn=custom_loss,
                                        alpha_late=alpha_late,
                                        beta_fn   =beta_fn,
                                        gamma_fp  =gamma_fp)
                    model.fit(X_inner_train, y_inner_train)
                    y_pred_val = model.predict(X_val)[0]
                    y_true_val = val_patient['label']

                    # accumulate confusion counts
                    if y_true_val > 0 and y_pred_val > 0:
                        tp += 1
                    elif y_true_val == 0 and y_pred_val > 0:
                        fp += 1
                    elif y_true_val > 0 and y_pred_val == 0:
                        fn += 1
                    else:
                        tn += 1

                # compute mean F1 & BA over the LOOCV rounds
                precision = tp / (tp+fp) if tp+fp else 0
                recall    = tp / (tp+fn) if tp+fn else 0
                f1_mean   = 2*precision*recall/(precision+recall) if precision+recall else 0
                sens      = recall
                spec      = tn / (tn+fp) if tn+fp else 0
                ba_mean   = (sens+spec)/2 if sens+spec else 0

                # choose best config

                if f1_mean > best_f1 or (f1_mean == best_f1 and ba_mean > best_ba):
                    logger.info(f'    Best config: {variant}, {alpha_late}, {beta_fn}, {gamma_fp} with f1: {f1_mean:.4f} and ba: {ba_mean:.4f}')
                    best_f1, best_ba = f1_mean, ba_mean
                    best_conf        = (variant, alpha_late, beta_fn, gamma_fp)
                elif f1_mean == best_f1 and ba_mean == best_ba:
                    logger.info(f'    Equal config: {variant}, {alpha_late}, {beta_fn}, {gamma_fp} with f1: {f1_mean:.4f} and ba: {ba_mean:.4f}')

        # ----- REFIT best config on *all* outer-training patients -----
        variant, alpha_late, beta_fn, gamma_fp = best_conf
        X_train_full  = [[extract_features(p['volume'][:t], variant)
                        for t in range(5, 36)] for p in train_patients]
        y_train_full  = [p['label'] for p in train_patients]

        best_model = ThresholdModel(custom_loss_fn=custom_loss,
                                    alpha_late=alpha_late,
                                    beta_fn   =beta_fn,
                                    gamma_fp  =gamma_fp)
        best_model.fit(X_train_full, y_train_full)

        # ----- EVALUATE on outer-test patients -----
        y_true, y_pred, det_time = [], [], []
        for patient in test_patients:
            X_seq = [[extract_features(patient['volume'][:t], variant)
                    for t in range(5, 36)]]
            pred  = best_model.predict(X_seq)[0]
            y_true.append(patient['label'])
            y_pred.append(pred)
            det_time.append(pred)

        summary_rows.append({
            'model'    : 'THRESH',
            'variant'  : variant,
            'params'   : (alpha_late, beta_fn, gamma_fp),
            'f1_outer' : calculate_f1(y_true, y_pred),
            'ba_outer' : calculate_balanced_accuracy(y_true, y_pred),
            'avg_det'  : np.mean([t for t in det_time if t > 0])
        })
        

# --------- results ----------
    summary_df = pd.DataFrame(summary_rows)
    save_folder = os.path.join(output_path, 'plots')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    summary_df.to_csv(os.path.join(save_folder, 'summary_submit.csv'), index=False)
    logger.info(f'\n{summary_df}')


    # Select best combination by lowest mean custom loss
    grouped = summary_df.groupby(['model', 'variant', 'params']).mean()
    # Most frequent feature param combination
    grouped['count'] = summary_df.groupby(['model', 'variant', 'params']).size()

    best_combo = grouped['count'].idxmax()
    best_model_name, best_variant, best_params = best_combo
    best_combco_avg_f1 = summary_df[(summary_df['model'] == best_model_name) & (summary_df['variant'] == best_variant) & (summary_df['params'] == best_params)]['f1_outer'].mean()
    logger.info(f"Best overall: {best_model_name} with features {best_variant} and params {best_params} with mean f1 {best_combco_avg_f1:.4f}")

    # Retrain final model on all patients
    X_full =  [[extract_features(p['volume'][:t], best_variant) for t in range(5, 36)] for p in patients]
    y_full = [p['label'] for p in patients]
    final_model = ThresholdModel(custom_loss_fn=custom_loss)
    final_model.alpha_late = best_params[0]
    final_model.beta_fn = best_params[1]
    final_model.gamma_fp = best_params[2]
    final_model.fit(X_full, y_full)
    logger.warning(f"Final model trained with {best_model_name} using features {best_variant} and params {best_params} has thresholds {final_model.thresholds}.")
    pred_label = final_model.predict(X_full)
    y_train = [p['label'] for p in patients]

    logger.info(f'mean difference: {np.mean(np.array(y_train)-np.array(pred_label))}')

    logger.info(f'f1: {calculate_f1(y_train, pred_label)}')
    logger.info(f'balanced acc: {calculate_balanced_accuracy(y_train, pred_label)}')
    logger.info( f'custom loss: {custom_loss(y_train, pred_label, *best_params)}')


    for patient, pred in zip(patients, pred_label):
        volumes = patient['volume']
        patient_id = patient['patient_id']
        labels = patient['label']
        plt.figure(figsize=(15, 5))
        plt.plot(np.arange(1, len(volumes)+1), volumes,  'b.', label=f'Patient {patient_id}',)
        plt.title(f'Patient {patient_id} - Replanning fraction: {labels}')
        plt.xlabel('Fraction')
        plt.ylabel('Volume difference to pCT [%]')
        
        plt.axvline(x=labels, color='g', linestyle='-', label='Replanning Day')
        plt.axvline(x=pred, color='r', linestyle='--', label='Estimated Replanning Day')
        plt.xticks(np.arange(0, len(volumes)+1, 1))
        plt.ylim(-14, 7)

        plt.grid()
        plt.legend()
        plt.savefig(os.path.join(save_folder, f'patient_{patient_id}_submit.png'))

    logger.info(f'Plots saved in {save_folder}')


if __name__ == '__main__':
    np.random.seed(42)
    # Configure logging
    logger = logging.getLogger(__name__)
    output_path = '/storage/homefs/tf24s166/code/Volumetric_ART/output'
    logger = configure_logging(logger, output_path)

    # Run the main function
    main()