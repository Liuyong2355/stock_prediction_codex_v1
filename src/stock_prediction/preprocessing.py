"""Fold-local exact medians and train-only StandardScaler, bounded-memory fit."""
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


class RidgePreprocessor:
    def __init__(self, feature_names, batch_size=100_000):
        self.feature_names = list(feature_names)
        self.batch_size = batch_size

    def fit(self, matrix, train_indices):
        indices = np.asarray(train_indices)
        if not len(indices):
            raise ValueError("No training rows for preprocessing")
        self.kept_indices_, self.dropped_features_, self.imputers_ = [], [], []
        for j, name in enumerate(self.feature_names):
            column = np.asarray(matrix[indices, j]).reshape(-1, 1)
            if np.isinf(column).any():
                raise ValueError("Basic40 inf violates feature contract")
            if np.isnan(column).all():
                self.dropped_features_.append(name)
                continue
            self.kept_indices_.append(j)
            self.imputers_.append(SimpleImputer(strategy="median").fit(column))
        if not self.kept_indices_:
            raise ValueError("All training features are NaN")
        self.scaler_ = StandardScaler()
        # Partial fitting computes the same population moments as batch fitting,
        # up to roundoff; every batch is drawn exclusively from train_indices.
        for start in range(0, len(indices), self.batch_size):
            batch = matrix[indices[start:start + self.batch_size]]
            self.scaler_.partial_fit(self.impute(batch))
        self.training_rows_ = len(indices)
        return self

    def impute(self, matrix):
        output = np.empty((len(matrix), len(self.kept_indices_)), dtype="float64")
        for k, (j, imputer) in enumerate(zip(self.kept_indices_, self.imputers_)):
            output[:, k] = imputer.transform(np.asarray(matrix[:, j]).reshape(-1, 1))[:, 0]
        return output

    def transform(self, matrix):
        return self.scaler_.transform(self.impute(matrix))

    def metadata(self):
        return {"fit_rows": self.training_rows_, "kept_features": [self.feature_names[j] for j in self.kept_indices_],
                "dropped_all_nan_features": self.dropped_features_,
                "median": [float(i.statistics_[0]) for i in self.imputers_],
                "scaler_mean": self.scaler_.mean_.tolist(), "scaler_var": self.scaler_.var_.tolist(),
                "scaler_scale": self.scaler_.scale_.tolist(), "scaler_fit_rows": int(self.scaler_.n_samples_seen_),
                "fit_scope": "Only finite-label training rows after one-date purge; validation never passed to fit."}
