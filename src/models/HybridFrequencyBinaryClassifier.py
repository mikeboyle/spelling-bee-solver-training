from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from .ClassifierProtocol import ClassifierProtocol


class HybridFrequencyBinaryClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, 
                 low_mid_threshold: float,
                 mid_high_threshold: float,
                 low_freq_model: ClassifierProtocol,
                 mid_freq_model: ClassifierProtocol,
                 high_freq_model: ClassifierProtocol):
        
        self.low_mid_threshold = low_mid_threshold
        self.mid_high_threshold = mid_high_threshold
        self.low_freq_model = low_freq_model
        self.mid_freq_model = mid_freq_model
        self.high_freq_model = high_freq_model
    
    def _split_high_low(self, freqs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        low_idx = freqs <= self.low_mid_threshold
        mid_idx = (freqs > self.low_mid_threshold) & (freqs < self.mid_high_threshold)
        high_idx = freqs >= self.mid_high_threshold
        
        return low_idx, mid_idx, high_idx
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "HybridFrequencyBinaryClassifier":
        freqs = X[:, 0] # assumes frequency is first feature
        low_idx, mid_idx, high_idx = self._split_high_low(freqs)

        if X[low_idx].shape[0] > 0:
            self.low_freq_model.fit(X[low_idx], y[low_idx])
        
        if X[mid_idx].shape[0] > 0:
            self.mid_freq_model.fit(X[mid_idx], y[mid_idx])
        
        if X[high_idx].shape[0] > 0:
            self.high_freq_model.fit(X[high_idx], y[high_idx])
             
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        freqs = X[:, 0] # assumes frequency is first feature
        preds = np.empty(X.shape[0], dtype=np.float64)
        
        low_idx, mid_idx, high_idx = self._split_high_low(freqs)

        if X[low_idx].shape[0] > 0:
            preds[low_idx] = self.low_freq_model.predict(X[low_idx])
        
        if X[mid_idx].shape[0] > 0:
            preds[mid_idx] = self.mid_freq_model.predict(X[mid_idx])

        if X[high_idx].shape[0] > 0:
            preds[high_idx] = self.high_freq_model.predict(X[high_idx])
        
        return preds

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        freqs = X[:, 0] # assumes frequency is first feature
        proba = np.zeros((X.shape[0], 2), dtype=np.float64)

        low_idx, mid_idx, high_idx = self._split_high_low(freqs)

        if X[low_idx].shape[0] > 0:
            proba[low_idx] = self.low_freq_model.predict_proba(X[low_idx])
        
        if X[mid_idx].shape[0] > 0:
            proba[mid_idx] = self.mid_freq_model.predict_proba(X[mid_idx])

        if X[high_idx].shape[0] > 0:
            proba[high_idx] = self.high_freq_model.predict_proba(X[high_idx])
        
        return proba
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return np.mean(self.predict(X) == y)
