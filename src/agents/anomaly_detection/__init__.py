"""Agent 2 - Anomaly Detection (CyberSentinel).

Independent, benign-only novelty detector: anomaly score, flag, feature
evidence and rationale. Does not predict attack classes or make decisions.
"""
from .api import AnomalyDetectionAgent, detect_anomalies, detect_anomaly, load_detector

__all__ = ["AnomalyDetectionAgent", "detect_anomaly", "detect_anomalies", "load_detector"]
