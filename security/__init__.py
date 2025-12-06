"""
MLSecOps Security Module for Wine Quality MLOps Project

This module provides comprehensive security testing and validation for ML pipelines:

- Lecke 113: Data and Model Poisoning Detection
- Lecke 114: Vertex AI Explainable AI and Model Monitoring
- Lecke 115-116: Model Robustness Testing (CleverHans, ART)
- Lecke 117-118: Dependency and Container Security (Cloud Build, Container Analysis)
- Lecke 119-120: Open-source Security Tools (Dependabot, Snyk)
- Lecke 121-122: External Model Auditing (Hugging Face validation)
"""

from .poisoning.data_poisoning_detector import DataPoisoningDetector
from .poisoning.model_poisoning_detector import ModelPoisoningDetector
from .robustness.adversarial_tester import AdversarialTester
from .model_audit.external_model_validator import ExternalModelValidator

__version__ = "1.0.0"
__all__ = [
    "DataPoisoningDetector",
    "ModelPoisoningDetector",
    "AdversarialTester",
    "ExternalModelValidator",
]
