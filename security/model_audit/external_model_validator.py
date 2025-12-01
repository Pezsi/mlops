"""
External Model Auditing and Validation Module (Lecke 121-122)

This module provides validation and auditing for external models:
- Hugging Face model validation
- Model provenance verification
- Security scanning for model files
- License compliance checking
- Model card validation

Reference: OWASP ML Security Top 10 - ML07:2023 Transfer Learning Attack
"""

import os
import json
import hashlib
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime
import subprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional imports
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from huggingface_hub import HfApi, hf_hub_download, model_info
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False
    logger.warning("huggingface_hub not available. Install with: pip install huggingface_hub")


class ExternalModelValidator:
    """
    Validator for external ML models from Hugging Face and other sources.

    Validates:
    1. Model authenticity and provenance
    2. Security of model files (pickle safety)
    3. License compliance
    4. Model card completeness
    5. Known vulnerabilities
    """

    # Trusted organizations on Hugging Face
    TRUSTED_ORGS = [
        "huggingface", "google", "facebook", "microsoft", "openai",
        "meta-llama", "mistralai", "EleutherAI", "bigscience",
        "stabilityai", "sentence-transformers", "allenai"
    ]

    # Acceptable licenses for commercial use
    COMMERCIAL_LICENSES = [
        "apache-2.0", "mit", "bsd-3-clause", "bsd-2-clause",
        "cc-by-4.0", "cc-by-sa-4.0", "openrail", "openrail++"
    ]

    # Required model card fields
    REQUIRED_CARD_FIELDS = [
        "model-index", "license", "language", "tags"
    ]

    def __init__(
        self,
        hf_token: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize External Model Validator.

        Args:
            hf_token: Hugging Face API token for private models
            cache_dir: Directory for caching downloaded models
        """
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "model_audit"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if HUGGINGFACE_AVAILABLE:
            self.hf_api = HfApi(token=self.hf_token)
        else:
            self.hf_api = None

    def validate_huggingface_model(
        self,
        model_id: str,
        check_security: bool = True,
        check_license: bool = True,
        check_card: bool = True
    ) -> Dict[str, Any]:
        """
        Validate a Hugging Face model.

        Args:
            model_id: Hugging Face model ID (e.g., "bert-base-uncased")
            check_security: Whether to perform security checks
            check_license: Whether to check license compliance
            check_card: Whether to validate model card

        Returns:
            Validation report
        """
        if not HUGGINGFACE_AVAILABLE:
            return {"error": "huggingface_hub not installed"}

        report = {
            "timestamp": datetime.now().isoformat(),
            "model_id": model_id,
            "validation_results": {},
            "warnings": [],
            "risks": [],
            "overall_status": "unknown"
        }

        try:
            # Get model info
            info = model_info(model_id, token=self.hf_token)
            report["model_info"] = {
                "author": info.author,
                "sha": info.sha,
                "created_at": str(info.created_at) if info.created_at else None,
                "last_modified": str(info.last_modified) if info.last_modified else None,
                "downloads": info.downloads,
                "likes": info.likes,
                "tags": info.tags,
                "pipeline_tag": info.pipeline_tag,
                "library_name": info.library_name
            }

            # Provenance check
            provenance = self._check_provenance(model_id, info)
            report["validation_results"]["provenance"] = provenance
            if not provenance["is_trusted"]:
                report["warnings"].append(
                    f"Model is from untrusted source: {info.author}"
                )

            # License check
            if check_license:
                license_check = self._check_license(info)
                report["validation_results"]["license"] = license_check
                if not license_check["is_compliant"]:
                    report["risks"].append(
                        f"License '{license_check['license']}' may not be suitable for commercial use"
                    )

            # Model card check
            if check_card:
                card_check = self._check_model_card(model_id, info)
                report["validation_results"]["model_card"] = card_check
                if not card_check["is_complete"]:
                    report["warnings"].append(
                        f"Model card is incomplete. Missing: {card_check['missing_fields']}"
                    )

            # Security check
            if check_security:
                security_check = self._check_model_security(model_id, info)
                report["validation_results"]["security"] = security_check
                if security_check["has_risks"]:
                    report["risks"].extend(security_check["risks"])

            # Determine overall status
            if report["risks"]:
                report["overall_status"] = "RISKY"
            elif report["warnings"]:
                report["overall_status"] = "WARNING"
            else:
                report["overall_status"] = "PASS"

        except Exception as e:
            logger.error(f"Validation error for {model_id}: {e}")
            report["error"] = str(e)
            report["overall_status"] = "ERROR"

        return report

    def _check_provenance(
        self,
        model_id: str,
        info: Any
    ) -> Dict[str, Any]:
        """Check model provenance and authenticity."""
        author = info.author if hasattr(info, 'author') else model_id.split("/")[0]

        result = {
            "author": author,
            "is_trusted": author.lower() in [org.lower() for org in self.TRUSTED_ORGS],
            "has_verified_badge": False,  # Would need API check
            "model_hash": info.sha if hasattr(info, 'sha') else None,
            "created_at": str(info.created_at) if hasattr(info, 'created_at') and info.created_at else None
        }

        # Check download count as trust indicator
        if hasattr(info, 'downloads') and info.downloads:
            result["downloads"] = info.downloads
            result["popularity_verified"] = info.downloads > 1000

        return result

    def _check_license(self, info: Any) -> Dict[str, Any]:
        """Check license compliance."""
        license_id = None

        if hasattr(info, 'card_data') and info.card_data:
            license_id = info.card_data.get('license')

        if not license_id and hasattr(info, 'tags'):
            for tag in info.tags or []:
                if tag.startswith('license:'):
                    license_id = tag.replace('license:', '')
                    break

        result = {
            "license": license_id or "unknown",
            "is_compliant": False,
            "commercial_use_allowed": False
        }

        if license_id:
            license_lower = license_id.lower()
            result["is_compliant"] = any(
                lic in license_lower for lic in self.COMMERCIAL_LICENSES
            )
            result["commercial_use_allowed"] = result["is_compliant"]

        return result

    def _check_model_card(
        self,
        model_id: str,
        info: Any
    ) -> Dict[str, Any]:
        """Check model card completeness."""
        result = {
            "has_card": False,
            "is_complete": False,
            "missing_fields": [],
            "card_fields": []
        }

        if hasattr(info, 'card_data') and info.card_data:
            result["has_card"] = True
            result["card_fields"] = list(info.card_data.keys())

            # Check for required fields
            for field in self.REQUIRED_CARD_FIELDS:
                if field not in info.card_data:
                    result["missing_fields"].append(field)

            result["is_complete"] = len(result["missing_fields"]) == 0

        return result

    def _check_model_security(
        self,
        model_id: str,
        info: Any
    ) -> Dict[str, Any]:
        """Check model files for security issues."""
        result = {
            "has_risks": False,
            "risks": [],
            "file_analysis": [],
            "pickle_files": [],
            "safetensors_available": False
        }

        # Check for safetensors (safer format)
        if hasattr(info, 'siblings'):
            for sibling in info.siblings or []:
                filename = sibling.rfilename if hasattr(sibling, 'rfilename') else str(sibling)

                # Check for safetensors
                if '.safetensors' in filename:
                    result["safetensors_available"] = True

                # Flag pickle files
                if any(ext in filename for ext in ['.pkl', '.pickle', '.bin', '.pt', '.pth']):
                    result["pickle_files"].append(filename)
                    result["file_analysis"].append({
                        "file": filename,
                        "type": "binary/pickle",
                        "risk": "medium",
                        "note": "Binary files can contain arbitrary code"
                    })

        # Risk assessment
        if result["pickle_files"] and not result["safetensors_available"]:
            result["risks"].append(
                "Model uses pickle format without safetensors alternative. "
                "Pickle files can execute arbitrary code during loading."
            )
            result["has_risks"] = True

        if not result["safetensors_available"]:
            result["risks"].append(
                "Consider requesting safetensors format for safer model loading."
            )

        return result

    def scan_pickle_file(
        self,
        file_path: str
    ) -> Dict[str, Any]:
        """
        Scan a pickle file for potentially malicious content.

        Args:
            file_path: Path to pickle file

        Returns:
            Security scan report
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "file_path": file_path,
            "file_hash": None,
            "is_safe": True,
            "warnings": [],
            "dangerous_operations": []
        }

        try:
            # Compute file hash
            with open(file_path, 'rb') as f:
                file_content = f.read()
                report["file_hash"] = hashlib.sha256(file_content).hexdigest()

            # Try to use fickling for deep analysis
            try:
                result = subprocess.run(
                    ["python", "-m", "fickling", file_path],
                    capture_output=True,
                    text=True,
                    timeout=30
                )

                if "UNSAFE" in result.stdout or "UNSAFE" in result.stderr:
                    report["is_safe"] = False
                    report["warnings"].append("Fickling detected unsafe operations")
                    report["dangerous_operations"].append(result.stdout)

            except (subprocess.TimeoutExpired, FileNotFoundError):
                # Fallback to basic analysis
                report["warnings"].append("Could not perform deep pickle analysis")

            # Basic pickle analysis
            dangerous_modules = [
                'os', 'subprocess', 'sys', 'builtins', '__builtin__',
                'socket', 'requests', 'urllib', 'shutil', 'commands'
            ]

            # Check for dangerous patterns in raw bytes
            content_str = str(file_content)
            for module in dangerous_modules:
                if module.encode() in file_content:
                    report["warnings"].append(
                        f"File contains reference to potentially dangerous module: {module}"
                    )
                    report["is_safe"] = False

            # Check for common exploit patterns
            exploit_patterns = [
                b"reduce", b"__reduce__", b"os.system", b"subprocess.call",
                b"exec(", b"eval(", b"compile("
            ]

            for pattern in exploit_patterns:
                if pattern in file_content:
                    report["dangerous_operations"].append(pattern.decode())
                    report["is_safe"] = False

        except Exception as e:
            logger.error(f"Pickle scan error: {e}")
            report["error"] = str(e)
            report["is_safe"] = False

        return report

    def verify_model_signature(
        self,
        model_path: str,
        expected_hash: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Verify model file integrity.

        Args:
            model_path: Path to model file
            expected_hash: Expected SHA256 hash

        Returns:
            Verification report
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "model_path": model_path,
            "verified": False,
            "computed_hash": None,
            "expected_hash": expected_hash
        }

        try:
            with open(model_path, 'rb') as f:
                computed_hash = hashlib.sha256(f.read()).hexdigest()

            report["computed_hash"] = computed_hash

            if expected_hash:
                report["verified"] = computed_hash == expected_hash
                if not report["verified"]:
                    report["error"] = "Hash mismatch - file may be corrupted or tampered"
            else:
                report["warning"] = "No expected hash provided for verification"

        except Exception as e:
            logger.error(f"Verification error: {e}")
            report["error"] = str(e)

        return report

    def audit_local_model(
        self,
        model_path: str,
        metadata_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Audit a locally stored model.

        Args:
            model_path: Path to model file or directory
            metadata_path: Path to model metadata file

        Returns:
            Audit report
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "model_path": model_path,
            "audit_results": {},
            "risks": [],
            "recommendations": []
        }

        model_path = Path(model_path)

        if model_path.is_file():
            files_to_check = [model_path]
        elif model_path.is_dir():
            files_to_check = list(model_path.glob("**/*"))
        else:
            report["error"] = f"Path does not exist: {model_path}"
            return report

        # Analyze each file
        for file_path in files_to_check:
            if not file_path.is_file():
                continue

            file_info = {
                "path": str(file_path),
                "size_bytes": file_path.stat().st_size,
                "extension": file_path.suffix
            }

            # Check for pickle files
            if file_path.suffix in ['.pkl', '.pickle', '.joblib', '.pt', '.pth', '.bin']:
                scan_result = self.scan_pickle_file(str(file_path))
                file_info["pickle_scan"] = scan_result

                if not scan_result["is_safe"]:
                    report["risks"].append(
                        f"Unsafe pickle file detected: {file_path.name}"
                    )

            # Check for safetensors
            if file_path.suffix == '.safetensors':
                file_info["format"] = "safetensors"
                file_info["is_safe_format"] = True

            report["audit_results"][str(file_path.name)] = file_info

        # Check metadata if provided
        if metadata_path:
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    report["metadata"] = metadata
            except Exception as e:
                report["metadata_error"] = str(e)

        # Generate recommendations
        has_safetensors = any(
            f.get("is_safe_format")
            for f in report["audit_results"].values()
        )

        if not has_safetensors:
            report["recommendations"].append(
                "Consider converting model to safetensors format for safer loading"
            )

        if report["risks"]:
            report["recommendations"].append(
                "Review and validate all pickle files before loading"
            )
            report["recommendations"].append(
                "Consider using torch.load with weights_only=True"
            )

        return report

    def generate_model_attestation(
        self,
        model_path: str,
        model_id: str,
        validation_report: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate a model attestation document for auditing purposes.

        Args:
            model_path: Path to model
            model_id: Model identifier
            validation_report: Previous validation report

        Returns:
            Attestation document
        """
        attestation = {
            "attestation_version": "1.0",
            "timestamp": datetime.now().isoformat(),
            "model_id": model_id,
            "model_path": model_path,
            "attestation_type": "security_audit",
            "auditor": "MLSecOps Automated Validator",
            "validation_summary": {
                "overall_status": validation_report.get("overall_status", "unknown"),
                "risks_found": len(validation_report.get("risks", [])),
                "warnings_found": len(validation_report.get("warnings", []))
            },
            "checks_performed": list(validation_report.get("validation_results", {}).keys()),
            "compliance": {
                "license_verified": validation_report.get("validation_results", {})
                    .get("license", {}).get("is_compliant", False),
                "provenance_verified": validation_report.get("validation_results", {})
                    .get("provenance", {}).get("is_trusted", False),
                "security_verified": not validation_report.get("validation_results", {})
                    .get("security", {}).get("has_risks", True)
            }
        }

        # Compute attestation hash
        attestation_str = json.dumps(attestation, sort_keys=True)
        attestation["attestation_hash"] = hashlib.sha256(
            attestation_str.encode()
        ).hexdigest()

        return attestation


def run_huggingface_audit_demo():
    """Demo function for Hugging Face model auditing."""
    print("=== External Model Validation Demo ===\n")

    if not HUGGINGFACE_AVAILABLE:
        print("huggingface_hub not installed. Install with: pip install huggingface_hub")
        return

    validator = ExternalModelValidator()

    # Example models to validate
    test_models = [
        "bert-base-uncased",
        "distilbert-base-uncased",
        "facebook/bart-base"
    ]

    for model_id in test_models:
        print(f"\n{'='*50}")
        print(f"Validating: {model_id}")
        print('='*50)

        report = validator.validate_huggingface_model(model_id)

        print(f"\nStatus: {report['overall_status']}")

        if "model_info" in report:
            info = report["model_info"]
            print(f"Author: {info.get('author', 'unknown')}")
            print(f"Downloads: {info.get('downloads', 'N/A')}")
            print(f"Pipeline: {info.get('pipeline_tag', 'N/A')}")

        if "validation_results" in report:
            results = report["validation_results"]

            if "provenance" in results:
                print(f"\nProvenance:")
                print(f"  Trusted: {results['provenance']['is_trusted']}")

            if "license" in results:
                print(f"\nLicense:")
                print(f"  License: {results['license']['license']}")
                print(f"  Commercial OK: {results['license']['commercial_use_allowed']}")

            if "security" in results:
                print(f"\nSecurity:")
                print(f"  Has Risks: {results['security']['has_risks']}")
                print(f"  Safetensors: {results['security']['safetensors_available']}")

        if report.get("warnings"):
            print(f"\nWarnings:")
            for w in report["warnings"]:
                print(f"  - {w}")

        if report.get("risks"):
            print(f"\nRisks:")
            for r in report["risks"]:
                print(f"  - {r}")


if __name__ == "__main__":
    run_huggingface_audit_demo()
