"""
Dependency Security Scanning Module (Lecke 117-120)

This module provides dependency and container security scanning:
- Python dependency vulnerability scanning
- License compliance checking
- Container image analysis
- Integration with Snyk and safety

Reference: OWASP ML Security Top 10 - ML10:2023 Model/Data Theft
"""

import subprocess
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DependencyScanner:
    """
    Dependency security scanner for Python projects.

    Scans for:
    1. Known vulnerabilities (CVEs)
    2. Outdated packages
    3. License compliance
    4. Malicious packages
    """

    def __init__(
        self,
        project_path: str = ".",
        requirements_file: str = "requirements.txt"
    ):
        """
        Initialize Dependency Scanner.

        Args:
            project_path: Path to project root
            requirements_file: Name of requirements file
        """
        self.project_path = Path(project_path)
        self.requirements_file = requirements_file
        self.vulnerabilities: List[Dict] = []

    def scan_with_safety(self) -> Dict[str, Any]:
        """
        Scan dependencies using safety (PyUp.io).

        Returns:
            Safety scan report
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "scanner": "safety",
            "vulnerabilities": [],
            "total_packages": 0,
            "vulnerable_packages": 0
        }

        try:
            # Try to run safety scan
            result = subprocess.run(
                ["safety", "check", "--json",
                 "-r", str(self.project_path / self.requirements_file)],
                capture_output=True,
                text=True,
                cwd=str(self.project_path)
            )

            if result.returncode == 0:
                logger.info("No vulnerabilities found by safety")
                report["status"] = "clean"
            else:
                try:
                    # Parse JSON output
                    vuln_data = json.loads(result.stdout)

                    for vuln in vuln_data:
                        if isinstance(vuln, list) and len(vuln) >= 5:
                            report["vulnerabilities"].append({
                                "package": vuln[0],
                                "installed_version": vuln[1],
                                "affected_versions": vuln[2],
                                "vulnerability_id": vuln[3],
                                "description": vuln[4]
                            })

                    report["vulnerable_packages"] = len(report["vulnerabilities"])
                    report["status"] = "vulnerabilities_found"
                except json.JSONDecodeError:
                    # Non-JSON output
                    report["raw_output"] = result.stdout
                    report["status"] = "parse_error"

        except FileNotFoundError:
            logger.warning("safety not installed. Install with: pip install safety")
            report["status"] = "scanner_not_available"
            report["error"] = "safety not installed"

        except Exception as e:
            logger.error(f"Safety scan error: {e}")
            report["status"] = "error"
            report["error"] = str(e)

        self.vulnerabilities.extend(report.get("vulnerabilities", []))
        return report

    def scan_with_pip_audit(self) -> Dict[str, Any]:
        """
        Scan dependencies using pip-audit.

        Returns:
            pip-audit scan report
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "scanner": "pip-audit",
            "vulnerabilities": [],
            "total_packages": 0,
            "vulnerable_packages": 0
        }

        try:
            result = subprocess.run(
                ["pip-audit", "--format", "json",
                 "-r", str(self.project_path / self.requirements_file)],
                capture_output=True,
                text=True,
                cwd=str(self.project_path)
            )

            try:
                vuln_data = json.loads(result.stdout)

                for item in vuln_data:
                    if "vulns" in item:
                        for vuln in item["vulns"]:
                            report["vulnerabilities"].append({
                                "package": item["name"],
                                "installed_version": item.get("version", "unknown"),
                                "vulnerability_id": vuln.get("id", "unknown"),
                                "description": vuln.get("description", ""),
                                "fixed_versions": vuln.get("fix_versions", []),
                                "aliases": vuln.get("aliases", [])
                            })

                report["vulnerable_packages"] = len(set(
                    v["package"] for v in report["vulnerabilities"]
                ))
                report["status"] = "vulnerabilities_found" if report["vulnerabilities"] else "clean"

            except json.JSONDecodeError:
                if "No known vulnerabilities" in result.stdout:
                    report["status"] = "clean"
                else:
                    report["raw_output"] = result.stdout
                    report["status"] = "parse_error"

        except FileNotFoundError:
            logger.warning("pip-audit not installed. Install with: pip install pip-audit")
            report["status"] = "scanner_not_available"
            report["error"] = "pip-audit not installed"

        except Exception as e:
            logger.error(f"pip-audit scan error: {e}")
            report["status"] = "error"
            report["error"] = str(e)

        self.vulnerabilities.extend(report.get("vulnerabilities", []))
        return report

    def check_for_typosquatting(self) -> Dict[str, Any]:
        """
        Check for potential typosquatting attacks in dependencies.

        Common legitimate packages and their typosquatting variants.

        Returns:
            Typosquatting check report
        """
        known_packages = {
            "numpy": ["numpi", "numppy", "nunpy"],
            "pandas": ["panda", "pandass", "pandsa"],
            "scikit-learn": ["scikitlearn", "sklearn", "scikit_learn"],
            "tensorflow": ["tensorfow", "tensorlow", "tenserflow"],
            "pytorch": ["pytorh", "pytoch", "pytourch"],
            "requests": ["reqeusts", "requets", "request"],
            "flask": ["flaask", "falsk"],
            "django": ["djano", "djang", "djnago"],
            "pillow": ["pil", "pillw", "pilow"],
            "cryptography": ["crytography", "cryptogrphy", "cyptography"]
        }

        report = {
            "timestamp": datetime.now().isoformat(),
            "check_type": "typosquatting",
            "suspicious_packages": [],
            "status": "clean"
        }

        try:
            with open(self.project_path / self.requirements_file, 'r') as f:
                requirements = f.readlines()

            installed_packages = []
            for req in requirements:
                # Parse package name
                match = re.match(r'^([a-zA-Z0-9_-]+)', req.strip())
                if match:
                    installed_packages.append(match.group(1).lower())

            # Check for typosquatting
            for pkg in installed_packages:
                for legitimate, variants in known_packages.items():
                    if pkg in variants:
                        report["suspicious_packages"].append({
                            "package": pkg,
                            "might_be": legitimate,
                            "reason": "possible_typosquatting"
                        })

            if report["suspicious_packages"]:
                report["status"] = "suspicious_found"
                logger.warning(f"Found {len(report['suspicious_packages'])} "
                             f"suspicious packages")

        except Exception as e:
            logger.error(f"Typosquatting check error: {e}")
            report["status"] = "error"
            report["error"] = str(e)

        return report

    def check_license_compliance(
        self,
        allowed_licenses: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Check license compliance for dependencies.

        Args:
            allowed_licenses: List of allowed license types

        Returns:
            License compliance report
        """
        if allowed_licenses is None:
            allowed_licenses = [
                "MIT", "Apache-2.0", "BSD-3-Clause", "BSD-2-Clause",
                "ISC", "PSF", "LGPL", "Apache License 2.0",
                "MIT License", "BSD License"
            ]

        report = {
            "timestamp": datetime.now().isoformat(),
            "check_type": "license_compliance",
            "allowed_licenses": allowed_licenses,
            "packages": [],
            "non_compliant": [],
            "status": "compliant"
        }

        try:
            result = subprocess.run(
                ["pip-licenses", "--format=json"],
                capture_output=True,
                text=True,
                cwd=str(self.project_path)
            )

            if result.returncode == 0:
                packages = json.loads(result.stdout)

                for pkg in packages:
                    license_type = pkg.get("License", "Unknown")
                    pkg_info = {
                        "name": pkg.get("Name", "unknown"),
                        "version": pkg.get("Version", "unknown"),
                        "license": license_type
                    }
                    report["packages"].append(pkg_info)

                    # Check compliance
                    is_compliant = any(
                        allowed.lower() in license_type.lower()
                        for allowed in allowed_licenses
                    )

                    if not is_compliant and license_type != "Unknown":
                        report["non_compliant"].append(pkg_info)

                if report["non_compliant"]:
                    report["status"] = "non_compliant_found"

        except FileNotFoundError:
            logger.warning("pip-licenses not installed. Install with: pip install pip-licenses")
            report["status"] = "tool_not_available"

        except Exception as e:
            logger.error(f"License check error: {e}")
            report["status"] = "error"
            report["error"] = str(e)

        return report

    def check_outdated_packages(self) -> Dict[str, Any]:
        """
        Check for outdated packages.

        Returns:
            Outdated packages report
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "check_type": "outdated_packages",
            "outdated": [],
            "total_packages": 0,
            "outdated_count": 0,
            "status": "up_to_date"
        }

        try:
            result = subprocess.run(
                ["pip", "list", "--outdated", "--format=json"],
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                outdated = json.loads(result.stdout)
                report["outdated"] = [
                    {
                        "name": pkg["name"],
                        "current_version": pkg["version"],
                        "latest_version": pkg["latest_version"],
                        "latest_filetype": pkg.get("latest_filetype", "unknown")
                    }
                    for pkg in outdated
                ]
                report["outdated_count"] = len(report["outdated"])

                if report["outdated_count"] > 0:
                    report["status"] = "outdated_found"

            # Get total package count
            result_all = subprocess.run(
                ["pip", "list", "--format=json"],
                capture_output=True,
                text=True
            )
            if result_all.returncode == 0:
                all_packages = json.loads(result_all.stdout)
                report["total_packages"] = len(all_packages)

        except Exception as e:
            logger.error(f"Outdated check error: {e}")
            report["status"] = "error"
            report["error"] = str(e)

        return report

    def generate_sbom(self, output_format: str = "cyclonedx") -> Dict[str, Any]:
        """
        Generate Software Bill of Materials (SBOM).

        Args:
            output_format: SBOM format (cyclonedx or spdx)

        Returns:
            SBOM generation report
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "format": output_format,
            "status": "pending"
        }

        try:
            output_file = self.project_path / f"sbom.{output_format}.json"

            if output_format == "cyclonedx":
                result = subprocess.run(
                    ["cyclonedx-py", "requirements",
                     "-r", str(self.project_path / self.requirements_file),
                     "-o", str(output_file)],
                    capture_output=True,
                    text=True
                )
            else:  # spdx
                result = subprocess.run(
                    ["spdx-sbom-generator", "-p", str(self.project_path)],
                    capture_output=True,
                    text=True
                )

            if result.returncode == 0:
                report["status"] = "success"
                report["output_file"] = str(output_file)
                logger.info(f"SBOM generated: {output_file}")
            else:
                report["status"] = "error"
                report["error"] = result.stderr

        except FileNotFoundError:
            logger.warning(f"SBOM generator not installed for format: {output_format}")
            report["status"] = "tool_not_available"
            report["install_hint"] = (
                "pip install cyclonedx-bom" if output_format == "cyclonedx"
                else "Install spdx-sbom-generator"
            )

        except Exception as e:
            logger.error(f"SBOM generation error: {e}")
            report["status"] = "error"
            report["error"] = str(e)

        return report

    def run_full_scan(self) -> Dict[str, Any]:
        """
        Run comprehensive dependency scan.

        Returns:
            Full scan report
        """
        logger.info("Starting comprehensive dependency scan...")

        report = {
            "timestamp": datetime.now().isoformat(),
            "project_path": str(self.project_path),
            "requirements_file": self.requirements_file,
            "scans": {}
        }

        # Safety scan
        report["scans"]["safety"] = self.scan_with_safety()

        # pip-audit scan
        report["scans"]["pip_audit"] = self.scan_with_pip_audit()

        # Typosquatting check
        report["scans"]["typosquatting"] = self.check_for_typosquatting()

        # License compliance
        report["scans"]["license_compliance"] = self.check_license_compliance()

        # Outdated packages
        report["scans"]["outdated"] = self.check_outdated_packages()

        # Summary
        total_vulns = len(self.vulnerabilities)
        unique_packages = len(set(v.get("package", "") for v in self.vulnerabilities))

        report["summary"] = {
            "total_vulnerabilities": total_vulns,
            "vulnerable_packages": unique_packages,
            "typosquatting_suspects": len(
                report["scans"]["typosquatting"].get("suspicious_packages", [])
            ),
            "non_compliant_licenses": len(
                report["scans"]["license_compliance"].get("non_compliant", [])
            ),
            "outdated_packages": report["scans"]["outdated"].get("outdated_count", 0),
            "overall_risk": self._calculate_risk_level(report)
        }

        logger.info(f"Scan complete. Risk level: {report['summary']['overall_risk']}")
        return report

    def _calculate_risk_level(self, report: Dict) -> str:
        """Calculate overall risk level."""
        vulns = len(self.vulnerabilities)
        typo = len(report["scans"]["typosquatting"].get("suspicious_packages", []))
        license_issues = len(
            report["scans"]["license_compliance"].get("non_compliant", [])
        )

        if vulns > 5 or typo > 0:
            return "CRITICAL"
        elif vulns > 0 or license_issues > 3:
            return "HIGH"
        elif license_issues > 0:
            return "MEDIUM"
        else:
            return "LOW"


class ContainerScanner:
    """
    Container image security scanner.

    Integrates with:
    - Trivy
    - GCP Container Analysis
    - Snyk Container
    """

    def __init__(self, image_name: str):
        """
        Initialize Container Scanner.

        Args:
            image_name: Docker image name/tag
        """
        self.image_name = image_name

    def scan_with_trivy(self) -> Dict[str, Any]:
        """
        Scan container with Trivy.

        Returns:
            Trivy scan report
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "scanner": "trivy",
            "image": self.image_name,
            "vulnerabilities": [],
            "status": "pending"
        }

        try:
            result = subprocess.run(
                ["trivy", "image", "--format", "json", self.image_name],
                capture_output=True,
                text=True
            )

            if result.returncode == 0 or result.stdout:
                try:
                    scan_data = json.loads(result.stdout)

                    for result_item in scan_data.get("Results", []):
                        for vuln in result_item.get("Vulnerabilities", []):
                            report["vulnerabilities"].append({
                                "vulnerability_id": vuln.get("VulnerabilityID"),
                                "package": vuln.get("PkgName"),
                                "installed_version": vuln.get("InstalledVersion"),
                                "fixed_version": vuln.get("FixedVersion"),
                                "severity": vuln.get("Severity"),
                                "title": vuln.get("Title"),
                                "description": vuln.get("Description", "")[:200]
                            })

                    # Count by severity
                    severity_counts = {}
                    for vuln in report["vulnerabilities"]:
                        sev = vuln.get("severity", "UNKNOWN")
                        severity_counts[sev] = severity_counts.get(sev, 0) + 1

                    report["severity_counts"] = severity_counts
                    report["status"] = "success"

                except json.JSONDecodeError:
                    report["raw_output"] = result.stdout
                    report["status"] = "parse_error"

        except FileNotFoundError:
            logger.warning("Trivy not installed. Install from: https://trivy.dev")
            report["status"] = "scanner_not_available"

        except Exception as e:
            logger.error(f"Trivy scan error: {e}")
            report["status"] = "error"
            report["error"] = str(e)

        return report

    def scan_with_grype(self) -> Dict[str, Any]:
        """
        Scan container with Grype (Anchore).

        Returns:
            Grype scan report
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "scanner": "grype",
            "image": self.image_name,
            "vulnerabilities": [],
            "status": "pending"
        }

        try:
            result = subprocess.run(
                ["grype", self.image_name, "-o", "json"],
                capture_output=True,
                text=True
            )

            if result.stdout:
                scan_data = json.loads(result.stdout)

                for match in scan_data.get("matches", []):
                    vuln = match.get("vulnerability", {})
                    artifact = match.get("artifact", {})

                    report["vulnerabilities"].append({
                        "vulnerability_id": vuln.get("id"),
                        "package": artifact.get("name"),
                        "installed_version": artifact.get("version"),
                        "severity": vuln.get("severity"),
                        "fix_versions": vuln.get("fix", {}).get("versions", [])
                    })

                report["status"] = "success"

        except FileNotFoundError:
            logger.warning("Grype not installed")
            report["status"] = "scanner_not_available"

        except Exception as e:
            logger.error(f"Grype scan error: {e}")
            report["status"] = "error"
            report["error"] = str(e)

        return report

    def generate_gcp_container_analysis_config(
        self,
        project_id: str,
        artifact_registry: str
    ) -> Dict[str, Any]:
        """
        Generate GCP Container Analysis configuration.

        Args:
            project_id: GCP project ID
            artifact_registry: Artifact Registry repository

        Returns:
            Container Analysis configuration
        """
        config = {
            "container_analysis": {
                "project": project_id,
                "artifact_registry": artifact_registry,
                "vulnerability_scanning": {
                    "enabled": True,
                    "scanning_config": {
                        "scanning_mode": "STANDARD"
                    }
                },
                "attestations": {
                    "enabled": True,
                    "binary_authorization": {
                        "policy": "projects/{}/policy".format(project_id)
                    }
                }
            },
            "cloud_build_steps": [
                {
                    "name": "gcr.io/cloud-builders/docker",
                    "args": ["build", "-t", "${_IMAGE_NAME}", "."]
                },
                {
                    "name": "gcr.io/cloud-builders/docker",
                    "args": ["push", "${_IMAGE_NAME}"]
                },
                {
                    "name": "gcr.io/google.com/cloudsdktool/cloud-sdk",
                    "entrypoint": "bash",
                    "args": [
                        "-c",
                        "gcloud artifacts docker images describe ${_IMAGE_NAME} "
                        "--show-package-vulnerability"
                    ]
                }
            ],
            "substitutions": {
                "_IMAGE_NAME": f"{artifact_registry}/{self.image_name}"
            }
        }

        return config


if __name__ == "__main__":
    print("=== Dependency Security Scanner Demo ===\n")

    # Initialize scanner
    scanner = DependencyScanner(project_path=".")

    # Run full scan
    report = scanner.run_full_scan()

    print("\n=== Scan Summary ===")
    print(f"Total vulnerabilities: {report['summary']['total_vulnerabilities']}")
    print(f"Vulnerable packages: {report['summary']['vulnerable_packages']}")
    print(f"Typosquatting suspects: {report['summary']['typosquatting_suspects']}")
    print(f"License issues: {report['summary']['non_compliant_licenses']}")
    print(f"Outdated packages: {report['summary']['outdated_packages']}")
    print(f"Overall risk: {report['summary']['overall_risk']}")

    # Show vulnerabilities if any
    if scanner.vulnerabilities:
        print("\n=== Vulnerabilities Found ===")
        for vuln in scanner.vulnerabilities[:5]:
            print(f"  - {vuln.get('package', 'unknown')}: "
                  f"{vuln.get('vulnerability_id', 'N/A')}")
