"""
Comprehensive Notification System for MLOps Pipeline

Supports multiple channels:
- Email (SMTP)
- Slack (Webhooks)
- Logging to metadata database
"""

import smtplib
import requests
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional, Dict, List
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class NotificationConfig:
    """Configuration for notification channels."""

    def __init__(
        self,
        # Email settings
        smtp_host: str = "smtp.gmail.com",
        smtp_port: int = 587,
        smtp_user: Optional[str] = None,
        smtp_password: Optional[str] = None,
        from_email: Optional[str] = None,
        to_emails: Optional[List[str]] = None,
        # Slack settings
        slack_webhook_url: Optional[str] = None,
        slack_channel: Optional[str] = None,
        # General settings
        enable_email: bool = True,
        enable_slack: bool = True,
        enable_db_logging: bool = True
    ):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.smtp_user = smtp_user or from_email
        self.smtp_password = smtp_password
        self.from_email = from_email
        self.to_emails = to_emails or []

        self.slack_webhook_url = slack_webhook_url
        self.slack_channel = slack_channel

        self.enable_email = enable_email and smtp_user and from_email
        self.enable_slack = enable_slack and slack_webhook_url
        self.enable_db_logging = enable_db_logging


class NotificationSystem:
    """Unified notification system for ML pipeline events."""

    def __init__(self, config: NotificationConfig, metadata_tracker=None):
        """
        Initialize notification system.

        Args:
            config: NotificationConfig instance
            metadata_tracker: MetadataTracker instance for logging
        """
        self.config = config
        self.metadata_tracker = metadata_tracker

    def send_notification(
        self,
        subject: str,
        message: str,
        level: str = "info",
        dag_id: Optional[str] = None,
        task_id: Optional[str] = None,
        event_data: Optional[Dict] = None,
        channels: Optional[List[str]] = None
    ) -> Dict[str, bool]:
        """
        Send notification through configured channels.

        Args:
            subject: Notification subject/title
            message: Notification message body
            level: Notification level (info, warning, error, critical)
            dag_id: DAG identifier
            task_id: Task identifier
            event_data: Additional event data
            channels: List of channels to use (email, slack, db). None = all enabled.

        Returns:
            Dictionary with channel: success mapping
        """
        results = {}

        # Determine which channels to use
        use_email = (channels is None or 'email' in channels) and self.config.enable_email
        use_slack = (channels is None or 'slack' in channels) and self.config.enable_slack
        use_db = (channels is None or 'db' in channels) and self.config.enable_db_logging

        # Log to database first (if enabled)
        event_id = None
        if use_db and self.metadata_tracker:
            try:
                event_id = self.metadata_tracker.log_pipeline_event(
                    dag_id=dag_id or "unknown",
                    event_type=level,
                    event_message=f"{subject}: {message}",
                    task_id=task_id,
                    event_data=event_data
                )
                results['db'] = True
            except Exception as e:
                logger.error(f"Failed to log event to database: {e}")
                results['db'] = False

        # Send email notification
        if use_email:
            email_success = self._send_email(subject, message, level)
            results['email'] = email_success

            if self.metadata_tracker and event_id:
                try:
                    self.metadata_tracker.log_notification(
                        event_id=event_id,
                        channel='email',
                        recipient=', '.join(self.config.to_emails),
                        subject=subject,
                        message=message,
                        status='sent' if email_success else 'failed'
                    )
                except Exception as e:
                    logger.error(f"Failed to log email notification: {e}")

        # Send Slack notification
        if use_slack:
            slack_success = self._send_slack(subject, message, level, event_data)
            results['slack'] = slack_success

            if self.metadata_tracker and event_id:
                try:
                    self.metadata_tracker.log_notification(
                        event_id=event_id,
                        channel='slack',
                        recipient=self.config.slack_channel or 'default',
                        subject=subject,
                        message=message,
                        status='sent' if slack_success else 'failed'
                    )
                except Exception as e:
                    logger.error(f"Failed to log Slack notification: {e}")

        return results

    def _send_email(self, subject: str, message: str, level: str) -> bool:
        """Send email notification via SMTP."""
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"[{level.upper()}] {subject}"
            msg['From'] = self.config.from_email
            msg['To'] = ', '.join(self.config.to_emails)

            # Create HTML version with styling based on level
            color_map = {
                'info': '#17a2b8',
                'warning': '#ffc107',
                'error': '#dc3545',
                'critical': '#721c24'
            }
            color = color_map.get(level, '#6c757d')

            html_content = f"""
            <html>
                <body style="font-family: Arial, sans-serif;">
                    <div style="border-left: 4px solid {color}; padding: 20px; background-color: #f8f9fa;">
                        <h2 style="color: {color}; margin-top: 0;">
                            {level.upper()}: {subject}
                        </h2>
                        <div style="color: #333; line-height: 1.6;">
                            {message.replace(chr(10), '<br>')}
                        </div>
                        <hr style="border: none; border-top: 1px solid #dee2e6; margin: 20px 0;">
                        <p style="color: #6c757d; font-size: 12px;">
                            Sent by Wine Quality MLOps Pipeline<br>
                            Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                        </p>
                    </div>
                </body>
            </html>
            """

            # Attach HTML and plain text versions
            msg.attach(MIMEText(message, 'plain'))
            msg.attach(MIMEText(html_content, 'html'))

            # Connect to SMTP server and send
            with smtplib.SMTP(self.config.smtp_host, self.config.smtp_port) as server:
                server.starttls()
                server.login(self.config.smtp_user, self.config.smtp_password)
                server.send_message(msg)

            logger.info(f"Email sent successfully: {subject}")
            return True

        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False

    def _send_slack(
        self,
        subject: str,
        message: str,
        level: str,
        event_data: Optional[Dict] = None
    ) -> bool:
        """Send Slack notification via webhook."""
        try:
            # Map level to Slack color
            color_map = {
                'info': '#36a64f',
                'warning': '#ff9900',
                'error': '#ff0000',
                'critical': '#8b0000'
            }
            color = color_map.get(level, '#808080')

            # Map level to emoji
            emoji_map = {
                'info': ':information_source:',
                'warning': ':warning:',
                'error': ':x:',
                'critical': ':fire:'
            }
            emoji = emoji_map.get(level, ':bell:')

            # Build Slack message payload
            payload = {
                'username': 'MLOps Pipeline Bot',
                'icon_emoji': ':robot_face:',
                'attachments': [
                    {
                        'color': color,
                        'title': f"{emoji} {level.upper()}: {subject}",
                        'text': message,
                        'footer': 'Wine Quality MLOps Pipeline',
                        'ts': int(datetime.now().timestamp())
                    }
                ]
            }

            # Add event data as fields if provided
            if event_data:
                fields = []
                for key, value in event_data.items():
                    fields.append({
                        'title': key,
                        'value': str(value),
                        'short': True
                    })
                payload['attachments'][0]['fields'] = fields

            # Override channel if specified
            if self.config.slack_channel:
                payload['channel'] = self.config.slack_channel

            # Send webhook request
            response = requests.post(
                self.config.slack_webhook_url,
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=10
            )

            if response.status_code == 200:
                logger.info(f"Slack notification sent successfully: {subject}")
                return True
            else:
                logger.error(f"Slack API error: {response.status_code} - {response.text}")
                return False

        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
            return False

    def notify_model_training_start(self, model_name: str, run_id: str, dag_id: str):
        """Convenience method for model training start notification."""
        return self.send_notification(
            subject=f"Model Training Started: {model_name}",
            message=f"Training run {run_id} has started for model '{model_name}'.",
            level="info",
            dag_id=dag_id,
            event_data={'run_id': run_id, 'model_name': model_name}
        )

    def notify_model_training_success(
        self,
        model_name: str,
        run_id: str,
        metrics: Dict[str, float],
        dag_id: str
    ):
        """Convenience method for successful training notification."""
        metrics_str = '\n'.join([f"  - {k}: {v:.4f}" for k, v in metrics.items()])

        return self.send_notification(
            subject=f"Model Training Completed: {model_name}",
            message=f"Training run {run_id} completed successfully.\n\nMetrics:\n{metrics_str}",
            level="info",
            dag_id=dag_id,
            event_data={'run_id': run_id, 'model_name': model_name, **metrics}
        )

    def notify_model_training_failure(
        self,
        model_name: str,
        run_id: str,
        error: str,
        dag_id: str
    ):
        """Convenience method for training failure notification."""
        return self.send_notification(
            subject=f"Model Training FAILED: {model_name}",
            message=f"Training run {run_id} failed with error:\n\n{error}",
            level="error",
            dag_id=dag_id,
            event_data={'run_id': run_id, 'model_name': model_name, 'error': error}
        )

    def notify_model_deployed(
        self,
        model_name: str,
        run_id: str,
        metrics: Dict[str, float],
        improvement: float,
        dag_id: str
    ):
        """Convenience method for model deployment notification."""
        metrics_str = '\n'.join([f"  - {k}: {v:.4f}" for k, v in metrics.items()])

        return self.send_notification(
            subject=f"New Model Deployed: {model_name}",
            message=f"Model {run_id} deployed to production.\n\nMetrics:\n{metrics_str}\n\nImprovement: {improvement:.2f}%",
            level="info",
            dag_id=dag_id,
            event_data={'run_id': run_id, 'model_name': model_name, 'improvement': improvement, **metrics}
        )

    def notify_model_comparison_failed(
        self,
        model_name: str,
        new_run_id: str,
        baseline_run_id: str,
        new_metrics: Dict[str, float],
        baseline_metrics: Dict[str, float],
        dag_id: str
    ):
        """Convenience method for model comparison failure notification."""
        comparison = '\n'.join([
            f"  - {k}: New={new_metrics.get(k, 0):.4f}, Baseline={baseline_metrics.get(k, 0):.4f}"
            for k in set(new_metrics.keys()) | set(baseline_metrics.keys())
        ])

        return self.send_notification(
            subject=f"Model Did Not Improve: {model_name}",
            message=f"New model {new_run_id} did not improve over baseline {baseline_run_id}.\n\nComparison:\n{comparison}",
            level="warning",
            dag_id=dag_id,
            event_data={
                'new_run_id': new_run_id,
                'baseline_run_id': baseline_run_id,
                'model_name': model_name
            }
        )

    def notify_data_drift_detected(
        self,
        feature_name: str,
        drift_score: float,
        threshold: float,
        dag_id: str
    ):
        """Convenience method for data drift notification."""
        return self.send_notification(
            subject=f"Data Drift Detected: {feature_name}",
            message=f"Data drift detected in feature '{feature_name}'.\n\nDrift Score: {drift_score:.4f}\nThreshold: {threshold:.4f}",
            level="warning",
            dag_id=dag_id,
            event_data={'feature': feature_name, 'drift_score': drift_score, 'threshold': threshold}
        )


# Factory function for easy initialization
def create_notification_system(
    smtp_user: Optional[str] = None,
    smtp_password: Optional[str] = None,
    to_emails: Optional[List[str]] = None,
    slack_webhook_url: Optional[str] = None,
    metadata_tracker=None
) -> NotificationSystem:
    """
    Factory function to create NotificationSystem with common defaults.

    Args:
        smtp_user: SMTP username (usually email)
        smtp_password: SMTP password or app password
        to_emails: List of recipient emails
        slack_webhook_url: Slack webhook URL
        metadata_tracker: MetadataTracker instance

    Returns:
        NotificationSystem instance
    """
    config = NotificationConfig(
        smtp_user=smtp_user,
        smtp_password=smtp_password,
        from_email=smtp_user,
        to_emails=to_emails,
        slack_webhook_url=slack_webhook_url
    )

    return NotificationSystem(config, metadata_tracker)
