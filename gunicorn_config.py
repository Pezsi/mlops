# Gunicorn configuration for MLflow server
bind = "0.0.0.0:5000"
workers = 4
timeout = 120
accesslog = "-"
errorlog = "-"
loglevel = "info"

# Allow connections from Docker network
forwarded_allow_ips = "*"
proxy_allow_ips = "*"
