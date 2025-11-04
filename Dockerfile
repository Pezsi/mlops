# Base Docker image: Anaconda
FROM continuumio/anaconda3

# Set working directory within the container
WORKDIR /app

# Copy environment configuration and requirements
COPY environment.yml /app/
COPY requirements.txt /app/

# Install conda environment based on environment.yml
RUN conda env create -f environment.yml

# Activate the environment and install pip dependencies
RUN /bin/bash -c "source activate wine_quality_mlops && pip install -r requirements.txt"

# Copy application files
COPY config.py /app/
COPY main.py /app/
COPY fastapi_app.py /app/
COPY flask_app.py /app/

# Copy source directories
COPY data/ /app/data/
COPY src/ /app/src/
COPY models/ /app/models/

# Create mlruns directory (will be populated at runtime)
RUN mkdir -p /app/mlruns /app/logs

# Set environment variables
ENV MLFLOW_TRACKING_URI="file:/app/mlruns"
ENV PATH="/opt/conda/envs/wine_quality_mlops/bin:$PATH"
ENV PYTHONPATH="/app"

# Set permissions for the mlruns and logs folders
RUN chmod -R 777 /app/mlruns /app/logs

# Expose ports for MLflow UI, FastAPI, and Flask
EXPOSE 5000 8000 8080

# Start MLflow server and FastAPI application
CMD ["/bin/bash", "-c", "source activate wine_quality_mlops && mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri file:/app/mlruns --default-artifact-root /app/mlruns & python fastapi_app.py"]