"""Main entry point for Wine Quality ML pipeline.

This script orchestrates the entire ML pipeline:
1. Load and split data
2. Create preprocessing pipeline
3. Train model with hyperparameter tuning
4. Evaluate model on test set
5. Save trained model

Usage:
    python main.py                    # Compare both pipelines
    python main.py --pipeline rf      # Train only RandomForest
    python main.py --pipeline gb      # Train only GradientBoosting

Follows PEP8 conventions with type hints.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path (required before imports)
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import mlflow  # noqa: E402

import config  # noqa: E402
from data.load_data import load_and_split_data  # noqa: E402
from src.preprocessing import (  # noqa: E402
    create_preprocessing_pipeline,
    create_alternative_pipeline,
)
from src.train import (  # noqa: E402
    train_model_with_grid_search,
    save_model,
    get_cv_results,
    log_model_to_mlflow,
    compare_pipelines,
)
from src.evaluate import (  # noqa: E402
    evaluate_model,
    print_prediction_summary,
    compare_with_baseline,
)

# Configure logging
logging.basicConfig(
    format=config.LOG_FORMAT,
    level=config.LOG_LEVEL,
    handlers=[
        logging.StreamHandler(),  # Console output
        logging.FileHandler(config.LOG_FILE),  # File output
    ],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Wine Quality ML Pipeline with MLflow tracking"
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        choices=["rf", "gb", "compare"],
        default="compare",
        help=(
            "Pipeline to train: 'rf' (RandomForest), 'gb' (GradientBoosting), "
            "or 'compare' (both pipelines). Default: compare"
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Run the complete Wine Quality ML pipeline with MLflow tracking.

    Raises:
        Exception: If any step in the pipeline fails.
    """
    # Parse command line arguments
    args = parse_args()

    # Set MLflow tracking URI and experiment
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)

    try:
        logger.info("\n" + "=" * 70)
        logger.info("WINE QUALITY ML PIPELINE - PRODUCTION VERSION WITH MLFLOW")
        logger.info("=" * 70)
        logger.info(f"Pipeline mode: {args.pipeline.upper()}")
        logger.info("=" * 70 + "\n")

        # Step 1: Load and split data
        logger.info("STEP 1: Loading and splitting data")
        X_train, X_test, y_train, y_test = load_and_split_data()

        # Choose pipeline based on argument
        if args.pipeline == "compare":
            # Compare both pipelines
            logger.info("\nSTEP 2: Creating both pipelines for comparison")
            pipeline_rf = create_preprocessing_pipeline()
            pipeline_gb = create_alternative_pipeline()

            logger.info("\nSTEP 3: Comparing pipelines...")
            results = compare_pipelines(
                pipeline_rf,
                pipeline_gb,
                X_train,
                y_train,
                X_test,
                y_test,
            )

            # Save both models
            logger.info("\nSTEP 4: Saving trained models")
            save_model(results["rf"], config.MODEL_PATH)
            save_model(results["gb"], config.GB_MODEL_PATH)

            logger.info("\n" + "=" * 70)
            logger.info("PIPELINE COMPARISON COMPLETED SUCCESSFULLY!")
            logger.info("=" * 70)
            logger.info("\nModels saved to:")
            logger.info(f"  RandomForest:      {config.MODEL_PATH}")
            logger.info(f"  GradientBoosting:  {config.GB_MODEL_PATH}")
            logger.info(f"\nLogs saved to:  {config.LOG_FILE}")
            logger.info("MLflow UI:      mlflow ui --port 5000")
            logger.info("\n" + "=" * 70 + "\n")

        else:
            # Train single pipeline
            pipeline_type = (
                "RandomForest" if args.pipeline == "rf" else "GradientBoosting"
            )
            model_path = (
                config.MODEL_PATH if args.pipeline == "rf" else config.GB_MODEL_PATH
            )
            param_grid = (
                config.HYPERPARAM_GRID
                if args.pipeline == "rf"
                else config.GB_HYPERPARAM_GRID
            )

            # Start MLflow run
            with mlflow.start_run(run_name=f"{pipeline_type}_Pipeline"):
                logger.info("MLflow run started")
                logger.info(f"Run ID: {mlflow.active_run().info.run_id}")
                logger.info(f"Experiment: {config.MLFLOW_EXPERIMENT_NAME}\n")

                # Log dataset info
                mlflow.log_param("train_samples", len(X_train))
                mlflow.log_param("test_samples", len(X_test))
                mlflow.log_param("n_features", X_train.shape[1])
                mlflow.log_param("test_size", config.TEST_SIZE)
                mlflow.log_param("pipeline_type", pipeline_type)

                # Step 2: Create preprocessing pipeline
                logger.info(f"\nSTEP 2: Creating {pipeline_type} pipeline")
                if args.pipeline == "rf":
                    pipeline = create_preprocessing_pipeline()
                else:
                    pipeline = create_alternative_pipeline()

                # Step 3: Train model with hyperparameter tuning
                logger.info(
                    f"\nSTEP 3: Training {pipeline_type} model with GridSearchCV"
                )
                trained_model = train_model_with_grid_search(
                    pipeline, X_train, y_train, param_grid
                )

                # Step 4: Evaluate model on test set
                logger.info("\nSTEP 4: Evaluating model on test set")
                metrics = evaluate_model(trained_model, X_test, y_test)

                # Print sample predictions
                print_prediction_summary(trained_model, X_test, y_test, n_samples=3)

                # Compare with baseline
                compare_with_baseline(metrics)

                # Step 5: Log to MLflow
                logger.info("\nSTEP 5: Logging to MLflow")
                log_model_to_mlflow(trained_model, metrics)

                # Step 6: Save trained model
                logger.info("\nSTEP 6: Saving trained model")
                save_model(trained_model, model_path)

                # Print CV results summary
                logger.info("\nCross-Validation Results (Top 5):")
                cv_results = get_cv_results(trained_model)
                logger.info(f"\n{cv_results.head().to_string()}")

                # Final summary
                logger.info("\n" + "=" * 70)
                logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
                logger.info("=" * 70)
                logger.info(f"\n{pipeline_type} Model Performance:")
                logger.info(f"  R² Score: {metrics['r2_score']:.4f}")
                logger.info(f"  RMSE:     {metrics['rmse']:.4f}")
                logger.info(f"\nModel saved to: {model_path}")
                logger.info(f"Logs saved to:  {config.LOG_FILE}")
                logger.info("MLflow UI:      mlflow ui --port 5000")
                logger.info("\n" + "=" * 70 + "\n")

    except Exception as e:
        logger.error(f"\n{'='*70}")
        logger.error("PIPELINE FAILED!")
        logger.error(f"{'='*70}")
        logger.error(f"Error: {e}", exc_info=True)
        logger.error(f"{'='*70}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
