"""Main entry point for Wine Quality ML pipeline.

This script orchestrates the entire ML pipeline:
1. Load and split data
2. Create preprocessing pipeline
3. Train model with hyperparameter tuning
4. Evaluate model on test set
5. Save trained model

Usage:
    python main.py

Follows PEP8 conventions with type hints.
"""

import logging
import sys
from pathlib import Path

# Add project root to path (required before imports)
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import config  # noqa: E402
from data.load_data import load_and_split_data  # noqa: E402
from src.preprocessing import create_preprocessing_pipeline  # noqa: E402
from src.train import (  # noqa: E402
    train_model_with_grid_search,
    save_model,
    get_cv_results,
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


def main() -> None:
    """Run the complete Wine Quality ML pipeline.

    Raises:
        Exception: If any step in the pipeline fails.
    """
    try:
        logger.info("\n" + "=" * 70)
        logger.info("WINE QUALITY ML PIPELINE - PRODUCTION VERSION")
        logger.info("=" * 70 + "\n")

        # Step 1: Load and split data
        logger.info("STEP 1: Loading and splitting data")
        X_train, X_test, y_train, y_test = load_and_split_data()

        # Step 2: Create preprocessing pipeline
        logger.info("\nSTEP 2: Creating preprocessing + model pipeline")
        pipeline = create_preprocessing_pipeline()

        # Step 3: Train model with hyperparameter tuning
        logger.info("\nSTEP 3: Training model with GridSearchCV")
        trained_model = train_model_with_grid_search(pipeline, X_train, y_train)

        # Step 4: Evaluate model on test set
        logger.info("\nSTEP 4: Evaluating model on test set")
        metrics = evaluate_model(trained_model, X_test, y_test)

        # Print sample predictions
        print_prediction_summary(trained_model, X_test, y_test, n_samples=3)

        # Compare with baseline
        compare_with_baseline(metrics)

        # Step 5: Save trained model
        logger.info("\nSTEP 5: Saving trained model")
        save_model(trained_model)

        # Print CV results summary
        logger.info("\nCross-Validation Results (Top 5):")
        cv_results = get_cv_results(trained_model)
        logger.info(f"\n{cv_results.head().to_string()}")

        # Final summary
        logger.info("\n" + "=" * 70)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 70)
        logger.info("\nFinal Model Performance:")
        logger.info(f"  R² Score: {metrics['r2_score']:.4f}")
        logger.info(f"  RMSE:     {metrics['rmse']:.4f}")
        logger.info(f"\nModel saved to: {config.MODEL_PATH}")
        logger.info(f"Logs saved to:  {config.LOG_FILE}")
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
