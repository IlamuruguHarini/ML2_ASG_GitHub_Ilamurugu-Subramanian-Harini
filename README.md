# ML2_ASG_GitHub_Ilamurugu-Subramanian-Harini

Bike Sharing Demand Prediction – MLOps Project

This project aims to predict daily bike sharing demand using machine learning while demonstrating a complete MLOps workflow. The focus is not only on building an accurate predictive model, but also on ensuring that the model is reliable, reproducible, and maintainable through proper experiment tracking, data drift analysis, and automation.

Using the Bike Sharing Daily dataset, multiple regression models were developed and evaluated to forecast daily bike rental counts based on weather, seasonal, and calendar-related features. Model performance was assessed using standard regression metrics, and experiments were systematically tracked using MLflow to enable transparent comparison and model selection.

Beyond model development, the project analyses data drift by comparing feature distributions across different time periods and evaluating how these changes impact model performance. This highlights real-world challenges where data evolves over time and demonstrates strategies such as retraining and monitoring to maintain model accuracy.

To support automation and quality control, GitHub Actions was implemented as a continuous integration (CI) pipeline. The pipeline automatically runs model evaluation tests on every push to the main branch and acts as a quality gate, ensuring that only models meeting predefined performance thresholds are considered acceptable.

Overall, this project showcases how machine learning, MLOps practices, experiment tracking, data drift management, and CI/CD automation can be integrated into a single, end-to-end workflow suitable for real-world deployment scenarios.
