import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
runs = mlflow.search_runs(experiment_names=["nerve_segmentation"])
runs["balanced"] = 0.5 * runs["metrics.dice_nerve_frames"] + 0.5 * runs["metrics.empty_kept_empty"]
cols = ["tags.mlflow.runName", "start_time", "metrics.dice_all", "metrics.dice_nerve_frames",
        "metrics.nerve_found", "metrics.empty_kept_empty", "balanced"]
runs[cols].sort_values("start_time")
