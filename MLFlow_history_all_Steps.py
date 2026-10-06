import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
runs = mlflow.search_runs(experiment_names=["nerve_segmentation"])
runs["balanced"] = 0.5 * runs["metrics.dice_nerve_frames"] + 0.5 * runs["metrics.empty_kept_empty"]
cols = ["tags.mlflow.runName", "start_time", "metrics.dice_all", "metrics.dice_nerve_frames",
        "metrics.nerve_found", "metrics.empty_kept_empty", "balanced"]
runs[cols].sort_values("start_time")



'''

                        #### OUTPUT #######


	tags.mlflow.runName	start_time	metrics.dice_all	metrics.dice_nerve_frames	metrics.nerve_found	metrics.empty_kept_empty	balanced
6	step1_baseline	        2026-10-05 19:34:22.395000+00:00	0.501620	0.303859	0.719934	0.704392	0.504125
5	step2_oversample	2026-10-05 19:41:16.987000+00:00	0.484054	0.437200	0.840198	0.532095	0.484647
4	step3_dice_loss	        2026-10-05 19:44:42.352000+00:00	0.561481	0.407275	0.680395	0.719595	0.563435
3	step3_dice_loss	        2026-10-05 19:50:47.941000+00:00	0.550733	0.476653	0.759473	0.626689	0.551671
2	step4_detection_head	2026-10-05 19:54:49.938000+00:00	0.606644	0.290554	0.392092	0.930743	0.610649
1	step5_scoring_cleanup	2026-10-05 19:57:40.364000+00:00	0.627499	0.409179	0.545305	0.851351	0.630265
0	step7_ci_and_export	2026-10-05 20:05:31.536000+00:00	0.498023	0.008451	0.011532	1.000000	0.504226



'''
