# Experiment Tracker

This is an experiment tracker file. Entries are not going to be deleted, only new rows can be appended to the table.

**File locations:**
- params: `models/<timestamp>/used_config.yaml`
- loss plot: `models/<timestamp>/loss_history_all_folds.png`

Adhere to the rules while adding a new entry

| timestamp | exp | params | kaggle lb | comment | loss |
|-----------|-----|--------|----|---------|----- |
| 2025-09-10-08-33-25 | initial exp with all modalities | ```bash arch=tf_efficientnet_b0, img_size=224, batch_size=8, epochs=5, lr=0.0001, modalities=[CTA,MRA,MRI T2,MRI T1post], roi_box_fraction=0.15, window_offsets=[-2,-1,0,1,2], 5-fold CV``` | 0.56 | initial benchmark | ![loss plot](models/2025-09-10-08-33-25/loss_history_all_folds.png) |
| 2025-09-11-20-34-47 | increased batch size and epochs | ```bash arch=tf_efficientnet_b0, img_size=224, batch_size=64, epochs=20, lr=0.0001, modalities=[CTA,MRA,MRI T2,MRI T1post], roi_box_fraction=0.15, window_offsets=[-2,-1,0,1,2], 5-fold CV``` | 0.59 | increased batch size from 8 to 64, epochs from 5 to 20 | ![loss plot](models/2025-09-11-20-34-47/loss_history_all_folds.png) |
| 2025-09-15-05-27-19 | binary classification model | ```bash arch=tf_efficientnet_b0, img_size=224, batch_size=64, epochs=20, lr=0.0001, modalities=[CTA,MRA,MRI T2,MRI T1post], roi_box_fraction=0.15, window_offsets=[-2,-1,0,1,2], 5-fold CV, num_classes=1``` | - | binary model for aneurysm present/absent classification, used in two-step inference | ![loss plot](models/2025-09-15-05-27-19/loss_history_all_folds.png) |
