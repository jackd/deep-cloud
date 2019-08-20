#!/bin/bash
# execute from root deep-cloud directory with `./examples/weighpoint/cls.sh`
python -m more_keras --mk_config=train_meta --config_dir=configs --config='
problems/pointnet
augment/j1e-2
models/weighpoint/cls
'
