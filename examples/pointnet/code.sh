#!/bin/bash
# execute from root deep-cloud directory with `./examples/pointnet/code.sh`
python -m more_keras \
--mk_config=train \
--config_dir=configs \
--config='models/pointnet/code'
