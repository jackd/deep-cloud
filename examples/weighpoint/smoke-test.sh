#!/bin/bash
# execute from root deep-cloud directory with `./examples/weighpoint/smoke-test.sh`
python -m more_keras \
    --mk_config=train_meta \
    --config_dir=configs \
    --config='
        models/weighpoint/cls
        smoke-test
        '
