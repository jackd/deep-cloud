from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def repeat_configurable(configurable, num_repeats, name='{orig}-{index}'):
    """Get `num_repeats + 1` versions of configurable."""
    config = configurable.get_config()
    cls = type(configurable)
    orig = config.pop('name')
    out = []
    for i in range(num_repeats + 1):
        config['name'] = name.format(orig=orig, index=i)
        out.append(cls.from_config(config))
    return out
