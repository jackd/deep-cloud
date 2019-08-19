from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
from more_keras.framework.problems.tfds import TfdsProblem
from more_keras.framework.pipelines import Pipeline
from more_keras.framework.train import train
from more_keras.callbacks import BetterModelCheckpoint

Pipeline = gin.external_configurable(Pipeline, module='mk')
train = gin.external_configurable(train, module='mk')
BetterModelCheckpoint = gin.external_configurable(BetterModelCheckpoint,
                                                  module='mk.callbacks')
