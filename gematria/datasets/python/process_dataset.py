# Copyright 2024 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Utility to processes and manipulate Gematria TFRecord datasets."""

import os
from random import shuffle

from absl import app
from absl import flags
from absl import logging

from gematria.io.python import tfrecord
from gematria.proto import throughput_pb2
import tensorflow as tf

_INPUT_TFRECORD_FILE = flags.DEFINE_string(
    'gematria_input_tfrecord',
    None,
    'The path to the Gematria TFRecord file to process',
    required=True,
)


def write_protos(protos, path):
  with tf.io.TFRecordWriter(path) as writer:
    for proto in protos:
      writer.write(proto.SerializeToString())
  logging.info(f'Wrote {len(protos)} protos to {path}.')

def split_in_ratio(protos, ratio):
  num_protos = len(protos)

  num_parts = sum(ratio)
  distribution = [num_protos * part // num_parts for part in ratio]
  distribution[-1] += num_protos - sum(distribution)

  splits, split_begin = [], 0
  for split_size in distribution:
    splits.append(protos[split_begin:split_begin + split_size])
    split_begin += split_size

  return splits

def main(argv) -> None:
  del argv  # Unused.

  protos = list(tfrecord.read_protos(
      [_INPUT_TFRECORD_FILE.value], throughput_pb2.BasicBlockWithThroughputProto
  ))

  output_dir = os.path.dirname(_INPUT_TFRECORD_FILE.value)

  shuffle(protos)
  train, test = split_in_ratio(protos, (9, 1))
  write_protos(train, os.path.join(output_dir, 'train.tfrecord'))
  write_protos(test, os.path.join(output_dir, 'test.tfrecord'))


if __name__ == '__main__':
  app.run(main)
