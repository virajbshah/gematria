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

import os

from absl import app
from absl import flags
import apache_beam as beam
from apache_beam.options import pipeline_options

from gematria.datasets.pipelines import compile_modules_lib
from gematria.datasets.python import bhive_to_exegesis

ANNOTATOR_MAPPING = {
    'fast': bhive_to_exegesis.AnnotatorType.fast,
    'exegesis': bhive_to_exegesis.AnnotatorType.exegesis,
}

_PARQUET_FOLDER = flags.DEFINE_string(
    'parquet_folder',
    None,
    'The path to the folder containing parquet files',
)

_OUTPUT_FILE = flags.DEFINE_string(
    'output_file', None, 'The path to the output tfrecord file.', required=True
)

_OUTPUT_VOCAB_FILE = flags.DEFINE_string(
    'output_vocab_file',
    None,
    'The path to the output vocab text file.',
    required=True,
)

_REMOVE_MEMORY_ACCESSING_INSTRUCTIONS = flags.DEFINE_bool(
    'remove_memory_accessing_instructions',
    False,
    'Whether to remove memory accessing instructions from the basic blocks.',
)

_ANNOTATOR_TYPE = flags.DEFINE_enum(
    'annotator_type',
    'fast',
    sorted(ANNOTATOR_MAPPING.keys()),
    'The type of annotator to use.',
)

_MAX_ANNOTATION_ATTEMPTS = flags.DEFINE_integer(
    'max_annotation_attempts',
    50,
    'The maximum number of times to try annotating a block before giving up',
)

_SKIP_NO_LOOP_REGISTER = flags.DEFINE_bool(
    'skip_no_loop_register',
    False,
    'Whether or not to skip emitting basic blocks for which a loop register'
    ' cannot be found.',
)

_INPUT_HEX_BBS_FILE_PATTERN = flags.DEFINE_string(
    'input_hex_bbs_file_pattern',
    None,
    'The path to text files containing new line separated basic blocks.',
)


def main(argv) -> None:
  del argv  # Unused.

  beam_options = pipeline_options.PipelineOptions()

  pipeline_constructor = compile_modules_lib.get_bbs(
      input_file_pattern=os.path.join(_PARQUET_FOLDER.value, '*.parquet'),
      output_file=_OUTPUT_FILE.value,
      remove_memory_accessing_instructions=_REMOVE_MEMORY_ACCESSING_INSTRUCTIONS.value,
      annotator_type=ANNOTATOR_MAPPING[_ANNOTATOR_TYPE.value],
      max_annotation_attempts=_MAX_ANNOTATION_ATTEMPTS.value,
      vocab_output_file=_OUTPUT_VOCAB_FILE.value,
      skip_no_loop_register=_SKIP_NO_LOOP_REGISTER.value,
      input_hex_bbs_file_pattern=_INPUT_HEX_BBS_FILE_PATTERN.value,
  )

  with beam.Pipeline(options=beam_options) as pipeline:
    pipeline_constructor(pipeline)


if __name__ == '__main__':
  app.run(main)
