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

r"""Creates an annotated Gematria data set from an ELF object and perf samples.

Reads basic blocks from an ELF object, along with event samples and block
latencies from a `perf.data`-like file, usually generated using `perf record`,
and writes a Gematria TFRecord dataset containing basic blocks with instruction
annotations as derived from the samples.
"""

from collections.abc import Sequence

from absl import app
from absl import flags
from absl import logging
from gematria.datasets.python import annotating_importer
from gematria.llvm.python import canonicalizer
from gematria.llvm.python import llvm_architecture_support
from gematria.proto import throughput_pb2
from pybind11_abseil import status
import tensorflow as tf


_INPUT_ELF_FILE = flags.DEFINE_string(
    'gematria_input_elf',
    None,
    'The name of the ELF file from which basic blocks are to be imported.',
    required=True,
)
_INPUT_PERF_FILE = flags.DEFINE_string(
    'gematria_input_perf_data',
    None,
    'The name of the `perf.data`-like file from which samples are to be'
    ' imported.',
    required=True,
)
_OUTPUT_TFRECORD_FILE = flags.DEFINE_string(
    'gematria_output_tfrecord',
    None,
    'The name of the TFRecord file to write the data to.',
    required=True,
)
_SOURCE_NAME = flags.DEFINE_string(
    'gematria_throughput_source_name',
    'perf_lbr_data',
    'The name of the throughput source used for the throughput data.',
    required=False,
)
_LLVM_TRIPLE = flags.DEFINE_string(
    'gematria_llvm_triple',
    'x86_64',
    'The LLVM triple used for disassembling the instructions in the data set.',
)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  try:
    llvm = llvm_architecture_support.LlvmArchitectureSupport.from_triple(
        _LLVM_TRIPLE.value
    )
  except status.StatusNotOk:
    logging.exception(
        'LLVM triple "%s" is not known or supported.', _LLVM_TRIPLE.value
    )
    exit(1)

  # TODO(ondrasej): Update this so that the canonicalizer is created using the
  # LLVM triple. As of 2024-08, this is OK, because we support only x86-64
  # anyway.
  canonicalizer_obj = canonicalizer.Canonicalizer.x86_64(llvm)
  importer = annotating_importer.LBRImporter.create(
      _INPUT_ELF_FILE.value, _INPUT_PERF_FILE.value, canonicalizer_obj
  )

  proto_generator = importer.get_lbr_trace_proto_generator(
      _SOURCE_NAME.value,
  )

  with tf.io.TFRecordWriter(_OUTPUT_TFRECORD_FILE.value) as writer:
    num_protos_written = 0
    while True:
      try:
        proto = proto_generator()
      except status.StatusNotOk as e:
        if e.status.code() == status.StatusCode.OUT_OF_RANGE:
          print(f'Wrote {num_protos_written} proto(s).')
          break
        else:
          raise e

      writer.write(proto.SerializeToString())
      num_protos_written += 1
      if num_protos_written % 1000 == 0:
        print(f'Wrote {num_protos_written} proto(s).')


if __name__ == '__main__':
  app.run(main)
