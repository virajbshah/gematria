# Copyright 2022 Google Inc.
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
"""Helper function for running inference with Gematria models."""

from collections.abc import Iterable
import itertools
from typing import Optional, Union

from absl import logging
from gematria.basic_block.python import throughput_protos
from gematria.model.python import model_base
from gematria.model.python import training
from gematria.proto import throughput_pb2


def _get_num_instructions_in_block_with_throughput_proto(
    proto: throughput_pb2.BasicBlockWithThroughputProto,
) -> int:
  return len(proto.basic_block.canonicalized_instructions)


def _get_num_instructions_in_block_with_throughput_list_proto(
    proto: throughput_pb2.BasicBlockWithThroughputListProto,
) -> int:
  return sum(
      _get_num_instructions_in_block_with_throughput_proto(block)
      for block in proto.basic_blocks
  )


def predict_for_protos(
    model: model_base.ModelBase,
    blocks_or_traces: Union[
        Iterable[throughput_pb2.BasicBlockWithThroughputProto],
        Iterable[throughput_pb2.BasicBlockWithThroughputListProto],
    ],
    max_traces_in_batch: Optional[int] = None,
    max_instructions_in_batch: Optional[int] = None,
) -> Iterable[throughput_pb2.BasicBlockWithThroughputListProto]:
  """Predicts the inverse throughput using the model.

  Assumes that model has been initialized and that it contains the appropriate
  weights. The input sequence is iterated through only once, and the method may
  safely be used with iterable objects that read the protos from a file or
  generate them on the fly.

  Args:
    model: The model used for inference.
    blocks_or_traces: The collection of basic blocks or basic block traces for
      which the inverse throughput is predicted.
    max_traces_in_batch: The maximal number of traces processed in a single
      batch. When not specified, the number of traces in a batch is unlimited.
    max_instructions_in_batch: The maximal number of instructions across all
      basic blocks processed in a single batch. When not specified, the number
      of instructions in a batch is unlimited.

  Yields:
    The basic block traces or individual blocks wrapped in traces from
    blocks_or_traces. Each basic block has a new inverse_throughputs value added
    to it with the prediction from the model.
  """
  trace_protos: Iterable[throughput_pb2.BasicBlockWithThroughputListProto]
  blocks_or_traces, peek = itertools.tee(blocks_or_traces)
  if isinstance(next(peek), throughput_pb2.BasicBlockWithThroughputProto):
    trace_protos = (
        throughput_pb2.BasicBlockWithThroughputListProto(basic_blocks=(block,))
        for block in blocks_or_traces
    )
  else:
    trace_protos = blocks_or_traces
  batches = training.batches(
      trace_protos,
      get_num_instructions=(
          _get_num_instructions_in_block_with_throughput_list_proto
      ),
      max_traces_in_batch=max_traces_in_batch,
      max_instructions_in_batch=max_instructions_in_batch,
  )
  for batch_index, protos in enumerate(batches):
    logging.info(
        'Processing proto batch %d (%d traces, %d blocks).',
        batch_index,
        len(protos),
        sum(len(trace.basic_blocks) for trace in protos),
    )
    traces = []
    trace_is_valid = [False] * len(protos)
    for proto_index, proto in enumerate(protos):
      # fmt: off
      trace = [
          block_with_throughput.block
          for block_with_throughput
          in throughput_protos.block_with_throughput_list_from_proto(proto)
      ]
      # fmt: on
      if all(model.validate_basic_block(block) for block in trace):
        trace_is_valid[proto_index] = True
        traces.append(trace)

    # Trace are already divided into batches according to the given criteria,
    # no need to use max_traces_in_batch and max_instructions_in_batch again.
    trace_predictions = iter(model.predict(traces))

    # Inject predictions into the input protos.
    for trace_proto, is_valid in zip(protos, trace_is_valid):
      if is_valid:
        trace_prediction = next(trace_predictions)
        for block_proto, block_prediction in zip(
            trace_proto.basic_blocks, trace_prediction
        ):
          for task_index, task_predictions in zip(
              range(model.num_tasks), block_prediction.throughputs
          ):
            task_prefix_predictions = (
                task_predictions.prefix_inverse_throughput_cycles
            )
            task_throughput = block_proto.inverse_throughputs.add(
                source=model.get_source_name(task_index),
                inverse_throughput_cycles=(
                    task_predictions.inverse_throughput_cycles
                ),
            )
            for prefix_predictions in task_prefix_predictions:
              task_throughput.prefix_inverse_throughputs.add(
                  inverse_throughput_cycles=prefix_predictions
              )
      yield trace_proto
