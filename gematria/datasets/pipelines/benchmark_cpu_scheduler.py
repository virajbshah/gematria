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

import abc
from collections.abc import Collection
import enum
import os
import re
from typing_extensions import override


class BenchmarkScheduler(metaclass=abc.ABCMeta):
  """Schedules a benchmark and the parent process to reduce noise.

  BenchmarkScheduler is an abstraction that provides two main pieces of
  functionality. Firstly, it provides a function
  (setup_and_get_benchmark_core) that allows an implementation to perform any
  necessary setup in the parent process and provide a core ID that should be
  used to perform any benchmarking. Additionally, implementations are
  intended to hold state to verify that the expected state is maintained and
  not changed by external software.
  """

  @abc.abstractmethod
  def setup_and_get_benchmark_core(self) -> int | None:
    """Sets up the parent process and chooses a benchmark core.

    This function will perform any relevant setup in the parent process,
    and return a core ID that can be used to run benchmarks on.

    Returns:
      Returns an integer core ID to specify a core that should be used for
      running benchmarks, or None to indicate that any core can be used.
    """

  @abc.abstractmethod
  def verify(self):
    """Verifies that conditions match what is expected.

    This function allows for implementations to verify that the original
    setup created in setup_and_get_benchmark_core is maintained for every
    benchmark that is run.
    """


class NoSchedulingBenchmarkScheduler(BenchmarkScheduler):
  """A basic BenchmarkScheduler implementation that does nothing.

  This BenchmarkScheduler implementation does nothing. It leaves scheduling
  of the benchmarking process to the operating system by specifying any core
  can be used for benchmarking, performs no setup in the parent process, and
  performs no verification.
  """

  @override
  def setup_and_get_benchmark_core(self) -> int | None:
    return None

  @override
  def verify(self):
    pass


def _get_neighboring_threads(cpu_index: int) -> list[int]:
  with open(
      f'/sys/devices/system/cpu/cpu{cpu_index}/topology/thread_siblings_list'
  ) as thread_sibling_list_handle:
    neighboring_threads_strings = re.split(
        '[-,]+', thread_sibling_list_handle.read().strip()
    )
    neighboring_threads = [
        int(cpu_index_str) for cpu_index_str in neighboring_threads_strings
    ]
  return neighboring_threads


class DefaultBenchmarkScheduler(BenchmarkScheduler):
  """A BenchmarkScheduler that schedules processes separately.

  DefaultBenchmarkScheduler schedules the main process and the benchmark
  subprocess on separate cores, making sure to reserve the second hyperthread
  on the benchmarking core to prevent interference. It expects that the main
  process is initially given a CPU Mask with three active threads, additionally
  assuming that two of the threads are neighboring (part of the same core).
  Errors are raised if these conditions are not met. The benchmarking core
  returned is one of the two neighboring threads. The main process has its
  CPU mask limited to the thread that neighbors neither of the other threads.
  """

  def __init__(self):
    self._cpu_mask = []

  def _get_aux_core_and_hyperthread_pair(
      self,
      cpu_mask: Collection[int],
  ) -> tuple[int, list[int]]:
    for cpu_index in cpu_mask:
      neighboring_threads = _get_neighboring_threads(cpu_index)
      if len(neighboring_threads) != 2:
        raise ValueError('Expected two hyperthreads per CPU.')

      if (
          neighboring_threads[0] not in cpu_mask
          or neighboring_threads[1] not in cpu_mask
      ):
        continue

      cpus = list(cpu_mask)
      cpus.remove(neighboring_threads[0])
      cpus.remove(neighboring_threads[1])

      if not cpus:
        continue

      return (cpus[0], [neighboring_threads[0], neighboring_threads[1]])
    raise ValueError(
        'Expected a pair of neighboring hyperthreads and an aux thread in the'
        ' CPU mask.'
    )

  @override
  def setup_and_get_benchmark_core(self) -> int | None:
    cpu_mask = os.sched_getaffinity(0)

    if len(cpu_mask) != 3:
      raise ValueError('Expected to have three CPUs.')

    aux_core, hyperthread_pair = self._get_aux_core_and_hyperthread_pair(
        cpu_mask
    )
    os.sched_setaffinity(0, [aux_core])
    self._cpu_mask = [aux_core]

    return hyperthread_pair[0]

  @override
  def verify(self):
    cpu_mask = list(os.sched_getaffinity(0))
    if self._cpu_mask != cpu_mask:
      raise ValueError('Expected the CPU mask to not change.')


class BenchmarkSchedulerImplementations(enum.Enum):
  NO_SCHEDULING = 1
  DEFAULT = 2


def construct_benchmark_scheduler(
    scheduler_type: BenchmarkSchedulerImplementations,
) -> BenchmarkScheduler:
  match scheduler_type:
    case BenchmarkSchedulerImplementations.NO_SCHEDULING:
      return NoSchedulingBenchmarkScheduler()
    case BenchmarkSchedulerImplementations.DEFAULT:
      return DefaultBenchmarkScheduler()
    case _:
      raise ValueError('Unexpected Benchmark Scheduler Type.')
