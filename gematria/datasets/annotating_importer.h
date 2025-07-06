// Copyright 2024 Google Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Contains support code for importing and annotating basic block data.

#ifndef THIRD_PARTY_GEMATRIA_GEMATRIA_DATASETS_ANNOTATING_IMPORTER_H_
#define THIRD_PARTY_GEMATRIA_GEMATRIA_DATASETS_ANNOTATING_IMPORTER_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "gematria/datasets/bhive_importer.h"
#include "gematria/llvm/canonicalizer.h"
#include "gematria/llvm/disassembler.h"
#include "gematria/proto/throughput.pb.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ELFTypes.h"
#include "quipper/perf_data.pb.h"
#include "quipper/perf_parser.h"
#include "quipper/perf_reader.h"

namespace gematria {

class PerfEventImporter {
 public:
  struct SampleTable {
    std::vector<std::string> sample_type_names;
    std::unordered_map<uint64_t, std::vector<int>> samples_at_ip;
  };

  // Creates and returns a `PerfEventImporter` that loads a `perf.data`-like
  // file from `perf_data_path`.
  static absl::StatusOr<PerfEventImporter> Create(
      std::string_view perf_data_path);

  // Searches all MMap events for the one that most likely corresponds to the
  // executable load segment with the given `binary_basename`.
  // This requires that the ELF object's filename has not changed from when it
  // was profiled, since we check its name against the filenames from the
  // recorded MMap events. Note the object file can still be moved, since we
  // check only the name and not the path.
  // TODO(virajbshah): Find a better way to identify the relevant mapping.
  absl::StatusOr<const quipper::PerfDataProto::MMapEvent*>
  GetMMapEventForBinary(std::string_view binary_basename);

  // Extracts samples corresponding to the object with name `binary_basename`.
  // Returns a tabular format holding `sample_type_name`, a header with the
  // names of sample types extracted, and `samples_at_ip`, the body mapping
  // instruction addresses to sample counts. Each count corresponds to the entry
  // of `sample_type_names` at the same index.
  absl::StatusOr<SampleTable> GetPerfSampleTable(
      std::string_view binary_basename);

  // Extracts events from the `perf.data`-like file loaded while creating this
  // `PerfEventImporter`. Returns a callable, each call to which returns one
  // perf event. The total number of events returned across all calls to this
  // callable can be set using `max_num_events`. If this is set to -1, the
  // number of events returned is not limited. Events can be filtered by
  // passing a callable taking an event as `event_filter`. Only events for which
  // this returns true will be included.
  template <typename Filter>
  absl::StatusOr<
      std::function<absl::StatusOr<quipper::PerfDataProto::PerfEvent>()>>
  GetPerfEvents(
      std::string_view binary_basename, int max_num_events = -1,
      Filter event_filter = [](auto... args) { return true; });

 private:
  explicit PerfEventImporter()
      : perf_reader_(std::make_unique<quipper::PerfReader>()) {};

  // Owns the `PerfDataProto` that samples and branch records are read from.
  std::unique_ptr<quipper::PerfReader> perf_reader_;
};

class LBRImporter {
  using AddressRange = std::pair<uint64_t, uint64_t>;
  struct LBRBlockData {
    AddressRange address_range;
    uint32_t latency;
  };
  using LBRTraceData = std::vector<LBRBlockData>;

 public:
  // Creates an `LBRImporter` that reads blocks from the binary at `binary_path`
  // based on branch stacks read from the `perf.data`-like file at
  // `perf_data_path`. The canonicalizer must be for the
  // architecture/microarchitecture of the data set. Does not take ownership of
  // the canonicalizer.
  static absl::StatusOr<LBRImporter> Create(std::string_view binary_path,
                                            std::string_view perf_data_path,
                                            const Canonicalizer* canonicalizer);

  // Extracts basic block trace protos from the `perf.data`-like file loaded
  // while creating this `LBRImporter`. Returns a callable, each call to which
  // returns one trace proto.
  absl::StatusOr<
      std::function<absl::StatusOr<BasicBlockWithThroughputListProto>()>>
  GetLBRTraceProtos(std::string_view source_name);

 private:
  explicit LBRImporter(
      llvm::object::OwningBinary<llvm::object::Binary>&& binary,
      const llvm::object::ELFObjectFileBase* elf_object,
      PerfEventImporter&& perf_event_importer, BHiveImporter&& bhive_importer)
      : binary_(std::move(binary)),
        elf_object_(elf_object),
        perf_event_importer_(std::move(perf_event_importer)),
        bhive_importer_(std::move(bhive_importer)) {};

  // Extracts basic block trace data (addresses and latencies) from the
  // `perf.data`-like file loaded while creating this `LBRImporter`. Returns
  // a callable, each call to which returns data for one trace.
  absl::StatusOr<std::function<absl::StatusOr<LBRTraceData>()>>
  GetLBRTraceData();

  // Disassembles and returns instructions between two addresses from the binary
  // loaded while creating this `LBRImporter`.
  absl::StatusOr<std::vector<DisassembledInstruction>>
  GetInstructionsInAddressRange(AddressRange address_range, uint64_t offset);

  // Disassembles and returns a single instruction at a given address from the
  // binary loaded while creating this `LBRImporter`.
  absl::StatusOr<DisassembledInstruction> GetInstructionAtAddress(
      uint64_t address, uint64_t offset);

  // Handles to the binary data is imported from. `binary_` owns the underlying
  // object, while `elf_object_` provides convenient access to it.
  llvm::object::OwningBinary<llvm::object::Binary> binary_;
  const llvm::object::ELFObjectFileBase* elf_object_;

  PerfEventImporter perf_event_importer_;
  BHiveImporter bhive_importer_;
};

// Importer for annotated basic blocks and traces.
class AnnotatingImporter {
 public:
  // Creates a new annotation collector from a given canonicalizer. The
  // canonicalizer must be for the architecture/microarchitecture of the data
  // set. Does not take ownership of the canonicalizer.
  explicit AnnotatingImporter(const Canonicalizer* canonicalizer)
      : importer_(canonicalizer) {};

  // Reads an ELF object along with a corresponding `perf.data`-like file and
  // returns a vector of annotated `BasicBlockProto`s consisting of basic blocks
  // from the ELF object annotated using samples from the `perf.data`-like file.
  absl::StatusOr<
      std::function<absl::StatusOr<BasicBlockWithThroughputListProto>()>>
  GetAnnotatedBasicBlockProtos(std::string_view elf_file_name,
                               std::string_view perf_data_file_name,
                               std::string_view source_name);

 private:
  BHiveImporter importer_;

  // Owns the `PerfDataProto` that samples and branch records are read from.
  quipper::PerfReader perf_reader_;
};

template <typename Filter>
absl::StatusOr<
    std::function<absl::StatusOr<quipper::PerfDataProto::PerfEvent>()>>
PerfEventImporter::GetPerfEvents(const std::string_view binary_basename,
                                 int max_num_events, Filter event_filter) {
  const quipper::PerfDataProto& perf_data = perf_reader_->proto();
  const auto& events = perf_data.events();

  int num_events = 0;
  auto sample_generator = [&events, event_it = events.cbegin(), num_events,
                           max_num_events, event_filter]() mutable
      -> absl::StatusOr<quipper::PerfDataProto::PerfEvent> {
    for (; event_it != events.cend() &&
           (max_num_events == -1 || num_events < max_num_events);
         ++event_it) {
      if (event_filter(*event_it)) {
        const quipper::PerfDataProto::PerfEvent& event = *event_it;
        ++event_it;
        ++num_events;
        return event;
      }
    }
    return absl::OutOfRangeError(
        "Iteration complete, all events have been returned.");
  };
  return sample_generator;
}

}  // namespace gematria

#endif  // THIRD_PARTY_GEMATRIA_GEMATRIA_DATASETS_ANNOTATING_IMPORTER_H_
