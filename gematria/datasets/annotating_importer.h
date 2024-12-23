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
#include "gematria/llvm/llvm_to_absl.h"
#include "gematria/proto/throughput.pb.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ELFTypes.h"
#include "llvm/Support/Error.h"
#include "quipper/perf_data.pb.h"
#include "quipper/perf_parser.h"
#include "quipper/perf_reader.h"

namespace gematria {

// Importer for annotated basic blocks.
class AnnotatingImporter {
 public:
  // Creates a new annotation collector from a given canonicalizer. The
  // canonicalizer must be for the architecture/microarchitecture of the data
  // set. Does not take ownership of the canonicalizer.
  explicit AnnotatingImporter(const Canonicalizer* canonicalizer);

  // Reads an ELF object along with a corresponding `perf.data`-like file and
  // returns a vector of annotated `BasicBlockProto`s consisting of basic blocks
  // from the ELF object annotated using samples from the `perf.data`-like file.
  absl::StatusOr<std::vector<BasicBlockWithThroughputProto>>
  GetAnnotatedBasicBlockProtos(std::string_view elf_file_name,
                               std::string_view perf_data_file_name,
                               std::string_view source_name);

 private:
  // Loads a `perf.data`-like file into the importer. Must be called before
  // `GetSamples`, `GetLBRData`, and `GetLBRBlocksWithLatency`.
  absl::Status LoadPerfData(std::string_view file_name);

  // Loads a binary into for use by the importer.
  absl::StatusOr<llvm::object::OwningBinary<llvm::object::Binary>> LoadBinary(
      std::string_view file_name);

  // Returns a pointer inside the loaded binary casted down to an ELF object.
  // The pointer is only valid for as long as the passed pointer to binary is.
  absl::StatusOr<llvm::object::ELFObjectFileBase const*> GetELFFromBinary(
      const llvm::object::Binary* binary);

  // Returns the program header corresponding to the main executable section.
  template <class ELFT>
  absl::StatusOr<llvm::object::Elf_Phdr_Impl<ELFT>> GetMainProgramHeader(
      const llvm::object::ELFObjectFile<ELFT>* elf_object);

  // Disassembles and returns instructions between two addresses in an ELF
  // object.
  absl::StatusOr<std::vector<DisassembledInstruction>> GetELFSlice(
      const llvm::object::ELFObjectFileBase* elf_object, uint64_t range_begin,
      uint64_t range_end, uint64_t file_offset);

  // Extracts basic blocks from an ELF object, and returns them as tuple
  // consisting the begin address, end address, and a vector of
  // `DisassembledInstruction`s belonging to the basic block.
  // TODO(virajbshah): Remove/refactor this in favor of having a single library
  // for extracting basic blocks i.e. merge with `extract_bbs_from_obj`.
  absl::StatusOr<std::vector<std::vector<DisassembledInstruction>>>
  GetBlocksFromELF(const llvm::object::ELFObjectFileBase* elf_object);

  // Extracts samples from the `perf.data`-file loaded using `LoadPerfData`,
  // usually obtained from `perf record`. Returns a {`sample_types`, `samples`}
  // pair. `sample_types` is a vector of sample type names, while `samples` is
  // a mapping between sample addresses and the corresponding sample values.
  // The ordering of the sample values matches the ordering of types in the
  // heading.
  absl::StatusOr<std::pair<std::vector<std::string>,
                           std::unordered_map<uint64_t, std::vector<int>>>>
  GetSamples();

  // Extracts start and end pairs, as well as latencies in cycles, of sequences
  // of straight-run code from branch stacks.
  // LBR data is extracted from the `perf.data`-like file loaded using
  // `LoadPerfData`.
  absl::StatusOr<std::vector<
      std::pair<std::vector<DisassembledInstruction>, std::vector<uint32_t>>>>
  GetLBRBlocksWithLatency(const llvm::object::ELFObjectFileBase* elf_object);

  BHiveImporter importer_;
  quipper::PerfReader perf_reader_;
  quipper::PerfParser perf_parser_;
  quipper::PerfDataProto::MMapEvent main_mapping_;
};

template <class ELFT>
absl::StatusOr<llvm::object::Elf_Phdr_Impl<ELFT>>
AnnotatingImporter::GetMainProgramHeader(
    const llvm::object::ELFObjectFile<ELFT>* elf_object) {
  llvm::object::Elf_Phdr_Impl<ELFT> main_header;
  bool found_main_header = false;
  auto program_headers = elf_object->getELFFile().program_headers();
  if (llvm::Error error = program_headers.takeError()) {
    return LlvmErrorToStatus(std::move(error));
  }
  for (const auto& program_header : *program_headers) {
    if (program_header.p_type == llvm::ELF::PT_LOAD &&
        program_header.p_flags & llvm::ELF::PF_R &&
        program_header.p_flags & llvm::ELF::PF_X) {
      if (found_main_header) {
        return absl::InvalidArgumentError(
            "The given object has multiple executable segments. This is "
            "currently not supported.");
      }
      main_header = program_header;
      found_main_header = true;
    }
  }

  return main_header;
}

}  // namespace gematria

#endif  // THIRD_PARTY_GEMATRIA_GEMATRIA_DATASETS_ANNOTATING_IMPORTER_H_
