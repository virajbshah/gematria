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

#include "gematria/datasets/annotating_importer.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/hash/hash.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "gematria/datasets/bhive_importer.h"
#include "gematria/llvm/canonicalizer.h"
#include "gematria/llvm/disassembler.h"
#include "gematria/llvm/llvm_to_absl.h"
#include "gematria/proto/basic_block.pb.h"
#include "gematria/proto/throughput.pb.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ELFTypes.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"
#include "quipper/perf_data.pb.h"
#include "quipper/perf_parser.h"
#include "quipper/perf_reader.h"

namespace gematria {
namespace {

#if __has_include(<sys/mman.h>)
#include <sys/mman.h>
#else
// Memory mapping protection flag bits from `sys/mman.h`.
constexpr int PROT_READ = 0x1;
constexpr int PROT_WRITE = 0x2;
constexpr int PROT_EXEC = 0x4;
#endif

constexpr int kMaxBlockSizeBytes = 65536;

llvm::StringRef GetBasenameFromPath(const llvm::StringRef path) {
  int idx = path.find_last_of('/');
  if (idx == llvm::StringRef::npos) {
    return path;
  }
  return path.substr(idx + 1);
}

// Returns a pointer inside `*binary` casted down to an ELF object. The pointer
// is only valid for as long as `binary` is.
absl::StatusOr<llvm::object::ELFObjectFileBase const *> GetELFObjectFromBinary(
    const llvm::object::Binary *binary) {
  if (!binary->isObject()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("The given binary (%s) is not an object.",
                        std::string(binary->getFileName())));
  }
  const auto *object = llvm::cast<llvm::object::ObjectFile>(binary);
  if (object == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Could not cast the binary (%s) to an ObjectFile.",
                        std::string(binary->getFileName())));
  }

  // Make sure the object is an ELF file.
  if (!object->isELF()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("The given object (%s) is not in ELF format.",
                        std::string(binary->getFileName())));
  }
  const auto *elf_object =
      llvm::dyn_cast<llvm::object::ELFObjectFileBase>(object);
  if (elf_object == nullptr) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Could not cast the object (%s) to an ELFObjectFileBase.",
        std::string(binary->getFileName())));
  }

  return elf_object;
}

// Validates and casts `elf_object` down to a typed ELF object.
absl::StatusOr<const llvm::object::ELF64LEObjectFile *> GetTypedELFObject(
    const llvm::object::ELFObjectFileBase *elf_object) {
  // TODO(vbshah): Consider making it possible to use other ELFTs rather than
  // only ELF64LE since only the implementation of GetMainProgramHeader
  // differs between different ELFTs.
  if (!elf_object->is64Bit() || !elf_object->isLittleEndian()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("The given object (%s) is not in ELF64LE format.",
                        elf_object->getFileName()));
  }
  const auto *typed_elf_object =
      llvm::dyn_cast<llvm::object::ELF64LEObjectFile>(elf_object);
  if (typed_elf_object == nullptr) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Could not cast the ELF object (%s) to an ELF64LEObjectFileBase.",
        elf_object->getFileName()));
  }
  return typed_elf_object;
}

// Returns the program header corresponding to the main executable section.
template <class ELFT>
absl::StatusOr<llvm::object::Elf_Phdr_Impl<ELFT>> GetMainProgramHeader(
    const llvm::object::ELFObjectFile<ELFT> *elf_object) {
  llvm::object::Elf_Phdr_Impl<ELFT> main_header;
  bool found_main_header = false;
  auto program_headers = elf_object->getELFFile().program_headers();
  if (llvm::Error error = program_headers.takeError()) {
    return LlvmErrorToStatus(std::move(error));
  }
  for (const auto &program_header : *program_headers) {
    if (program_header.p_type == llvm::ELF::PT_LOAD &&
        program_header.p_flags & llvm::ELF::PF_R &&
        program_header.p_flags & llvm::ELF::PF_X) {
      if (found_main_header) {
        return absl::InvalidArgumentError(
            "The given object has multiple executable segments. This is"
            " currently not supported.");
      }
      main_header = program_header;
      found_main_header = true;
    }
  }

  return main_header;
}

}  // namespace

absl::StatusOr<PerfEventImporter> PerfEventImporter::Create(
    const std::string_view perf_data_path) {
  PerfEventImporter importer;

  // Read and parse the `perf.data`-like file into something more tractable.
  if (!importer.perf_reader_->ReadFile(std::string(perf_data_path))) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "The given `perf.data`-like file (%s) could not be read.",
        perf_data_path));
  }

  quipper::PerfParser perf_parser(
      importer.perf_reader_.get(),
      quipper::PerfParserOptions{.do_remap = true,
                                 .discard_unused_events = true,
                                 .sort_events_by_time = false,
                                 .combine_mappings = true});
  if (!perf_parser.ParseRawEvents()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "The given `perf.data`-like file (%s) could not be parsed.",
        perf_data_path));
  }

  return importer;
}

absl::StatusOr<const quipper::PerfDataProto::MMapEvent *>
PerfEventImporter::GetMMapEventForBinary(
    const std::string_view binary_basename) {
  // TODO(vbshah): There may be multiple mappings corresponding to the
  // profiled binary. Record and match samples from all of them instead of
  // assuming there is only one and returning after finding it.
  for (const auto &event : perf_reader_->events()) {
    if (event.has_mmap_event() &&
        GetBasenameFromPath(event.mmap_event().filename()) ==
            llvm::StringRef(binary_basename) &&
        event.mmap_event().prot() & PROT_READ &&
        event.mmap_event().prot() & PROT_EXEC) {
      return &event.mmap_event();
    }
  }

  return absl::InvalidArgumentError(absl::StrFormat(
      "The given `perf.data`-like file does not have a mapping corresponding"
      " to the given object (%s).",
      binary_basename));
}

absl::StatusOr<PerfEventImporter::SampleTable>
PerfEventImporter::GetPerfSampleTable(const std::string_view binary_basename) {
  const quipper::PerfDataProto &perf_data = perf_reader_->proto();
  SampleTable table;

  auto mapping = GetMMapEventForBinary(binary_basename);
  if (!mapping.ok()) {
    return mapping.status();
  }
  const uint64_t mmap_begin_addr = (*mapping)->start();
  const uint64_t mmap_end_addr = mmap_begin_addr + (*mapping)->len();

  // Extract event type information,
  const int num_sample_types = perf_data.event_types_size();
  table.sample_type_names = std::vector<std::string>(num_sample_types);
  std::unordered_map<uint64_t, int> event_code_to_idx;
  for (int sample_type_idx = 0; sample_type_idx < num_sample_types;
       ++sample_type_idx) {
    const auto &event_type = perf_data.event_types()[sample_type_idx];
    table.sample_type_names[sample_type_idx] = event_type.name();
    event_code_to_idx[event_type.id()] = sample_type_idx;
  }
  std::unordered_map<uint64_t, uint64_t> event_id_to_code;
  for (const auto &event_type : perf_data.file_attrs()) {
    // Mask out bits identifying the PMU and not the event.
    uint64_t event_code = event_type.attr().config() & 0xffff;
    for (uint64_t event_id : event_type.ids()) {
      event_id_to_code[event_id] = event_code;
    }
  }

  // If the profile has multiple event types, lookups are needed to find the
  // event type corresponding to a sample. In the other case, this is neither
  // required nor possible - since samples are not associated with IDs for
  // lookup in the proto.
  const bool has_multiple_sample_types = num_sample_types > 1;

  // Process sample events.
  for (const auto &event : perf_data.events()) {
    // Filter out non-sample events.
    if (!event.has_sample_event()) {
      continue;
    }

    // Filter out sample events from outside the profiled binary.
    if (!event.sample_event().has_pid() ||
        !(event.sample_event().pid() == (*mapping)->pid())) {
      continue;
    }
    uint64_t sample_ip = event.sample_event().ip();
    if (sample_ip < mmap_begin_addr || sample_ip >= mmap_end_addr) {
      continue;
    }

    std::vector<int> &samples_at_same_addr = table.samples_at_ip[sample_ip];
    if (samples_at_same_addr.empty()) {
      samples_at_same_addr.resize(num_sample_types);
    }
    int event_idx = 0;
    if (has_multiple_sample_types) {
      event_idx =
          event_code_to_idx[event_id_to_code[event.sample_event().id()]];
    }
    samples_at_same_addr[event_idx] += 1;
  }

  return table;
}

absl::StatusOr<LBRImporter> LBRImporter::Create(
    const std::string_view binary_path, const std::string_view perf_data_path,
    const Canonicalizer *canonicalizer) {
  // Obtain a reference to the underlying object.
  llvm::Expected<llvm::object::OwningBinary<llvm::object::Binary>>
      owning_binary = llvm::object::createBinary(binary_path);
  if (llvm::Error error = owning_binary.takeError()) {
    return LlvmErrorToStatus(std::move(error));
  }
  const auto elf_object = GetELFObjectFromBinary(owning_binary->getBinary());
  if (!elf_object.ok()) {
    return elf_object.status();
  }

  absl::StatusOr<PerfEventImporter> perf_event_importer =
      PerfEventImporter::Create(perf_data_path);
  if (!perf_event_importer.ok()) {
    return perf_event_importer.status();
  }
  BHiveImporter bhive_importer(canonicalizer);

  return LBRImporter(std::move(*owning_binary), *elf_object,
                     *std::move(perf_event_importer),
                     std::move(bhive_importer));
}

absl::StatusOr<std::function<absl::StatusOr<LBRImporter::LBRTraceData>()>>
LBRImporter::GetLBRTraceData() {
  const auto typed_elf_object = GetTypedELFObject(elf_object_);
  if (!typed_elf_object.ok()) {
    return typed_elf_object.status();
  }
  const auto main_header = GetMainProgramHeader(*typed_elf_object);
  if (!main_header.ok()) {
    return main_header.status();
  }

  const auto mapping = perf_event_importer_.GetMMapEventForBinary(
      GetBasenameFromPath(elf_object_->getFileName()));
  if (!mapping.ok()) {
    return mapping.status();
  }

  auto branch_stack_event_generator = perf_event_importer_.GetPerfEvents(
      elf_object_->getFileName(), -1,
      [mapping =
           *mapping](const quipper::PerfDataProto::PerfEvent &event) -> bool {
        return event.has_sample_event() &&
               event.sample_event().branch_stack_size() > 0 &&
               event.sample_event().has_pid() &&
               event.sample_event().pid() == mapping->pid();
      });

  auto lbr_trace_data_generator =
      [mapping = *mapping,
       branch_stack_event_generator = *std::move(
           branch_stack_event_generator)]() -> absl::StatusOr<LBRTraceData> {
    const uint64_t mmap_begin_addr = mapping->start();
    const uint64_t mmap_end_addr = mmap_begin_addr + mapping->len();

    while (true) {
      auto event = branch_stack_event_generator();
      if (absl::IsOutOfRange(event.status())) {
        return absl::OutOfRangeError(
            "Iteration complete, all trace data has been returned.");
      }
      if (!event.ok()) {
        return event.status();
      }
      const auto &branch_stack = event->sample_event().branch_stack();
      LBRTraceData trace_data(branch_stack.size() - 1);
      bool is_valid = true;
      for (int branch_idx = branch_stack.size() - 2; branch_idx >= 0;
           --branch_idx) {
        const auto &branch_entry = branch_stack[branch_idx + 1];
        const auto &next_branch_entry = branch_stack[branch_idx];

        const uint64_t block_begin = branch_entry.to_ip();
        const uint64_t block_end = next_branch_entry.from_ip();

        // Simple validity checks: the block must start before it ends, cannot
        // be larger than some threshold, and must belong to the binary we are
        // importing from.
        if (block_begin >= block_end ||
            block_end - block_begin > kMaxBlockSizeBytes ||
            block_begin < mmap_begin_addr || mmap_end_addr < block_end) {
          is_valid = false;
          break;
        }

        trace_data[branch_idx] = {
            .address_range = {block_begin, block_end},
            .latency = next_branch_entry.cycles(),
        };
      }
      if (is_valid) {
        return trace_data;
      }
    }
  };

  return lbr_trace_data_generator;
}

absl::StatusOr<
    std::function<absl::StatusOr<BasicBlockWithThroughputListProto>()>>
LBRImporter::GetLBRTraceProtos(const std::string_view source_name) {
  const auto mapping = perf_event_importer_.GetMMapEventForBinary(
      GetBasenameFromPath(elf_object_->getFileName()));
  if (!mapping.ok()) {
    return mapping.status();
  }

  auto lbr_trace_data_generator = GetLBRTraceData();
  if (!lbr_trace_data_generator.ok()) {
    return lbr_trace_data_generator.status();
  }

  std::unordered_map<AddressRange, BasicBlockProto, absl::Hash<AddressRange>>
      cache;
  auto trace_proto_generator =
      [this, mapping = *mapping, source_name, cache = std::move(cache),
       lbr_trace_data_generator =
           *std::move(lbr_trace_data_generator)]() mutable
      -> absl::StatusOr<BasicBlockWithThroughputListProto> {
    while (true) {
      const auto trace_data = lbr_trace_data_generator();
      if (absl::IsOutOfRange(trace_data.status())) {
        break;
      }
      if (!trace_data.ok()) {
        return trace_data.status();
      }

      // Resolve the contents of each block in the trace.
      BasicBlockWithThroughputListProto trace_proto;
      for (const LBRBlockData &block : *trace_data) {
        if (cache.find(block.address_range) == cache.cend()) {
          const auto [block_begin, block_end] = block.address_range;
          absl::StatusOr<std::vector<DisassembledInstruction>> instrs =
              GetInstructionsInAddressRange(block.address_range,
                                            mapping->pgoff());
          if (!instrs.ok()) {
            // TODO(vbshah): Make the importer so something better than simply
            // exiting upon encountering something unexpected.
            return instrs.status();
          }

          // TODO(vbshah): Consider dropping the added tail branch instruction
          // from the last block of each trace.
          const absl::StatusOr<DisassembledInstruction> tail_instr =
              GetInstructionAtAddress(block_end, mapping->pgoff());
          if (!tail_instr.ok()) {
            return tail_instr.status();
          }
          instrs->push_back(*std::move(tail_instr));

          cache[block.address_range] =
              bhive_importer_.BasicBlockProtoFromInstructions(*instrs);
        }
        if (cache[block.address_range].machine_instructions_size() == 0) {
          continue;
        }

        BasicBlockWithThroughputProto &block_with_throughput =
            *trace_proto.add_basic_blocks();
        *block_with_throughput.mutable_basic_block() =
            cache[block.address_range];
        ThroughputWithSourceProto &throughput =
            *block_with_throughput.add_inverse_throughputs();
        throughput.set_source(source_name);
        throughput.add_inverse_throughput_cycles(block.latency);
      }

      return trace_proto;
    }
    return absl::OutOfRangeError(
        "Iteration complete, all trace protos have been returned.");
  };

  return trace_proto_generator;
}

absl::StatusOr<std::vector<DisassembledInstruction>>
LBRImporter::GetInstructionsInAddressRange(AddressRange address_range,
                                           uint64_t offset) {
  llvm::StringRef binary_buf = elf_object_->getData();
  auto [range_begin, range_end] = address_range;

  llvm::ArrayRef<uint8_t> machine_code(
      reinterpret_cast<const uint8_t *>(
          binary_buf.slice(range_begin + offset, range_end + offset).data()),
      range_end - range_begin);
  absl::StatusOr<std::vector<DisassembledInstruction>> instructions =
      bhive_importer_.DisassembledInstructionsFromMachineCode(machine_code,
                                                              range_begin);
  if (!instructions.ok()) {
    return instructions.status();
  }
  return instructions;
}

absl::StatusOr<DisassembledInstruction> LBRImporter::GetInstructionAtAddress(
    uint64_t address, uint64_t offset) {
  llvm::StringRef binary_buf = elf_object_->getData();

  constexpr int kMaxInstructionSize = 15;
  llvm::ArrayRef<uint8_t> machine_code(
      reinterpret_cast<const uint8_t *>(
          binary_buf
              .slice(address + offset, address + offset + kMaxInstructionSize)
              .data()),
      kMaxInstructionSize);
  absl::StatusOr<DisassembledInstruction> instruction =
      bhive_importer_.SingleDisassembledInstructionFromMachineCode(machine_code,
                                                                   address);
  if (!instruction.ok()) {
    return instruction.status();
  }
  return instruction;
}

// absl::StatusOr<std::vector<std::vector<DisassembledInstruction>>>
// AnnotatingImporter::GetBlocksFromELF(
//     const llvm::object::ELFObjectFileBase *elf_object) {
//   // Read the associated `BBAddrMap` and `PGOAnalysisData`.
//   std::vector<llvm::object::PGOAnalysisMap> pgo_analyses;
//   llvm::Expected<std::vector<llvm::object::BBAddrMap>> bb_addr_map =
//       elf_object->readBBAddrMap(
//           /* TextSectionIndex = */ std::nullopt,
//           /* PGOAnalyses = */ &pgo_analyses);
//   if (llvm::Error error = bb_addr_map.takeError()) {
//     return LlvmErrorToStatus(std::move(error));
//   }

//   // TODO(vbshah): Consider making it possible to use other ELFTs rather than
//   // only ELF64LE since only the implementation of GetMainProgramHeader
//   // differs between different ELFTs.
//   if (!elf_object->is64Bit() || !elf_object->isLittleEndian()) {
//     return absl::InvalidArgumentError(
//         absl::StrFormat("The given object (%s) is not in ELF64LE format.",
//                         elf_object->getFileName()));
//   }
//   const auto *typed_elf_object =
//       llvm::dyn_cast<llvm::object::ELF64LEObjectFile>(elf_object);
//   if (typed_elf_object == nullptr) {
//     return absl::InvalidArgumentError(absl::StrFormat(
//         "Could not cast the ELF object (%s) to an ELF64LEObjectFileBase.",
//         elf_object->getFileName()));
//   }

//   const auto main_header = GetMainProgramHeader(typed_elf_object);
//   if (!main_header.ok()) {
//     return main_header.status();
//   }

//   // Populate a vector with all of the basic blocks.
//   std::vector<std::vector<DisassembledInstruction>> basic_blocks;
//   for (const llvm::object::BBAddrMap &map : bb_addr_map.get()) {
//     uint64_t function_addr = map.getFunctionAddress();
//     for (const llvm::object::BBAddrMap::BBRangeEntry &bb_range :
//          map.getBBRanges()) {
//       for (const llvm::object::BBAddrMap::BBEntry &bb : bb_range.BBEntries) {
//         uint64_t begin_idx = function_addr + bb.Offset;
//         uint64_t end_idx = begin_idx + bb.Size;
//         if (begin_idx == end_idx) {
//           continue;  // Skip any empty basic blocks.
//         }
//         const auto &basic_block = GetInstructionsInAddressRange(
//             elf_object, AddressRange(begin_idx, end_idx),
//             main_header->p_vaddr);
//         if (!basic_block.ok()) {
//           return basic_block.status();
//         }
//         basic_blocks.push_back(*basic_block);
//       }
//     }
//   }
//   return basic_blocks;
// }

// absl::StatusOr<
//     std::function<absl::StatusOr<BasicBlockWithThroughputListProto>()>>
// AnnotatingImporter::GetAnnotatedBasicBlockProtos(
//     std::string_view elf_file_name, std::string_view perf_data_file_name,
//     std::string_view source_name) {
//   std::cout << "Entered C++" << std::endl;
//   // Try to load the binary and cast it down to an ELF object.
//   absl::StatusOr<llvm::object::OwningBinary<llvm::object::Binary>>
//       owning_binary = LoadBinary(elf_file_name);
//   if (!owning_binary.ok()) {
//     return owning_binary.status();
//   }
//   const auto elf_object = GetELFFromBinary(owning_binary->getBinary());
//   if (!elf_object.ok()) {
//     return elf_object.status();
//   }

//   // Try to load the perf profile and locate its main mapping, i.e. the one
//   // corresponding to the executable load segment of the given object file.
//   absl::StatusOr<const quipper::PerfDataProto *> perf_data =
//       LoadPerfData(perf_data_file_name);
//   if (!perf_data.ok()) {
//     return perf_data.status();
//   }
//   auto main_mapping = GetMainMapping(*elf_object, *perf_data);
//   if (!main_mapping.ok()) {
//     return main_mapping.status();
//   }

//   std::cout << "Creating proto generator." << std::endl;
//   // Get the raw basic blocks, perf samples, and LBR data for annotation.
//   auto basic_block_generator = GetLBRBlocksWithLatency(
//       *elf_object, *perf_data, *main_mapping, source_name);
//   if (!basic_block_generator.ok()) {
//     return basic_block_generator.status();
//   }
//   std::cout << "Created proto generator." << std::endl;
//   auto sample_types_and_samples = GetSamples(*perf_data, *main_mapping);
//   if (!sample_types_and_samples.ok()) {
//     return sample_types_and_samples.status();
//   }

//   std::cout << "Creating anno generator." << std::endl;
//   // Annotate the blocks using samples.
//   auto annotated_basic_block_generator =
//       [=, basic_block_generator = *std::move(basic_block_generator),
//        sample_types_and_samples = *std::move(sample_types_and_samples)]()
//       -> absl::StatusOr<BasicBlockWithThroughputListProto> {
//     std::cout << "Called anno generator." << std::endl;
//     auto trace = basic_block_generator();
//     if (!trace.ok()) {
//       return trace.status();
//     }

//     const auto &[sample_types, samples] = sample_types_and_samples;
//     for (BasicBlockWithThroughputProto &block_with_throughput :
//          *trace->mutable_basic_blocks()) {
//       BasicBlockProto &block = *block_with_throughput.mutable_basic_block();

//       // Loop over and annotate individual instructions.
//       for (int instruction_idx = 0;
//            instruction_idx < block.machine_instructions_size();
//            ++instruction_idx) {
//         uint64_t instruction_addr =
//             block.machine_instructions()[instruction_idx].address();
//         if (!samples.count(instruction_addr)) continue;

//         const std::vector<int> &annotations = samples.at(instruction_addr);
//         auto &instruction_proto =
//             block.mutable_canonicalized_instructions()->at(instruction_idx);
//         for (int annotation_idx = 0; annotation_idx < annotations.size();
//              ++annotation_idx) {
//           if (annotations[annotation_idx]) {
//             *instruction_proto.add_instruction_annotations() =
//                 ProtoFromAnnotation(Annotation(
//                     /* name = */ sample_types.at(annotation_idx),
//                     /* value = */ annotations.at(annotation_idx)));
//           }
//         }
//       }
//     }
//     return trace;
//   };
//   std::cout << "Created anno generator." << std::endl;

//   return annotated_basic_block_generator;
// }

}  // namespace gematria
