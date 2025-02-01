# Copyright 2023 Google Inc.
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
"""Contains portable rules for building and testing Python code."""

load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

def gematria_py_binary(name = None, visibility = [], **kwargs):
    if "//visibility:public" in visibility or "//visibility:private" in visibility:
        py_binary(name = name, visibility = visibility, **kwargs)
    else:
        py_binary(
            name = name,
            visibility = visibility + ["//tools:__subpackages__"],
            **kwargs
        )

def gematria_py_library(name = None, **kwargs):
    py_library(name = name, **kwargs)

def gematria_py_test(name = None, **kwargs):
    py_test(name = name, **kwargs)

def gematria_pybind_extension(name = None, **kwargs):
    pybind_extension(name = name, **kwargs)
