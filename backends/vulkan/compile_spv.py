#!/usr/bin/env python3
"""Compile GLSL 4.60 compute shaders to SPIR-V C arrays using libshaderc_shared.so.1.

Zero external CLI tools required on Fedora/Ubuntu/Arch systems shipping libshaderc.
Supports both Tier 0 portable SPIR-V 1.3 (Vulkan 1.2) and Tier 1 SPIR-V 1.6 (Vulkan 1.3
with subgroups, integer dot product, float16/int8/bfloat16, and VK_KHR_cooperative_matrix).
"""

import argparse
import ctypes
import os
import struct
import sys

SHADERC_TARGET_ENV_VULKAN = 0
SHADERC_ENV_VULKAN_1_2 = (1 << 22) | (2 << 12)
SHADERC_ENV_VULKAN_1_3 = (1 << 22) | (3 << 12)
SHADERC_SPIRV_1_3 = 0x00010300
SHADERC_SPIRV_1_6 = 0x00010600
SHADERC_OPT_PERFORMANCE = 2
SHADERC_COMPUTE_SHADER = 2


def load_shaderc():
    for name in (
        "libshaderc_shared.so.1",
        "libshaderc_shared.so",
        "/usr/lib64/libshaderc_shared.so.1",
        "/usr/lib/x86_64-linux-gnu/libshaderc_shared.so.1",
    ):
        try:
            lib = ctypes.CDLL(name)
            break
        except OSError:
            continue
    else:
        raise RuntimeError("libshaderc_shared.so.1 not found")

    lib.shaderc_compiler_initialize.restype = ctypes.c_void_p
    lib.shaderc_compiler_release.argtypes = [ctypes.c_void_p]
    lib.shaderc_compile_options_initialize.restype = ctypes.c_void_p
    lib.shaderc_compile_options_release.argtypes = [ctypes.c_void_p]
    lib.shaderc_compile_options_set_target_env.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_uint32,
    ]
    lib.shaderc_compile_options_set_target_spirv.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
    ]
    lib.shaderc_compile_options_set_optimization_level.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    lib.shaderc_compile_options_add_macro_definition.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    lib.shaderc_compile_into_spv.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_void_p,
    ]
    lib.shaderc_compile_into_spv.restype = ctypes.c_void_p
    lib.shaderc_result_release.argtypes = [ctypes.c_void_p]
    lib.shaderc_result_get_compilation_status.argtypes = [ctypes.c_void_p]
    lib.shaderc_result_get_compilation_status.restype = ctypes.c_int
    lib.shaderc_result_get_length.argtypes = [ctypes.c_void_p]
    lib.shaderc_result_get_length.restype = ctypes.c_size_t
    lib.shaderc_result_get_bytes.argtypes = [ctypes.c_void_p]
    lib.shaderc_result_get_bytes.restype = ctypes.c_void_p
    lib.shaderc_result_get_error_message.argtypes = [ctypes.c_void_p]
    lib.shaderc_result_get_error_message.restype = ctypes.c_char_p
    return lib


def compile_glsl(lib, compiler, src_bytes, filename, env_ver, spv_ver, defines=None):
    opts = lib.shaderc_compile_options_initialize()
    lib.shaderc_compile_options_set_target_env(opts, SHADERC_TARGET_ENV_VULKAN, env_ver)
    lib.shaderc_compile_options_set_target_spirv(opts, spv_ver)
    lib.shaderc_compile_options_set_optimization_level(opts, SHADERC_OPT_PERFORMANCE)
    if defines:
        for k, v in defines.items():
            kb = k.encode()
            vb = str(v).encode()
            lib.shaderc_compile_options_add_macro_definition(opts, kb, len(kb), vb, len(vb))
    res = lib.shaderc_compile_into_spv(
        compiler, src_bytes, len(src_bytes), SHADERC_COMPUTE_SHADER, filename.encode(), b"main", opts
    )
    status = lib.shaderc_result_get_compilation_status(res)
    if status != 0:
        err = lib.shaderc_result_get_error_message(res).decode("utf-8", errors="replace")
        lib.shaderc_result_release(res)
        lib.shaderc_compile_options_release(opts)
        raise RuntimeError(f"GLSL compile failed ({filename}, status={status}):\n{err}")
    nbytes = lib.shaderc_result_get_length(res)
    ptr = lib.shaderc_result_get_bytes(res)
    raw = ctypes.string_at(ptr, nbytes)
    lib.shaderc_result_release(res)
    lib.shaderc_compile_options_release(opts)
    words = struct.unpack(f"<{nbytes // 4}I", raw)
    return words


def format_c_array(name, words):
    lines = [f"static const uint32_t {name}[{len(words)}] = {{"]
    for i in range(0, len(words), 8):
        chunk = ", ".join(f"0x{w:08x}u" for w in words[i : i + 8])
        lines.append(f"  {chunk},")
    lines.append("};")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output C header path")
    ap.add_argument("--prefix", default="vk_spv_", help="C symbol prefix")
    ap.add_argument("--vulkan", choices=("1.2", "1.3"), default="1.3")
    ap.add_argument("shaders", nargs="+", help="Pairs of name=path.comp[:DEF=VAL,...]")
    args = ap.parse_args()

    env_ver = SHADERC_ENV_VULKAN_1_2 if args.vulkan == "1.2" else SHADERC_ENV_VULKAN_1_3
    spv_ver = SHADERC_SPIRV_1_3 if args.vulkan == "1.2" else SHADERC_SPIRV_1_6

    lib = load_shaderc()
    comp = lib.shaderc_compiler_initialize()
    try:
        out_parts = [
            f"// Auto-generated SPIR-V {args.vulkan} header via backends/vulkan/compile_spv.py",
            "#pragma once",
            "#include <stdint.h>",
            "",
        ]
        total_bytes = 0
        for spec in args.shaders:
            name, rhs = spec.split("=", 1)
            parts = rhs.split(":")
            path = parts[0]
            defs = {}
            for d in parts[1:]:
                if "=" in d:
                    dk, dv = d.split("=", 1)
                    defs[dk] = dv
                elif d:
                    defs[d] = "1"
            with open(path, "rb") as f:
                src = f.read()
            words = compile_glsl(lib, comp, src, os.path.basename(path), env_ver, spv_ver, defs)
            out_parts.append(format_c_array(f"{args.prefix}{name}", words))
            out_parts.append("")
            total_bytes += len(words) * 4
            print(f"  [spv] {args.prefix}{name}: {len(words) * 4} bytes ({len(words)} words)")
        with open(args.out, "w") as f:
            f.write("\n".join(out_parts))
        print(f"Wrote {args.out} ({total_bytes} total SPIR-V bytes)")
    finally:
        lib.shaderc_compiler_release(comp)


if __name__ == "__main__":
    main()
