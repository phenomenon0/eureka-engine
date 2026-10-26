// tools/qwen35_engine.cu — CUDA build of the Qwen3.5 hybrid engine.
// The whole engine lives in qwen35_engine.c; compiled by nvcc it enables the
// device gemv (--gpu), compiled by gcc it is the CPU twin.
#include "qwen35_engine.c"
