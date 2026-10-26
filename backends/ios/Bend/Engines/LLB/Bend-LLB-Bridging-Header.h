// The C API of LLBEngine.xcframework (engine-phone2 build/include/llb_engine.h, copied unmodified) and the engine
// tokenizer linked in the same library (engine tree tools/qwen35_tokenizer.h, copied unmodified).
#include "include/llb_engine.h"
#include "include/qwen35_tokenizer.h"
// G38: EARS v3 (Vendor/EARS.xcframework; Vendor/macos/ears-include for the Mac harness): the ASR tower, -voiceASR ears.
#if __has_include("ears.h")
#include "ears.h"
#endif
