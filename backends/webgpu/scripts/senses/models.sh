#!/bin/bash
# dl.sh: mirror the in-page sense models into work/models/hf/<repo>/ (HF layout, so transformers.js loads them with allowRemoteModels=false).
M="${1:?usage: models.sh <models dir served at /models/>}"; mkdir -p "$M/hf" "$M/det-web"; cd "$M/hf" || exit 1
get() { r=$1; shift; for f in "$@"; do mkdir -p "$r/$(dirname $f)"; [ -s "$r/$f" ] || curl -sfL --retry 3 -o "$r/$f" "https://huggingface.co/$r/resolve/main/$f" || echo "MISS $r/$f"; done; }
J="config.json generation_config.json preprocessor_config.json tokenizer.json tokenizer_config.json special_tokens_map.json"
get onnx-community/whisper-tiny.en $J added_tokens.json normalizer.json onnx/encoder_model.onnx onnx/decoder_model_merged_q4.onnx onnx/encoder_model_quantized.onnx onnx/decoder_model_merged_quantized.onnx
get onnx-community/whisper-base.en $J added_tokens.json normalizer.json onnx/encoder_model.onnx onnx/decoder_model_merged_q4.onnx onnx/encoder_model_quantized.onnx onnx/decoder_model_merged_quantized.onnx
get onnx-community/Kokoro-82M-v1.0-ONNX config.json tokenizer.json tokenizer_config.json onnx/model.onnx onnx/model_quantized.onnx voices/af_bella.bin voices/af_heart.bin
get HuggingFaceTB/SmolVLM-256M-Instruct $J processor_config.json chat_template.json chat_template.jinja onnx/vision_encoder.onnx onnx/embed_tokens.onnx onnx/decoder_model_merged_q4.onnx onnx/vision_encoder_quantized.onnx onnx/embed_tokens_quantized.onnx onnx/decoder_model_merged_quantized.onnx
echo DONE; du -sh *
# D-FINE for the browser: the exported ONNX (mac-reflex-coreml onnx_ref.py) with its float64 position embedding folded.
PY="${PY:-python3}"   # needs onnx + onnxruntime (full build)
for n in dfine_n dfine_s; do [ -s "$M/det-web/$n.onnx" ] || "$PY" "$(dirname "$0")/fold_dfine.py" "${DFINE_ONNX_DIR:?set DFINE_ONNX_DIR to the exported D-FINE ONNX dir}/$n.onnx" "$M/det-web/$n.onnx"; done
