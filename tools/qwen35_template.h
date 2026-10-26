// tools/qwen35_template.h — the Qwen3.5 chat template (text roles), rendered in
// our code. Mirrors the GGUF's Jinja for the no-tools/no-vision case:
//   system   : <|im_start|>system\n{trim(content)}<|im_end|>\n
//   user     : <|im_start|>user\n{trim(content)}<|im_end|>\n
//   assistant: <|im_start|>assistant\n ... <|im_end|>\n   (post-query turns get
//              the <think> reasoning block; see below)
//   gen      : <|im_start|>assistant\n  + think piece
// think mode for the generation prompt: 0 none (oracle-compatible ChatML),
// 1 <think>\n  (enable_thinking=true), 2 <think>\n\n</think>\n\n (false).
#ifndef QWEN35_TEMPLATE_H
#define QWEN35_TEMPLATE_H

typedef struct { const char *role; const char *content; } Q35Msg;

// Returns bytes written (excluding NUL); if out_cap is 0/too small, returns the
// needed size so the caller can grow.
int q35_chat_render(const Q35Msg *msgs, int n, int add_generation_prompt, int think_mode,
                    char *out, int out_cap);

#endif
