// glue_out.h: G4 F1 readout rows. output.weight rows of the trunk GGUF, dequantized
// (Q8_0, Q6_K, F32), for the served path's "readout" (dual.c) when no clef head is
// resident (LLB_SYS1_NOHEAD=1). The width is the tensor's own (ne[0], at most 8192; 4096 on the
// 9B trunks); a tied-embedding trunk without output.weight reads token_embd.weight.
#pragma once
void glue_out_open(const char *gguf);
void glue_out_row(int id, float *out);   // out holds glue_out_width() floats
int glue_out_width(void);                // 0 before glue_out_open
// G24 F4: tensor_table_sha256 of a GGUF (relayout_gguf.py's definition; threaded, reads
// every tensor once) and a whole-file sha256. 0 ok, -1 on error.
int glue_table_sha(const char *gguf, char out[65]);
int glue_file_sha(const char *path, char out[65]);
// The same through the stat-keyed cache DIR/tabsha (NULL/"" = no cache); *hit = 1 on a hit.
int glue_table_sha_cached(const char *gguf, const char *dir, char out[65], int *hit);
