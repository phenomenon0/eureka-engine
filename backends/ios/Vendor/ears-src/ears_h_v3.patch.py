import sys
p=sys.argv[1]; s=open(p).read()
old='const char *ears_version(void);'
new='''const char *ears_version(void);
// G38 (EARS v3): ids (as ears_transcribe gave them) -> UTF-8 text, the LLM's vocab only (no weights). Trailing EOG ids
// are dropped; special tokens are rendered (Qwen3-ASR answers "language <L><asr_text><text>"). Returns the byte length
// (out NUL-terminated), < 0 on error.
int ears_detokenize(const char *llm, const int32_t *ids, int n_ids, char *out, int cap, char *err /*512*/);'''
assert old in s and 'ears_detokenize' not in s; open(p,'w').write(s.replace(old,new,1))
