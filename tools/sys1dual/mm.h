// mm.h: G5 — multimodal message content for llb-serve (/v1/chat/completions).
// Parses OpenAI content parts (text, image_url, image, input_audio), decodes and
// sniffs the media, enforces size caps, and renders the content exactly as
// flash-chat_template.jinja's render_content() does: each image becomes
// <|vision_start|><|image_pad|><|vision_end|> in place, text parts are verbatim.
// No engine code: the media bytes are handed to the engine (see docs/reference/multimodal-content.md,
// "engine media hook"), which encodes them and expands each <|image_pad|>.
#pragma once
#include "../clef3/json.h"
#include <stddef.h>

#define MM_IMAGE_TAG "<|vision_start|><|image_pad|><|vision_end|>"
#define MM_IMAGE_PAD "<|image_pad|>"
// EARS-CANON: the ear's tag = the Qwen3-ASR template's audio part. With --ear the server replaces it by the ear's
// transcript before the trunk sees the prompt (mm_replace_audio); inside the ear's own decoder prompt <|audio_pad|>
// expands to the tower's rows, one chunk per 8 s mel window (tools/ears/ears.cpp, the slim qwen3a chunker).
#define MM_AUDIO_TAG "<|audio_start|><|audio_pad|><|audio_end|>"

typedef enum { MM_IMAGE = 1, MM_AUDIO = 2 } mm_kind;

typedef struct {
    mm_kind kind;
    int msg, part;          // message index, content-part index (for error params)
    char mime[24];          // sniffed from the bytes (the bytes decide)
    char declared[64];      // data: URI media type, or input_audio.format
    char detail[8];         // image_url.detail: auto|low|high ("auto" if absent)
    char *url;              // http(s) URL, fetched later by mm_fetch (NULL for data:)
    unsigned char *bytes;   // decoded bytes (NULL until fetched for a URL)
    size_t n;
    unsigned w, h;          // image header dimensions (0 for audio)
    char sha256[65];
} mm_item;

typedef struct {
    size_t image_bytes;     // per image, decoded
    size_t audio_bytes;     // per audio clip, decoded
    size_t total_bytes;     // all media in one request, decoded
    int max_images, max_audio;
    unsigned max_side;      // per image side, from the header (decompression-bomb guard)
    long long max_pixels;   // per image w*h
    int allow_remote;       // http(s) image URLs (fetched by the server, pinned DNS, no redirects)
    int allow_private;      // let remote URLs resolve to loopback/private/link-local addresses
    int fetch_timeout_s;
    int images, audio;      // the engine accepts this kind (else: 400 after full validation)
} mm_caps;

typedef struct {
    mm_item *v;
    size_t n, cap, total;
    int images, audio;
} mm_list;

typedef struct {
    int status;             // HTTP status: 400, 413, 415
    char param[48];         // e.g. messages[1].content[2]
    char msg[256];
} mm_err;

void mm_caps_default(mm_caps *c);
void mm_list_free(mm_list *L);

// Render one message's content (string, null or part array) into out, appending its
// media to L. is_system: images/audio raise like the template does. 0 ok, -1 error.
int mm_render_content(const jv *content, int msg, int is_system, const mm_caps *c, mm_list *L, sb *out, mm_err *e);

// Fetch every URL item in L (caps re-checked on the fetched bytes). 0 ok, -1 error.
int mm_fetch_all(mm_list *L, const mm_caps *c, mm_err *e);

// After a successful render: refuse kinds the engine does not take. 0 ok, -1 error.
int mm_check_engine(const mm_list *L, const mm_caps *c, mm_err *e);

// Replace the i-th <|image_pad|> in s by n_rows[i] copies (the engine side of the
// contract; exported so the harness can check the expansion rule). 0 ok, -1 error.
int mm_expand_pads(const char *s, const unsigned *n_rows, size_t n_images, sb *out, mm_err *e);

// EARS-CANON: one input_audio object ({"data":B64,"format":F}, the chat part's inner object) as a media item of L,
// with the chat part's checks (format, sniff, caps). param names the field in errors. 0 ok, -1 error.
int mm_audio_item(const jv *input_audio, const char *param, const mm_caps *c, mm_list *L, mm_err *e);

// EARS-CANON: the chat side of the ear. Replace the i-th MM_AUDIO_TAG in s by text[i] (the ear's transcript of the
// i-th audio item), so the audio becomes the user turn's text. 0 ok, -1 on a count mismatch.
int mm_replace_audio(const char *s, const char *const *text, size_t n_audio, sb *out, mm_err *e);

// helpers (exported for the unit tests)
int mm_b64_decode(const char *s, size_t len, unsigned char **out, size_t *n); // strict RFC 4648; 0 ok
const char *mm_sniff_image(const unsigned char *b, size_t n, unsigned *w, unsigned *h);
const char *mm_sniff_audio(const unsigned char *b, size_t n);
void mm_sha256_hex(const unsigned char *b, size_t n, char out[65]);
