// ec_recipes.c — Photoreal Prompt Libraries, Recipes & Universal ComfyUI Importer
#define _POSIX_C_SOURCE 200809L
#include "ec_recipes.h"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

static void json_esc(const char *src, char *dst, size_t cap) {
  size_t j = 0;
  for (size_t i = 0; src && src[i] && j + 2 < cap; i++) {
    unsigned char c = (unsigned char)src[i];
    if (c == '"' || c == '\\') { dst[j++] = '\\'; dst[j++] = (char)c; }
    else if (c == '\n') { dst[j++] = '\\'; dst[j++] = 'n'; }
    else if (c == '\r') { dst[j++] = '\\'; dst[j++] = 'r'; }
    else if (c == '\t') { dst[j++] = '\\'; dst[j++] = 't'; }
    else if (c >= 0x20) dst[j++] = (char)c;
  }
  dst[j] = '\0';
}

static const ec_recipe_t BUILTIN_RECIPES[] = {
  // --- Photoreal Image & Editing (8) ---
  {
    "editorial-portrait-85mm", "Editorial Portrait — 85mm f/1.4", "Portrait", "image",
    "Candid editorial portrait of a weathered ceramic artist in a sunlit Kyoto workshop, natural window sidelight catching dust motes and fine skin texture, shot on Hasselblad X2D 100C, 85mm f/1.4, shallow depth of field, Kodak Portra 400 color science, authentic candid expression, photorealistic",
    "Hasselblad X2D · 85mm f/1.4 · Portra 400", "flux2_klein_4b_bf16", "", 0.8f, 768, 1024, 4, 1042, 1
  },
  {
    "tokyo-rain-automotive", "Automotive Night Rain — Tokyo 35mm", "Automotive", "image",
    "A cinematic 35mm photograph of a crimson Ferrari F40 parked on a rain-slicked neon Tokyo alley at midnight, crisp wet asphalt reflections of kanji signs, mist rising from warm engine vents, anamorphic lens flare, shot on Leica SL3 35mm f/1.4, Cinestill 800T grain, photorealistic, ultra detailed",
    "Leica SL3 · 35mm f/1.4 · Cinestill 800T", "flux2_klein_4b_bf16", "", 0.8f, 1024, 768, 4, 42, 1
  },
  {
    "architectural-brutalist", "Architectural Monograph — 24mm Tilt-Shift", "Architecture", "image",
    "Architectural photography of a serene concrete and cedar pavilion cantilevered over a misty Norwegian fjord at dawn, crisp rectilinear verticals, calm mirror water reflections, warm interior tungsten glow contrasting cool blue twilight, shot on Phase One IQ4 24mm tilt-shift f/8, magazine editorial",
    "Phase One IQ4 · 24mm Tilt-Shift f/8", "flux2_klein_4b_bf16", "", 0.8f, 1024, 768, 4, 2026, 1
  },
  {
    "luxury-watch-macro", "Luxury Product Macro — 100mm Studio", "Product", "image",
    "Commercial macro studio photograph of a brushed titanium mechanical chronograph watch resting on wet dark slate, dramatic sculpted rim lighting from overhead strip softbox, sapphire crystal anti-reflective glint, microscopic gear detail, shot on Canon R5 100mm macro f/5.6, focus stacked, photorealistic",
    "Canon R5 · 100mm Macro f/5.6 · Strip Softbox", "flux2_klein_4b_bf16", "", 0.8f, 768, 768, 4, 777, 1
  },
  {
    "leica-street-doc", "Street Documentary — Rainy Paris 50mm", "Street", "image",
    "Timeless street documentary photograph of a jazz saxophonist under a warm glowing Paris cafe awning during a light evening rain, wet cobblestones, subtle motion blur of passing umbrellas, Leica M11 50mm Summilux f/1.4, authentic film grain, candid human moment",
    "Leica M11 · 50mm Summilux f/1.4", "flux2_klein_4b_bf16", "", 0.8f, 1024, 768, 4, 314, 1
  },
  {
    "nordic-interior-sun", "Interior Design — Nordic Morning Light", "Interior", "image",
    "Editorial interior photograph of a minimalist Copenhagen living room bathed in crisp morning sunlight, oak herringbone floor, travertine coffee table with stoneware espresso cup, linen sofa, soft architectural shadows across limewash wall, shot on Fujifilm GFX 100 II 32mm f/5.6",
    "Fuji GFX 100 II · 32mm f/5.6 · Daylight", "flux2_klein_4b_bf16", "", 0.8f, 1024, 768, 4, 512, 1
  },
  {
    "scifi-cinema-still", "Cinema Sci-Fi Film Still — 65mm IMAX", "Cinema", "image",
    "Anamorphic 65mm film still of a lone astronaut walking across an obsidian volcanic plain under two pale crescent moons, subtle dust storm catching amber helmet visor lights, Denis Villeneuve cinematography, atmospheric scale, muted naturalistic palette, shot on ARRI Alexa 65",
    "ARRI Alexa 65 · Prime DNA 40mm · Naturalistic", "flux2_klein_4b_bf16", "", 0.8f, 1024, 576, 4, 2049, 1
  },
  {
    "typography-poster-swiss", "Swiss Editorial Poster — Bold Typography", "Design", "image",
    "Minimalist International Typographic Style architectural exhibition poster on textured heavyweight cotton paper, bold black Helvetica headline 'EUREKA 2026' aligned to crisp Swiss grid, burnt vermilion geometric accent circle, raking gallery spotlight, studio photograph",
    "Swiss Grid · Cotton Stock · Editorial Print", "flux2_klein_4b_bf16", "", 0.8f, 768, 1024, 4, 1968, 1
  },
  // --- Cinema Video (T2V / I2V / FLF2V / Loop / Vertical) (6) ---
  {
    "cinema-video-dolly", "Cinema Video — 35mm Orbital Dolly", "Video", "video",
    "Slow cinematic 35mm orbital camera dolly around a rain-beaded vintage bronze astrolabe glowing softly by candlelight inside an observatory, smooth temporal continuity, shallow focus pull, 24fps NVENC H.264 + stereo score",
    "Wan 2.2 TI2V-5B · Orbital Dolly · 16:9", "wan2.2_ti2v_5b_fp16", "", 0.0f, 832, 480, 20, 303, 1
  },
  {
    "iceland-basalt-drone", "Iceland Basalt Cliffs — Aerial FPV", "Video", "video",
    "Sweeping low-altitude aerial drone shot gliding along wet black volcanic basalt sea cliffs in Iceland at golden sunset, turquoise North Atlantic surf crashing in slow motion, sea mist catching warm amber light, 35mm cinema lens",
    "Wan 2.2 TI2V-5B · Aerial Glide · 832x480", "wan2.2_ti2v_5b_fp16", "", 0.0f, 832, 480, 20, 42, 1
  },
  {
    "tokyo-neon-i2v-rain", "Neon Alley Rain — Start-Frame I2V", "Video I2V", "video",
    "Subtle living motion from start frame: shimmering rain streaks falling across glowing Tokyo alley neon signs, steam drifting softly from street grate, wet pavement reflections rippling gently, slow cinematic push-in",
    "Wan 2.2 TI2V-5B · I2V Anchor · Slow Push", "wan2.2_ti2v_5b_fp16", "", 0.0f, 832, 480, 20, 808, 1
  },
  {
    "morph-flf2v-bridge", "Start -> End Morph Transition (FLF2V)", "Video FLF2V", "video",
    "Fluid continuous camera glide seamlessly transitioning from opening frame composition into closing frame geometry, physically coherent lighting evolution, smooth optical flow, zero jump cuts, cinema 35mm",
    "Wan 2.2 TI2V-5B · First+Last Frame FLF2V", "wan2.2_ti2v_5b_fp16", "", 0.0f, 832, 480, 20, 515, 1
  },
  {
    "seamless-ambient-loop", "Seamless Cinemagraph Loop — Tea Steam", "Video Loop", "video",
    "Hypnotic seamless loop of delicate white steam curling upward from a handcrafted black stoneware matcha bowl on a rain-misted cypress windowsill, locked-off tripod framing, calm Zen light",
    "Wan 2.2 TI2V-5B · Seamless Loop · Tripod", "wan2.2_ti2v_5b_fp16", "", 0.0f, 640, 640, 20, 108, 1
  },
  {
    "vertical-reel-fashion", "Vertical 9:16 Editorial Fashion Reel", "Video Vertical", "video",
    "Vertical 9:16 fashion film of silk trench coat billowing in golden hour wind on a brutalist Milan rooftop, medium telephoto 85mm tracking shot, natural film grain, editorial cinema",
    "Wan 2.2 TI2V-5B · 480x832 Vertical Reel", "wan2.2_ti2v_5b_fp16", "", 0.0f, 480, 832, 20, 916, 1
  },
  // --- 3D Assets (TRELLIS.2 1.3B Text->3D & Photo->3D) (4) ---
  {
    "hero-3d-prop", "3D Hero PBR Prop — Espresso Machine", "3D Asset", "3d",
    "Hand-crafted brushed brass and matte charcoal ceramic lever espresso machine with clean watertight silhouette, neutral studio turntable lighting, subtle chamfered edges",
    "TRELLIS.2 1.3B · PBR GLB + 3DGS PLY", "trellis2_ss_flow_1.3b_bf16", "", 0.0f, 512, 512, 12, 77, 1
  },
  {
    "ceramic-teapot-3d", "Sculptural Porcelain Teapot — 3D Asset", "3D Asset", "3d",
    "Sculptural matte-white porcelain teapot with graceful curved spout and bent smoked bamboo handle, clean silhouette, neutral studio turntable",
    "TRELLIS.2 1.3B · Matte Ceramic PBR", "trellis2_ss_flow_1.3b_bf16", "", 0.0f, 512, 512, 12, 777, 1
  },
  {
    "scifi-mecha-helmet-3d", "Hard-Surface Sci-Fi Pilot Helmet — 3D", "3D Game", "3d",
    "Hard-surface sci-fi orbital pilot helmet in weathered titanium and dark amber visor glass, recessed mechanical vents, symmetric industrial design",
    "TRELLIS.2 1.3B · Hard-Surface Game Prop", "trellis2_ss_flow_1.3b_bf16", "", 0.0f, 512, 512, 12, 2099, 1
  },
  {
    "midcentury-chair-3d", "Mid-Century Oak Lounge Chair — ArchViz", "3D Furniture", "3d",
    "Danish mid-century sculpted walnut lounge chair with saddle tan leather cushion, clean tapered legs, gallery product render",
    "TRELLIS.2 1.3B · ArchViz Furniture GLB", "trellis2_ss_flow_1.3b_bf16", "", 0.0f, 512, 512, 12, 1958, 1
  },
  // --- Music & Audio (ACE-Step 1.5 Turbo: 2/4/8 Stems, Vocals, Scores, Beats) (6) ---
  {
    "cinema-synth-stems", "Analog Cinema Score — 8-Stem Master", "Music", "music",
    "Warm analog Moog bassline, crisp acoustic brushed drums, lush Juno-106 pad harmonies, expressive Rhodes electric piano, subtle cello strings, 124 BPM in D minor, studio mastered 48kHz stereo",
    "124 BPM · D minor · 8-Stem (ACE-Step 1.5)", "acestep_v15_turbo_bf16", "", 0.0f, 0, 0, 8, 101, 1
  },
  {
    "kyoto-lofi-jazzhop", "Kyoto Rain Lo-Fi Jazzhop — Rhodes & Vinyl", "Music", "music",
    "Warm dusty lo-fi hip-hop instrumental, mellow Fender Rhodes seventh chords, upright walking bass, swung boom-bap acoustic snare with gentle rain ambience, intimate analog tape saturation, 88 BPM in F major",
    "88 BPM · F major · 4-Stem Tape Mix", "acestep_v15_turbo_bf16", "", 0.0f, 0, 0, 8, 880, 1
  },
  {
    "indie-vocal-anthem", "Indie Dream-Pop Song — Female Vocals", "Music Vocals", "music",
    "Shimmering indie dream-pop anthem with airy female lead vocals, jangly chorus electric guitar arpeggios, punchy live acoustic drums, warm analog synth pads, driving bassline, 118 BPM in A major, radio master",
    "118 BPM · A major · Vocal + Instrumental", "acestep_v15_turbo_bf16", "", 0.0f, 0, 0, 8, 404, 1
  },
  {
    "berlin-melodic-techno", "Berlin Melodic Techno — Warehouse 132 BPM", "Music Club", "music",
    "Hypnotic Berlin melodic techno track, deep rolling sub-bass, punchy 909 kick and crisp offbeat hi-hats, evolving Prophet-6 arpeggiator with cavernous reverb tail, tension build and drop, 132 BPM in C minor",
    "132 BPM · C minor · 8-Stem Club Mix", "acestep_v15_turbo_bf16", "", 0.0f, 0, 0, 8, 132, 1
  },
  {
    "orchestral-trailer-epic", "Orchestral Film Trailer — Brass & Taiko", "Music Score", "music",
    "Sweeping cinematic orchestral film score, resonant cellos and soaring violins, heroic French horns, thunderous taiko percussion, emotional piano motif building to soaring climax, 96 BPM in G minor, Hans Zimmer soundstage",
    "96 BPM · G minor · Full Orchestral", "acestep_v15_turbo_bf16", "", 0.0f, 0, 0, 8, 609, 1
  },
  {
    "afrobeats-amapiano-groove", "Afrobeats & Amapiano — Log Drum Groove", "Music Groove", "music",
    "Infectious modern Afrobeats Amapiano fusion, bouncy deep log drum bassline, silky electric guitar riffs, shakers and crisp rimshots, lush Rhodes chords, sunlit Lagos rooftop summer energy, 112 BPM in E minor",
    "112 BPM · E minor · 8-Stem Groove", "acestep_v15_turbo_bf16", "", 0.0f, 0, 0, 8, 112, 1
  }
};

static void load_user_recipes(ec_recipe_lib_t *lib) {
  if (!lib->store_path[0]) return;
  FILE *fp = fopen(lib->store_path, "rb");
  if (!fp) return;
  fseek(fp, 0, SEEK_END);
  long sz = ftell(fp);
  fseek(fp, 0, SEEK_SET);
  if (sz <= 0 || sz > 512 * 1024) { fclose(fp); return; }
  char *buf = (char *)malloc((size_t)sz + 1);
  if (!buf || fread(buf, 1, (size_t)sz, fp) != (size_t)sz) { free(buf); fclose(fp); return; }
  buf[sz] = '\0';
  fclose(fp);

  const char *p = buf;
  while ((p = strchr(p, '{')) != NULL && lib->count < EC_MAX_RECIPES) {
    const char *end = strchr(p, '}');
    if (!end) break;
    ec_recipe_t r;
    memset(&r, 0, sizeof r);
    r.width = 512; r.height = 512; r.steps = 4; r.lora_strength = 0.8f; r.builtin = 0;

    #define GRAB_S(key, dst) do { \
      const char *k = strstr(p, "\"" key "\""); \
      if (k && k < end) { \
        k = strchr(k, ':'); if (k && k < end) { \
          while (k < end && *k != '"') k++; \
          if (k < end && *k == '"') { \
            k++; size_t j = 0; \
            while (k < end && *k != '"' && j + 1 < sizeof(dst)) { \
              if (*k == '\\' && k + 1 < end) k++; \
              (dst)[j++] = *k++; \
            } \
            (dst)[j] = '\0'; \
          } \
        } \
      } \
    } while (0)
    #define GRAB_I(key, dst) do { \
      const char *k = strstr(p, "\"" key "\""); \
      if (k && k < end) { k = strchr(k, ':'); if (k && k < end) (dst) = (int)strtol(k + 1, NULL, 10); } \
    } while (0)

    GRAB_S("id", r.id);
    GRAB_S("name", r.name);
    GRAB_S("category", r.category);
    GRAB_S("modality", r.modality);
    GRAB_S("prompt", r.prompt);
    GRAB_S("camera", r.camera);
    GRAB_S("checkpoint", r.checkpoint);
    GRAB_S("lora", r.lora);
    GRAB_I("width", r.width);
    GRAB_I("height", r.height);
    GRAB_I("steps", r.steps);
    int sd = 42; GRAB_I("seed", sd); r.seed = (uint32_t)sd;
    #undef GRAB_S
    #undef GRAB_I
    if (r.id[0] && r.name[0]) lib->items[lib->count++] = r;
    p = end + 1;
  }
  free(buf);
}

static void persist_user_recipes(const ec_recipe_lib_t *lib) {
  if (!lib->store_path[0]) return;
  FILE *fp = fopen(lib->store_path, "wb");
  if (!fp) return;
  fprintf(fp, "[\n");
  int first = 1;
  for (int i = 0; i < lib->count; i++) {
    const ec_recipe_t *r = &lib->items[i];
    if (r->builtin) continue;
    char ep[1536], en[192], ec[256], eck[256], el[256];
    json_esc(r->prompt, ep, sizeof ep);
    json_esc(r->name, en, sizeof en);
    json_esc(r->camera, ec, sizeof ec);
    json_esc(r->checkpoint, eck, sizeof eck);
    json_esc(r->lora, el, sizeof el);
    fprintf(fp, "%s  {\"id\":\"%s\",\"name\":\"%s\",\"category\":\"%s\",\"modality\":\"%s\","
                "\"prompt\":\"%s\",\"camera\":\"%s\",\"checkpoint\":\"%s\",\"lora\":\"%s\","
                "\"lora_strength\":%.2f,\"width\":%d,\"height\":%d,\"steps\":%d,\"seed\":%u}\n",
            first ? "" : ",", r->id, en, r->category, r->modality, ep, ec, eck, el,
            r->lora_strength, r->width, r->height, r->steps, r->seed);
    first = 0;
  }
  fprintf(fp, "]\n");
  fclose(fp);
}

void ec_recipes_init(ec_recipe_lib_t *lib, const char *user_store_json_path) {
  memset(lib, 0, sizeof *lib);
  if (user_store_json_path && user_store_json_path[0]) {
    snprintf(lib->store_path, sizeof lib->store_path, "%s", user_store_json_path);
  }
  size_t nb = sizeof BUILTIN_RECIPES / sizeof BUILTIN_RECIPES[0];
  for (size_t i = 0; i < nb && lib->count < EC_MAX_RECIPES; i++) {
    lib->items[lib->count++] = BUILTIN_RECIPES[i];
  }
  load_user_recipes(lib);
}

int ec_recipes_save_user(ec_recipe_lib_t *lib, const ec_recipe_t *r) {
  if (!lib || !r || !r->id[0] || !r->name[0]) return -1;
  for (int i = 0; i < lib->count; i++) {
    if (strcmp(lib->items[i].id, r->id) == 0) {
      if (lib->items[i].builtin) return -2;
      lib->items[i] = *r;
      lib->items[i].builtin = 0;
      persist_user_recipes(lib);
      return 0;
    }
  }
  if (lib->count >= EC_MAX_RECIPES) return -1;
  lib->items[lib->count] = *r;
  lib->items[lib->count].builtin = 0;
  lib->count++;
  persist_user_recipes(lib);
  return 0;
}

int ec_recipes_delete_user(ec_recipe_lib_t *lib, const char *id) {
  if (!lib || !id) return -1;
  for (int i = 0; i < lib->count; i++) {
    if (strcmp(lib->items[i].id, id) == 0 && !lib->items[i].builtin) {
      for (int j = i; j + 1 < lib->count; j++) lib->items[j] = lib->items[j + 1];
      lib->count--;
      persist_user_recipes(lib);
      return 0;
    }
  }
  return -1;
}

int ec_recipes_to_json(const ec_recipe_lib_t *lib, char *buf, size_t cap) {
  int n = snprintf(buf, cap, "{\"count\":%d,\"recipes\":[", lib->count);
  for (int i = 0; i < lib->count && (size_t)n < cap - 1024; i++) {
    const ec_recipe_t *r = &lib->items[i];
    char ep[1536], en[192], ec[256], eck[256], el[256];
    json_esc(r->prompt, ep, sizeof ep);
    json_esc(r->name, en, sizeof en);
    json_esc(r->camera, ec, sizeof ec);
    json_esc(r->checkpoint, eck, sizeof eck);
    json_esc(r->lora, el, sizeof el);
    n += snprintf(buf + n, cap - (size_t)n,
        "%s{\"id\":\"%s\",\"name\":\"%s\",\"category\":\"%s\",\"modality\":\"%s\","
        "\"prompt\":\"%s\",\"camera\":\"%s\",\"checkpoint\":\"%s\",\"lora\":\"%s\","
        "\"lora_strength\":%.2f,\"width\":%d,\"height\":%d,\"steps\":%d,\"seed\":%u,\"builtin\":%s}",
        i ? "," : "", r->id, en, r->category, r->modality, ep, ec, eck, el,
        r->lora_strength, r->width, r->height, r->steps, r->seed, r->builtin ? "true" : "false");
  }
  n += snprintf(buf + n, cap - (size_t)n, "]}");
  return n;
}

int ec_prompt_library_to_json(char *buf, size_t cap) {
  return snprintf(buf, cap,
    "{"
    "\"cameras\":[\"Hasselblad X2D 100C\",\"Leica M11 Summilux\",\"ARRI Alexa 65 IMAX\",\"Phase One IQ4 150MP\",\"Sony A1 Mirrorless\",\"Fujifilm GFX 100 II\",\"Canon EOS R5\",\"Mamiya RZ67 Medium Format\",\"RED V-Raptor 8K VV\",\"Contax T2 35mm Point-and-Shoot\"],"
    "\"lenses\":[\"14mm ultra-wide rectilinear f/2.8\",\"24mm tilt-shift architectural f/8\",\"35mm anamorphic cinema f/1.4\",\"50mm Summilux f/1.4\",\"85mm editorial portrait f/1.4\",\"100mm macro focus-stack f/5.6\",\"135mm compressed bokeh f/1.8\",\"400mm wildlife telephoto f/2.8\",\"vintage Cooke Panchro 40mm\",\"Helios 44-2 swirly bokeh 58mm\"],"
    "\"lighting\":[\"natural window sidelight with dust motes\",\"golden hour rim backlight\",\"rain-slicked neon midnight reflections\",\"overhead studio strip softbox\",\"overcast diffused Nordic skylight\",\"chiaroscuro tungsten rim light\",\"practical warm candlelight\",\"volumetric god-rays through mist\",\"bioluminescent ambient glow\",\"high-key editorial cyclorama\"],"
    "\"film_stocks\":[\"Kodak Portra 400 color science\",\"Cinestill 800T tungsten halation\",\"Ilford HP5 Plus 400 monochrome grain\",\"Fujifilm Velvia 50 vivid landscape\",\"Kodak Ektachrome E100 slide transparency\",\"Kodak Vision3 500T cinema stock\",\"ARRI LogC4 naturalistic film grade\",\"Fuji Classic Chrome muted editorial\"],"
    "\"compositions\":[\"shallow depth of field creamy bokeh\",\"rule of thirds editorial framing\",\"rectilinear architectural verticals\",\"overhead flat-lay tabletop\",\"sweeping 2.39:1 anamorphic wide shot\",\"intimate macro focus stack\",\"low-angle monumental perspective\",\"symmetrical Kubrick one-point corridor\",\"over-the-shoulder candid documentary\"],"
    "\"color_grades\":[\"muted teal and burnt ochre cinema grade\",\"warm Kodak amber highlights & deep charcoal shadows\",\"crisp high-contrast monochrome silver gelatin\",\"pastel Nordic daylight minimal palette\",\"cyberpunk magenta & sodium-vapor night grade\",\"bleached bypass gritty film look\",\"velvet emerald & antique gold editorial\"],"
    "\"negative_prompts\":[\"blurry, low quality, distorted anatomy, watermark, text overlay, plastic skin\",\"static frozen frame, flicker, temporal jitter, morphing artifacts, overexposed\",\"extra limbs, deformed hands, bad proportions, oversaturated HDR, CGI sheen\",\"noisy audio, clipping distortion, muddiness, harsh sibilance, off-key pitch\"],"
    "\"video_camera_moves\":[\"slow 360 orbital dolly around subject\",\"smooth steady dolly-in push forward\",\"sweeping crane camera rise upward\",\"aerial FPV drone low-altitude glide\",\"handheld documentary tracking shot\",\"locked-off tripod cinemagraph frame\",\"subtle rack focus foreground to background\",\"panoramic whip-pan transition\",\"slow-motion 120fps tracking orbit\"],"
    "\"video_physics\":[\"crisp water droplets & wet surface specular ripples\",\"billowing silk fabric in golden hour wind\",\"delicate curling steam & volumetric smoke drift\",\"realistic ember sparks & drifting ash\",\"soft dappled leaf shadows swaying gently\",\"ocean surf foam breaking in slow motion\",\"warm breath vapor in crisp winter air\"],"
    "\"materials_3d\":[\"brushed titanium + anti-reflective sapphire glass PBR\",\"hand-thrown matte stoneware ceramic glaze\",\"weathered patinated bronze + oiled walnut\",\"translucent frosted acrylic + anodized aluminum\",\"forged carbon fiber weave + knurled steel\",\"sculpted travertine marble with natural veins\",\"hand-stitched saddle leather + solid brass\",\"iridescent dichroic optical glass\"],"
    "\"3d_styles\":[\"clean watertight studio turntable asset\",\"hard-surface industrial design prototype\",\"stylized hand-painted game hero prop\",\"architectural scale maquette model\",\"museum-grade sculptural porcelain artifact\",\"organic botanical macro specimen\",\"mid-century Scandinavian furniture render\"],"
    "\"music_genres\":[\"warm jazz house groove with Rhodes chords\",\"Kyoto rain lo-fi boom-bap beats\",\"Berlin warehouse melodic techno 132 BPM\",\"shimmering indie dream-pop vocal anthem\",\"sweeping Hans Zimmer orchestral trailer score\",\"Afrobeats Amapiano log drum summer groove\",\"Nordic ambient modular analog soundscape\",\"neo-soul funk pocket with live horns\",\"UK garage 2-step chopped vocal club\",\"acoustic folk fingerpicked intimate ballad\"],"
    "\"music_instruments\":[\"Fender Rhodes electric piano\",\"Moog Sub 37 analog bass\",\"Roland Juno-106 lush chorus pad\",\"acoustic brushed jazz kit\",\"upright double bass\",\"solo expressive cello + chamber strings\",\"vintage TR-909 punchy percussion\",\"chorus-drenched Fender Jazzmaster guitar\",\"deep Amapiano log drum\",\"felt grand piano with intimate hammer noise\"],"
    "\"music_structures\":[\"[Intro]\\n\",\"[Verse 1]\\n\",\"[Pre-Chorus]\\n\",\"[Chorus]\\n\",\"[Verse 2]\\n\",\"[Bridge]\\n\",\"[Drop]\\n\",\"[Instrumental Solo]\\n\",\"[Breakdown]\\n\",\"[Outro]\\n\"],"
    "\"music_palettes\":[\"warm analog Moog bass + brushed drums 124 BPM\",\"lush strings + solo cello cinematic 84 BPM\",\"lo-fi Rhodes chords + dusty vinyl breaks 92 BPM\",\"driving Berlin techno arpeggio + sub 132 BPM\"]"
    "}");
}

// ===========================================================================
// Pre-Seeded 20-Item Elite LoRA & QLoRA (NF4 / FP8 / DoRA / LoHa / LoKr) Catalog
// ===========================================================================
typedef struct {
  const char *id;
  const char *name;
  const char *arch;       // "flux2" | "wan22" | "acestep" | "trellis2"
  const char *format;     // "lora" | "qlora_nf4" | "qlora_fp8" | "dora" | "loha" | "lokr"
  int         rank;
  float       strength;
  const char *trigger;
  const char *repo;
  const char *filename;
  const char *desc;
} ec_lora_catalog_item_t;

static const ec_lora_catalog_item_t ELITE_LORA_CATALOG[] = {
  // --- FLUX.2-klein-4B Photoreal Image LoRAs (8 live-verified HuggingFace repos) ---
  {"flux2-consistence-edit", "Identity & Subject Consistency (r128)", "flux2", "lora", 128, 0.85f,
   "consistent subject, natural lighting", "xocialize/consistence-edit-FLUX.2-klein-4B-lora", "f2k_4B_consist_20260314.safetensors",
   "High-rank BF16 adapter locking character identity, wardrobe, and studio lighting continuity across multi-shot edits."},
  {"flux2-refcontrol-depth", "Depth & Architectural Structure Control", "flux2", "lora", 32, 0.85f,
   "crisp depth perspective, rectilinear lines", "thedeoxen/refcontrol-FLUX.2-klein-4B-reference-depth-lora", "flux2_klein_4b_refcontrol_depth.safetensors",
   "Precision spatial depth conditioning adapter for razor-sharp architectural geometry, interior staging, and product perspective."},
  {"flux2-refcontrol-pose", "Editorial Portrait & Pose Director", "flux2", "lora", 32, 0.80f,
   "editorial pose, natural anatomy", "xocialize/refcontrol-FLUX.2-klein-4B-pose-lora", "refcontrol-pose-klein-4b.safetensors",
   "Articulated human pose and gesture control tuned on medium-format fashion and documentary portrait sessions."},
  {"flux2-360-outpaint", "360 Panorama & Equirectangular Outpaint", "flux2", "lora", 16, 0.90f,
   "360 panorama, seamless horizon", "nomadoor/flux-2-klein-4B-360-erp-outpaint-lora", "flux-2-klein-4B-360-erp-outpaint-lora_V1.safetensors",
   "Seamless 360-degree environment & skybox expansion for virtual production stages and 3D lighting domes."},
  {"flux2-studio-outpaint", "Seamless Canvas & Frame Expansion", "flux2", "lora", 32, 0.85f,
   "wide angle expansion, seamless border", "xocialize/outpaint-FLUX.2-klein-4B-lora", "flux-outpaint-lora.safetensors",
   "Eliminates seam boundaries when expanding aspect ratios from portrait/square into wide cinema frames."},
  {"flux2-vintage-editorial", "Vintage Print & Editorial Book Cover", "flux2", "lora", 32, 0.85f,
   "vintage book cover, tactile paper grain", "Norod78/flux2-klein-4b-base-lora-vintage-book-cover", "flux2-klein-4b-base-lora-vintage-book-cover.safetensors",
   "Warm halftone ink texture, classic serif cover composition, and authentic mid-century print color science."},
  {"flux2-clean-bg-isolate", "Studio Packshot & Clean Isolation", "flux2", "lora", 32, 0.90f,
   "isolated studio backdrop, crisp silhouette", "xocialize/background-remove-FLUX.2-klein-4B-lora", "flux-background-remove-lora.safetensors",
   "Isolates hero subjects on pure studio backdrops — ideal before lifting shots into TRELLIS.2 3D reconstruction."},
  {"flux2-pixel-art", "Crisp 16-Bit Pixel Art & Isometric", "flux2", "lora", 128, 0.85f,
   "pixel art, crisp sprite palette", "adirik/pixel-art-lora-flux.2-klein-4B", "pixelart_lora.safetensors",
   "True grid-locked pixel clusters, hand-dithered shading, and zero anti-aliased smear for game assets."},
  // --- Wan 2.2 TI2V-5B Cinema Video LoRAs (6 live-verified HuggingFace repos) ---
  {"wan22-turbo-r64-fp16", "Wan 2.2 TI2V-5B 6-Step Turbo (FP16 r64)", "wan22", "lora", 64, 1.00f,
   "", "hum-ma/Wan2.2-TI2V-5B-Turbo-GGUF", "Wan22_TI2V_5B_Turbo_lora_rank_64_fp16.safetensors",
   "Step-distillation rank-64 FP16 adapter enabling crisp 6-step cinema video generation at 2.5x faster DiT latency."},
  {"wan22-fastwan-r128-bf16", "FastWan Full-Attention High-Motion (r128)", "wan22", "lora", 128, 0.95f,
   "dynamic camera motion, sharp temporal detail", "hum-ma/Wan2.2-TI2V-5B-Turbo-GGUF", "Wan2_2_5B_FastWanFullAttn_lora_rank_128_bf16.safetensors",
   "Full-attention rank-128 BF16 distillation adapter preserving fast subject motion and camera pans at low step counts."},
  {"wan22-cinemagraph-motion", "Subtle Living Cinemagraph Loop", "wan22", "lora", 16, 0.85f,
   "cinemagraph, locked tripod, subtle atmospheric motion", "hmb/wan22-ti2v-5b-cinemagraph-lora", "checkpoint-1000/pytorch_lora_weights.safetensors",
   "Locks architectural & portrait framing on a rock-steady tripod while animating water, steam, foliage, and light."},
  {"wan22-travel-camera", "Cinema Gimbal & Travel Documentary", "wan22", "lora", 16, 0.85f,
   "gimbal glide, golden hour travel cinematography", "satoshinji/wan22-ti2v-5b-travel-lora", "travel_lora_step380.safetensors",
   "Smooth forward Steadicam & aerial travel cinematography with natural parallax and golden-hour atmospheric depth."},
  {"wan22-oil-cinema", "Impasto Oil Painting Motion Cinema", "wan22", "lora", 32, 0.80f,
   "living oil painting, expressive brushwork", "svjack/Landscape_to_Oil_Painting_wan_2_2_5B_TI2V_lora", "wan2_2_oil_painting_lora_v1_000000250.safetensors",
   "Transforms landscapes and start-frame keyframes into tactile animated oil-on-canvas brushwork."},
  {"wan22-stylized-3d-motion", "Stylized 3D Studio Animation", "wan22", "lora", 32, 0.80f,
   "stylized 3d animation, clean studio lighting", "FalconNet/Wan2.2-TI2V-5B-ZackDFilms-LoRA", "step-600.safetensors",
   "Clean physically-lit stylized 3D motion picture aesthetic with crisp silhouettes and smooth keyframe transitions."},
  // --- ACE-Step 1.5 Music & Audio Adapters (4 live-verified HuggingFace repos) ---
  {"acestep-afro-amapiano-nf4", "Naija Afrobeats & Amapiano Percussion", "acestep", "lora", 128, 0.85f,
   "afrobeats groove, talking drum, amapiano log drum", "David-A-Amoo/ACE-Step-1.5-Naija-Legacy-Rhythms-LoRA-v1", "adapter_model.safetensors",
   "Authentic West African percussion, syncopated log-drum basslines, warm shakers, and modern Lagos studio pocket."},
  {"acestep-vocal-clarity-dora", "Raspy Soul & Expressive Lead Vocals", "acestep", "lora", 64, 0.80f,
   "intimate raspy lead vocal, live studio mic", "DisturbingTheField/ACE-Step-v1.5-raspy-vocal-and-instrumental-5-LoRAs", "adapter_model.safetensors",
   "Enhances emotional vocal grit, breath realism, and center-channel lyric articulation for rock, soul, and indie."},
  {"acestep-acoustic-studio", "Acoustic Guitar & Unplugged Session", "acestep", "lora", 64, 0.85f,
   "close-mic acoustic guitar, fingerpicked warmth", "DisturbingTheField/ACE-Step-v1.5-acoustic-guitar-and-a-merge-LoRA", "vocal_instrument_merge_adapter_model.safetensors",
   "Captures natural steel-string transient detail, wood body resonance, and intimate unplugged vocal-guitar balance."},
  {"acestep-ambient-cinema", "Ambient Dreamscape & Film Texture", "acestep", "lora", 64, 0.85f,
   "ambient dreamscape, lush analog pads, wide reverb", "DisturbingTheField/ACE-Step-v1.5-ambient_dream1-LoRA", "adapter_model.safetensors",
   "Expansive stereo soundstage tuned for atmospheric cinema scores, modular synth pads, and meditative textures."},
  // --- TRELLIS.2 1.3B 3D Asset LoRAs (2 live-verified HuggingFace repos) ---
  {"trellis2-hardsurface-cad-lora", "Automotive & Hard-Surface CAD Sculpt", "trellis2", "lora", 16, 0.85f,
   "clean automotive surfacing, crisp panel bevels", "PoochieMama/trellis2-cars-lora-v2", "shape_lora/adapter-1200.safetensors",
   "Sharpens hard-surface body lines, wheel arches, and symmetric industrial silhouettes in TRELLIS.2 sparse-structure flow."},
  {"trellis2-stylized-sculpt-nf4", "Voxel Block & Stylized Game Prop", "trellis2", "lora", 16, 0.85f,
   "crisp voxel block geometry, stylized game asset", "unfundedResearcher/TRELLIS2-Minecraft-Texture-LoRA-v1", "adapter_model.safetensors",
   "Produces crisp orthogonal block silhouettes and stylized low-poly game prop proportions."}
};

int ec_lora_catalog_to_json(const ec_storage_vault_t *vault, char *buf, size_t cap) {
  size_t n_items = sizeof(ELITE_LORA_CATALOG) / sizeof(ELITE_LORA_CATALOG[0]);
  int n = snprintf(buf, cap, "{\"count\":%zu,\"catalog\":[", n_items);
  for (size_t i = 0; i < n_items && (size_t)n < cap - 640; i++) {
    const ec_lora_catalog_item_t *c = &ELITE_LORA_CATALOG[i];
    const ec_storage_asset_t *found = vault ? ec_storage_find(vault, EC_ASSET_LORA, c->id) : NULL;
    if (!found && vault && !strchr(c->filename, '/') &&
        strcmp(c->filename, "adapter_model.safetensors") != 0 &&
        strcmp(c->filename, "pytorch_lora_weights.safetensors") != 0) {
      found = ec_storage_find(vault, EC_ASSET_LORA, c->filename);
    }
    char ed[512], et[256], ep[512];
    json_esc(c->desc, ed, sizeof ed);
    json_esc(c->trigger, et, sizeof et);
    json_esc(found ? found->path : "", ep, sizeof ep);
    n += snprintf(buf + n, cap - (size_t)n,
        "%s{\"id\":\"%s\",\"name\":\"%s\",\"arch\":\"%s\",\"format\":\"%s\","
        "\"rank\":%d,\"strength\":%.2f,\"trigger\":\"%s\",\"repo\":\"%s\","
        "\"filename\":\"%s\",\"desc\":\"%s\",\"installed\":%s,\"local_path\":\"%s\"}",
        i ? "," : "", c->id, c->name, c->arch, c->format, c->rank, c->strength,
        et, c->repo, c->filename, ed, found ? "true" : "false", ep);
  }
  n += snprintf(buf + n, cap - (size_t)n, "]}");
  return n;
}

// ===========================================================================
// Pure-C11 PNG Embedded ComfyUI Metadata Extractor (tEXt / iTXt, zero libpng!)
// ===========================================================================
static uint32_t rd_be32(const uint8_t *p) {
  return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) | (uint32_t)p[3];
}

int ec_png_extract_comfy_json(const uint8_t *png, size_t len, char *out_json, size_t out_cap) {
  static const uint8_t SIG[8] = { 137, 80, 78, 71, 13, 10, 26, 10 };
  if (!png || len < 33 || memcmp(png, SIG, 8) != 0 || !out_json || out_cap < 32) return -1;

  size_t off = 8;
  size_t best_len = 0;
  int found_prompt = 0;

  while (off + 12 <= len) {
    uint32_t clen = rd_be32(png + off);
    const char *ctype = (const char *)(png + off + 4);
    if (off + 12 + clen > len) break;
    const uint8_t *data = png + off + 8;

    if (memcmp(ctype, "tEXt", 4) == 0 && clen > 7) {
      const uint8_t *nul = (const uint8_t *)memchr(data, 0, clen);
      if (nul) {
        size_t klen = (size_t)(nul - data);
        int is_prompt = (klen == 6 && memcmp(data, "prompt", 6) == 0);
        int is_wf     = (klen == 8 && memcmp(data, "workflow", 8) == 0);
        if (is_prompt || (is_wf && !found_prompt)) {
          size_t vlen = clen - klen - 1;
          if (vlen > 2 && vlen < out_cap) {
            memcpy(out_json, nul + 1, vlen);
            out_json[vlen] = '\0';
            best_len = vlen;
            if (is_prompt) found_prompt = 1;
          }
        }
      }
    } else if (memcmp(ctype, "iTXt", 4) == 0 && clen > 10) {
      const uint8_t *nul = (const uint8_t *)memchr(data, 0, clen);
      if (nul && (size_t)(nul - data) + 5 < clen) {
        size_t klen = (size_t)(nul - data);
        uint8_t comp_flag = nul[1];
        if (comp_flag == 0) {
          int is_prompt = (klen == 6 && memcmp(data, "prompt", 6) == 0);
          int is_wf     = (klen == 8 && memcmp(data, "workflow", 8) == 0);
          if (is_prompt || (is_wf && !found_prompt)) {
            // Skip compression_flag(1), compression_method(1), language\0, translated\0
            const uint8_t *q = nul + 3;
            const uint8_t *end = data + clen;
            while (q < end && *q != 0) q++;
            if (q < end) q++;
            while (q < end && *q != 0) q++;
            if (q < end) q++;
            size_t vlen = (size_t)(end - q);
            if (vlen > 2 && vlen < out_cap) {
              memcpy(out_json, q, vlen);
              out_json[vlen] = '\0';
              best_len = vlen;
              if (is_prompt) found_prompt = 1;
            }
          }
        }
      }
    }
    off += 12 + (size_t)clen;
  }
  return (int)best_len;
}

// ===========================================================================
// Universal ComfyUI JSON Normalizer (API format OR LiteGraph UI format!)
// ===========================================================================
static void skip_json_ws(const char **pp) {
  while (**pp && (**pp == ' ' || **pp == '\n' || **pp == '\r' || **pp == '\t')) (*pp)++;
}

static int grab_json_string(const char **pp, char *dst, size_t cap) {
  skip_json_ws(pp);
  if (**pp != '"') return -1;
  (*pp)++;
  size_t j = 0;
  while (**pp && **pp != '"') {
    if (**pp == '\\' && (*pp)[1]) {
      (*pp)++;
      char c = **pp;
      if (c == 'n') c = '\n';
      else if (c == 't') c = '\t';
      if (j + 1 < cap) dst[j++] = c;
      (*pp)++;
    } else {
      if (j + 1 < cap) dst[j++] = **pp;
      (*pp)++;
    }
  }
  if (**pp != '"') return -1;
  (*pp)++;
  dst[j] = '\0';
  return 0;
}

static void skip_json_val(const char **pp) {
  skip_json_ws(pp);
  char c = **pp;
  if (c == '"') {
    char tmp[8];
    grab_json_string(pp, tmp, sizeof tmp);
  } else if (c == '{' || c == '[') {
    char open = c, close = (c == '{') ? '}' : ']';
    int depth = 1;
    (*pp)++;
    while (**pp && depth > 0) {
      if (**pp == '"') {
        (*pp)++;
        while (**pp && **pp != '"') { if (**pp == '\\' && (*pp)[1]) (*pp)++; (*pp)++; }
        if (**pp == '"') (*pp)++;
      } else {
        if (**pp == open) depth++;
        else if (**pp == close) depth--;
        (*pp)++;
      }
    }
  } else {
    while (**pp && **pp != ',' && **pp != '}' && **pp != ']' && !isspace((unsigned char)**pp)) (*pp)++;
  }
}

static int ci_has(const char *h, const char *n) {
  if (!h || !n) return 0;
  size_t nl = strlen(n);
  for (const char *p = h; *p; p++) if (strncasecmp(p, n, nl) == 0) return 1;
  return 0;
}

// Find a direct child key of a JSON object `{ ... }` (immune to nested objects/arrays!)
static const char *find_obj_key(const char *obj_s, const char *obj_e, const char *target_key) {
  if (!obj_s || !obj_e || *obj_s != '{') return NULL;
  const char *p = obj_s + 1;
  while (p < obj_e) {
    skip_json_ws(&p);
    if (p >= obj_e || *p == '}') break;
    if (*p == ',') { p++; continue; }
    char k[96] = {0};
    if (grab_json_string(&p, k, sizeof k) != 0) break;
    skip_json_ws(&p);
    if (p < obj_e && *p == ':') p++;
    skip_json_ws(&p);
    if (strcmp(k, target_key) == 0) return p;
    skip_json_val(&p);
  }
  return NULL;
}

typedef struct {
  int    id;
  char   type[128];
  char   wstrs[6][1024];
  int    n_wstrs;
  double wnums[12];
  int    n_wnums;
  char   in_names[16][48];
  int    in_link_ids[16];
  int    n_ins;
  // Optional Power Lora Loader (rgthree) inline specs
  char   rg_loras[4][192];
  float  rg_strengths[4];
  int    n_rg_loras;
} lg_node_t;

typedef struct {
  int id;
  int from_node;
  int from_slot;
  int to_node;
  int to_slot;
} lg_link_t;

typedef struct {
  char       uuid[64];
  char       in_names[16][48];
  int        n_sg_in;
  lg_node_t *nodes;
  int        n_nodes;
  lg_link_t *links;
  int        n_links;
} lg_subgraph_t;

#define LG_MAX_NODES 512
#define LG_MAX_LINKS 2048
#define LG_MAX_SG    16

static int parse_lg_links(const char *arr_s, const char *arr_e, lg_link_t *out, int cap) {
  int cnt = 0;
  if (!arr_s || *arr_s != '[') return 0;
  const char *p = arr_s + 1;
  while (p < arr_e && *p != ']' && cnt < cap) {
    skip_json_ws(&p);
    if (*p == ',') { p++; continue; }
    if (*p == '[') {
      // Classic LiteGraph tuple: [id, origin_id, origin_slot, target_id, target_slot, type]
      p++;
      lg_link_t L = {0, -1, 0, -1, 0};
      L.id = (int)strtol(p, (char **)&p, 10);
      if (*p == ',') p++;
      L.from_node = (int)strtol(p, (char **)&p, 10);
      if (*p == ',') p++;
      L.from_slot = (int)strtol(p, (char **)&p, 10);
      if (*p == ',') p++;
      L.to_node = (int)strtol(p, (char **)&p, 10);
      if (*p == ',') p++;
      L.to_slot = (int)strtol(p, (char **)&p, 10);
      out[cnt++] = L;
      while (p < arr_e && *p != ']') p++;
      if (p < arr_e && *p == ']') p++;
    } else if (*p == '{') {
      // ComfyUI 2026 Subgraph link object: {"id":137,"origin_id":68,"origin_slot":0,"target_id":62,"target_slot":0}
      const char *os = p;
      skip_json_val(&p);
      const char *oe = p;
      lg_link_t L = {0, -1, 0, -1, 0};
      const char *v;
      if ((v = find_obj_key(os, oe, "id")))          L.id = (int)strtol(v, NULL, 10);
      if ((v = find_obj_key(os, oe, "origin_id")))   L.from_node = (int)strtol(v, NULL, 10);
      if ((v = find_obj_key(os, oe, "origin_slot"))) L.from_slot = (int)strtol(v, NULL, 10);
      if ((v = find_obj_key(os, oe, "target_id")))   L.to_node = (int)strtol(v, NULL, 10);
      if ((v = find_obj_key(os, oe, "target_slot"))) L.to_slot = (int)strtol(v, NULL, 10);
      out[cnt++] = L;
    } else {
      break;
    }
  }
  return cnt;
}

static int parse_lg_nodes(const char *arr_s, const char *arr_e, lg_node_t *out, int cap) {
  int cnt = 0;
  if (!arr_s || *arr_s != '[') return 0;
  const char *p = arr_s + 1;
  while (p < arr_e && *p != ']' && cnt < cap) {
    skip_json_ws(&p);
    if (*p == ',') { p++; continue; }
    if (*p != '{') break;
    const char *os = p;
    skip_json_val(&p);
    const char *oe = p;

    lg_node_t *nd = &out[cnt];
    memset(nd, 0, sizeof *nd);
    nd->id = -1;

    const char *q = os + 1;
    while (q < oe) {
      skip_json_ws(&q);
      if (q >= oe || *q == '}') break;
      if (*q == ',') { q++; continue; }
      char fk[64] = {0};
      if (grab_json_string(&q, fk, sizeof fk) != 0) break;
      skip_json_ws(&q);
      if (*q == ':') q++;
      skip_json_ws(&q);

      if (strcmp(fk, "id") == 0) {
        nd->id = (int)strtol(q, (char **)&q, 10);
      } else if (strcmp(fk, "type") == 0) {
        grab_json_string(&q, nd->type, sizeof nd->type);
      } else if (strcmp(fk, "widgets_values") == 0 && *q == '[') {
        q++;
        while (q < oe && *q != ']') {
          skip_json_ws(&q);
          if (*q == ',') { q++; continue; }
          if (*q == '"') {
            char tmp[1024] = {0};
            grab_json_string(&q, tmp, sizeof tmp);
            if (nd->n_wstrs < 6) {
              snprintf(nd->wstrs[nd->n_wstrs], sizeof nd->wstrs[0], "%s", tmp);
              nd->n_wstrs++;
            }
          } else if (*q == '-' || isdigit((unsigned char)*q)) {
            double dv = strtod(q, (char **)&q);
            if (nd->n_wnums < 12) nd->wnums[nd->n_wnums++] = dv;
          } else if (*q == '{') {
            // Support Power Lora Loader (rgthree): {"on":true,"lora":"foo.safetensors","strength":0.8}
            const char *ws = q;
            skip_json_val(&q);
            const char *we = q;
            const char *on_v = find_obj_key(ws, we, "on");
            const char *lo_v = find_obj_key(ws, we, "lora");
            const char *st_v = find_obj_key(ws, we, "strength");
            int is_on = (!on_v || strncmp(on_v, "true", 4) == 0);
            if (is_on && lo_v && *lo_v == '"' && nd->n_rg_loras < 4) {
              const char *lv = lo_v;
              char lname[192] = {0};
              if (grab_json_string(&lv, lname, sizeof lname) == 0 && lname[0] && strcasecmp(lname, "None") != 0) {
                snprintf(nd->rg_loras[nd->n_rg_loras], sizeof nd->rg_loras[0], "%s", lname);
                nd->rg_strengths[nd->n_rg_loras] = st_v ? strtof(st_v, NULL) : 0.85f;
                nd->n_rg_loras++;
              }
            }
          } else {
            skip_json_val(&q);
          }
        }
        if (q < oe && *q == ']') q++;
      } else if (strcmp(fk, "inputs") == 0 && *q == '[') {
        q++;
        while (q < oe && *q != ']') {
          skip_json_ws(&q);
          if (*q == ',') { q++; continue; }
          if (*q == '{') {
            const char *is = q;
            skip_json_val(&q);
            const char *ie = q;
            const char *nv = find_obj_key(is, ie, "name");
            const char *lv = find_obj_key(is, ie, "link");
            if (nv && *nv == '"' && lv && nd->n_ins < 16) {
              char iname[48] = {0};
              const char *nv2 = nv;
              grab_json_string(&nv2, iname, sizeof iname);
              if (*lv == '-' || isdigit((unsigned char)*lv)) {
                snprintf(nd->in_names[nd->n_ins], sizeof nd->in_names[0], "%s", iname);
                nd->in_link_ids[nd->n_ins] = (int)strtol(lv, NULL, 10);
                nd->n_ins++;
              }
            }
          } else {
            skip_json_val(&q);
          }
        }
        if (q < oe && *q == ']') q++;
      } else {
        skip_json_val(&q);
      }
    }
    if (nd->id >= 0 && nd->type[0]) cnt++;
  }
  return cnt;
}

static const lg_node_t *find_lg_node(const lg_node_t *nodes, int n_nodes, int id) {
  for (int i = 0; i < n_nodes; i++) if (nodes[i].id == id) return &nodes[i];
  return NULL;
}

static const lg_link_t *find_lg_link(const lg_link_t *links, int n_links, int lid) {
  for (int i = 0; i < n_links; i++) if (links[i].id == lid) return &links[i];
  return NULL;
}

// Trace backward from any conditioning/string node to recover exact positive vs negative prompt!
static int trace_lg_prompt(
    const lg_node_t *nodes, int n_nodes, const lg_link_t *links, int n_links,
    int node_id, int from_slot, char *out, size_t cap, int depth) {
  if (depth > 8 || !out || cap < 2) return 0;
  const lg_node_t *nd = find_lg_node(nodes, n_nodes, node_id);
  if (!nd) return 0;
  const char *t = nd->type;

  // Explicit empty negative conditioning -> do NOT overwrite with heuristics!
  if (ci_has(t, "ConditioningZeroOut")) {
    out[0] = '\0';
    return 1;
  }

  // Qwen-Image 2.1 dual-output text encoder: slot 0 = positive (wstrs[0]), slot 1 = negative (wstrs[1])
  if (ci_has(t, "TextEncodeQwenImage")) {
    int idx = (from_slot == 1 && nd->n_wstrs > 1) ? 1 : 0;
    if (nd->n_wstrs > idx && nd->wstrs[idx][0]) {
      snprintf(out, cap, "%.*s", (int)(cap - 1), nd->wstrs[idx]);
      return 1;
    }
  }

  // ACE-Step 1.5 audio text encoder: wstrs[0] = tags/caption
  if (ci_has(t, "TextEncodeAceStepAudio")) {
    if (nd->n_wstrs > 0 && nd->wstrs[0][0]) {
      snprintf(out, cap, "%.*s", (int)(cap - 1), nd->wstrs[0]);
      return 1;
    }
  }

  // Standard CLIPTextEncode / CLIPTextEncodeFlux / PrimitiveString* / ShowText
  if (ci_has(t, "CLIPTextEncode") || ci_has(t, "PrimitiveString") ||
      ci_has(t, "StringLiteral") || ci_has(t, "PromptText") || ci_has(t, "ShowText")) {
    // First follow linked "text" / "string" / "clip_l" / "t5xxl" input if wired!
    for (int i = 0; i < nd->n_ins; i++) {
      if (ci_has(nd->in_names[i], "text") || ci_has(nd->in_names[i], "string") ||
          ci_has(nd->in_names[i], "prompt") || ci_has(nd->in_names[i], "t5")) {
        const lg_link_t *lk = find_lg_link(links, n_links, nd->in_link_ids[i]);
        if (lk && trace_lg_prompt(nodes, n_nodes, links, n_links, lk->from_node, lk->from_slot, out, cap, depth + 1)) {
          if (out[0]) return 1;
        }
      }
    }
    if (nd->n_wstrs > 0 && nd->wstrs[0][0]) {
      snprintf(out, cap, "%.*s", (int)(cap - 1), nd->wstrs[0]);
      return 1;
    }
  }

  // PrimitiveNode holding a string
  if (strcmp(t, "PrimitiveNode") == 0 && nd->n_wstrs > 0 && nd->wstrs[0][0]) {
    snprintf(out, cap, "%.*s", (int)(cap - 1), nd->wstrs[0]);
    return 1;
  }

  // KJNodes GetNode -> find matching SetNode with same channel name wstrs[0]
  if (strcmp(t, "GetNode") == 0 && nd->n_wstrs > 0) {
    for (int i = 0; i < n_nodes; i++) {
      if (strcmp(nodes[i].type, "SetNode") == 0 && nodes[i].n_wstrs > 0 &&
          strcmp(nodes[i].wstrs[0], nd->wstrs[0]) == 0 && nodes[i].n_ins > 0) {
        const lg_link_t *lk = find_lg_link(links, n_links, nodes[i].in_link_ids[0]);
        if (lk) return trace_lg_prompt(nodes, n_nodes, links, n_links, lk->from_node, lk->from_slot, out, cap, depth + 1);
      }
    }
  }

  // Passthrough conditioning modifiers (FluxGuidance, ConditioningCombine, ConditioningSetArea, etc.)
  if (ci_has(t, "Conditioning") || ci_has(t, "Guidance") || ci_has(t, "ControlNetApply") || ci_has(t, "Reroute")) {
    for (int i = 0; i < nd->n_ins; i++) {
      const lg_link_t *lk = find_lg_link(links, n_links, nd->in_link_ids[i]);
      if (lk && trace_lg_prompt(nodes, n_nodes, links, n_links, lk->from_node, lk->from_slot, out, cap, depth + 1)) {
        return 1;
      }
    }
  }
  return 0;
}

static void detect_modality_from_type(const char *ct, int *has_music, int *has_3d, int *has_video) {
  if (!ct || !ct[0]) return;
  if (ci_has(ct, "AceStep") || ci_has(ct, "VAEDecodeAudio") || ci_has(ct, "SaveAudio") || ci_has(ct, "LatentAudio"))
    *has_music = 1;
  if (ci_has(ct, "Trellis") || ci_has(ct, "Pixal3D") || ci_has(ct, "Save3D") || ci_has(ct, "MeshToFile3D") || ci_has(ct, "Preview3D"))
    *has_3d = 1;
  if (ci_has(ct, "Wan22") || ci_has(ct, "WanVideo") || ci_has(ct, "CreateVideo") || ci_has(ct, "SaveVideo") ||
      ci_has(ct, "VHS_VideoCombine") || ci_has(ct, "LatentVideo") || ci_has(ct, "LTXV"))
    *has_video = 1;
}

static int is_better_checkpoint(const char *cand, const char *cur, const ec_storage_vault_t *vault) {
  if (!cand || !cand[0]) return 0;
  if (!cur || !cur[0]) return 1;
  // Prefer distilled 4-step flux-2-klein-4b over flux-2-klein-base-4b when both appear in multi-subgraph templates
  if (ci_has(cur, "base") && !ci_has(cand, "base")) return 1;
  if (vault) {
    const ec_storage_asset_t *ac = ec_storage_find(vault, EC_ASSET_CHECKPOINT, cand);
    const ec_storage_asset_t *ao = ec_storage_find(vault, EC_ASSET_CHECKPOINT, cur);
    if (ac && !ao) return 1;
  }
  return 0;
}

int ec_comfy_normalize_and_resolve(
    const char *raw_json,
    const ec_storage_vault_t *vault,
    char *out_api_json, size_t out_api_cap,
    ec_comfy_import_info_t *info) {
  memset(info, 0, sizeof *info);
  snprintf(info->modality, sizeof info->modality, "image");
  info->width = 512; info->height = 512; info->steps = 4; info->seed = 42;
  if (!raw_json || !out_api_json || out_api_cap < 64) {
    snprintf(info->error, sizeof info->error, "empty workflow json");
    return -1;
  }

  const char *root_s = raw_json;
  skip_json_ws(&root_s);
  const char *root_e = raw_json + strlen(raw_json);
  const char *top_nodes = (*root_s == '{') ? find_obj_key(root_s, root_e, "nodes") : NULL;
  const char *top_links = (*root_s == '{') ? find_obj_key(root_s, root_e, "links") : NULL;
  int is_litegraph = (top_nodes != NULL && *top_nodes == '[' && top_links != NULL && *top_links == '[');

  snprintf(info->source_format, sizeof info->source_format, "%s",
           is_litegraph ? "comfy_litegraph_ui" : "comfy_api");

  int has_music = 0, has_3d = 0, has_video = 0;

  if (!is_litegraph) {
    // API format: extract all nodes, modality, pos/neg link targets, 2026 node classes
    snprintf(out_api_json, out_api_cap, "%s", raw_json);
    typedef struct { char id[32]; char ct[128]; char text[768]; char text2[768]; char pos_ref[32]; int pos_slot; char neg_ref[32]; int neg_slot; } api_nd_t;
    api_nd_t *ands = (api_nd_t *)calloc(256, sizeof(api_nd_t));
    int n_ands = 0;

    if (*root_s == '{') {
      const char *p = root_s + 1;
      while (p < root_e && n_ands < 256) {
        skip_json_ws(&p);
        if (p >= root_e || *p == '}') break;
        if (*p == ',') { p++; continue; }
        char nid[32] = {0};
        if (grab_json_string(&p, nid, sizeof nid) != 0) break;
        skip_json_ws(&p);
        if (*p == ':') p++;
        skip_json_ws(&p);
        if (*p != '{') { skip_json_val(&p); continue; }
        const char *os = p;
        skip_json_val(&p);
        const char *oe = p;

        const char *ct_v = find_obj_key(os, oe, "class_type");
        if (!ct_v || *ct_v != '"') continue;
        api_nd_t *an = &ands[n_ands];
        snprintf(an->id, sizeof an->id, "%s", nid);
        const char *ct_p = ct_v;
        grab_json_string(&ct_p, an->ct, sizeof an->ct);
        n_ands++;
        info->node_count++;

        detect_modality_from_type(an->ct, &has_music, &has_3d, &has_video);
        const char *ins = find_obj_key(os, oe, "inputs");
        if (!ins || *ins != '{') continue;
        const char *ins_end = ins;
        skip_json_val(&ins_end);

        const char *v;
        if (ci_has(an->ct, "LoadImage")) {
          info->needs_image_input = 1;
          if ((v = find_obj_key(ins, ins_end, "image")) && *v == '"') {
            const char *vp = v; grab_json_string(&vp, info->input_image_hint, sizeof info->input_image_hint);
          }
        }
        if (ci_has(an->ct, "TextEncodeAceStepAudio")) {
          if ((v = find_obj_key(ins, ins_end, "tags")) && *v == '"') {
            const char *vp = v; grab_json_string(&vp, an->text, sizeof an->text);
            snprintf(info->extracted_prompt, sizeof info->extracted_prompt, "%s", an->text);
          }
          if ((v = find_obj_key(ins, ins_end, "lyrics")) && *v == '"') {
            const char *vp = v; grab_json_string(&vp, info->extracted_lyrics, sizeof info->extracted_lyrics);
          }
          if ((v = find_obj_key(ins, ins_end, "bpm")))      info->bpm = (int)strtol(v, NULL, 10);
          if ((v = find_obj_key(ins, ins_end, "duration"))) info->duration_sec = strtof(v, NULL);
        } else if (ci_has(an->ct, "TextEncodeQwenImage")) {
          if ((v = find_obj_key(ins, ins_end, "prompt")) && *v == '"') { const char *vp = v; grab_json_string(&vp, an->text, sizeof an->text); }
          else if ((v = find_obj_key(ins, ins_end, "text")) && *v == '"') { const char *vp = v; grab_json_string(&vp, an->text, sizeof an->text); }
          if ((v = find_obj_key(ins, ins_end, "negative_prompt")) && *v == '"') { const char *vp = v; grab_json_string(&vp, an->text2, sizeof an->text2); }
        } else if (ci_has(an->ct, "CLIPTextEncode") || ci_has(an->ct, "PrimitiveString") || ci_has(an->ct, "Prompt")) {
          if ((v = find_obj_key(ins, ins_end, "text")) && *v == '"') { const char *vp = v; grab_json_string(&vp, an->text, sizeof an->text); }
          else if ((v = find_obj_key(ins, ins_end, "value")) && *v == '"') { const char *vp = v; grab_json_string(&vp, an->text, sizeof an->text); }
          else if ((v = find_obj_key(ins, ins_end, "t5xxl")) && *v == '"') { const char *vp = v; grab_json_string(&vp, an->text, sizeof an->text); }
        }
        if (ci_has(an->ct, "CheckpointLoader") || ci_has(an->ct, "UNETLoader") || ci_has(an->ct, "DiffusionModel")) {
          v = find_obj_key(ins, ins_end, "ckpt_name");
          if (!v) v = find_obj_key(ins, ins_end, "unet_name");
          if (v && *v == '"') {
            char ck[192] = {0}; const char *vp = v; grab_json_string(&vp, ck, sizeof ck);
            if (is_better_checkpoint(ck, info->extracted_checkpoint, vault))
              snprintf(info->extracted_checkpoint, sizeof info->extracted_checkpoint, "%s", ck);
          }
        }
        if (ci_has(an->ct, "LoraLoader") && info->n_loras < 4) {
          if ((v = find_obj_key(ins, ins_end, "lora_name")) && *v == '"') {
            const char *vp = v;
            if (grab_json_string(&vp, info->extracted_loras[info->n_loras], sizeof info->extracted_loras[0]) == 0 &&
                info->extracted_loras[info->n_loras][0]) {
              const char *sv = find_obj_key(ins, ins_end, "strength_model");
              info->extracted_lora_strengths[info->n_loras] = sv ? strtof(sv, NULL) : 0.85f;
              info->n_loras++;
            }
          }
        }
        if (ci_has(an->ct, "EmptyAceStep") && ci_has(an->ct, "Audio")) {
          if ((v = find_obj_key(ins, ins_end, "seconds"))) info->duration_sec = strtof(v, NULL);
        } else if (ci_has(an->ct, "LatentVideo") || ci_has(an->ct, "ImageToVideoLatent")) {
          if ((v = find_obj_key(ins, ins_end, "width")))  info->width  = (int)strtol(v, NULL, 10);
          if ((v = find_obj_key(ins, ins_end, "height"))) info->height = (int)strtol(v, NULL, 10);
          if ((v = find_obj_key(ins, ins_end, "length"))) info->frames = (int)strtol(v, NULL, 10);
        } else if (ci_has(an->ct, "Empty") && ci_has(an->ct, "Latent")) {
          if ((v = find_obj_key(ins, ins_end, "width")))  info->width  = (int)strtol(v, NULL, 10);
          if ((v = find_obj_key(ins, ins_end, "height"))) info->height = (int)strtol(v, NULL, 10);
        }
        if (ci_has(an->ct, "Sampler") || ci_has(an->ct, "Scheduler") || ci_has(an->ct, "Guider") || ci_has(an->ct, "RandomNoise")) {
          if ((v = find_obj_key(ins, ins_end, "steps")))      info->steps = (int)strtol(v, NULL, 10);
          if ((v = find_obj_key(ins, ins_end, "seed")))       info->seed  = (uint32_t)strtoull(v, NULL, 10);
          if ((v = find_obj_key(ins, ins_end, "noise_seed"))) info->seed  = (uint32_t)strtoull(v, NULL, 10);
          if ((v = find_obj_key(ins, ins_end, "positive")) && *v == '[') {
            const char *vp = v + 1; grab_json_string(&vp, an->pos_ref, sizeof an->pos_ref);
            skip_json_ws(&vp); if (*vp == ',') { vp++; an->pos_slot = (int)strtol(vp, NULL, 10); }
          }
          if ((v = find_obj_key(ins, ins_end, "negative")) && *v == '[') {
            const char *vp = v + 1; grab_json_string(&vp, an->neg_ref, sizeof an->neg_ref);
            skip_json_ws(&vp); if (*vp == ',') { vp++; an->neg_slot = (int)strtol(vp, NULL, 10); }
          }
        }
      }
    }

    // Resolve positive & negative prompts via API sampler links first!
    for (int i = 0; i < n_ands; i++) {
      if (ands[i].pos_ref[0]) {
        for (int j = 0; j < n_ands; j++) {
          if (strcmp(ands[j].id, ands[i].pos_ref) == 0 && ands[j].text[0]) {
            snprintf(info->extracted_prompt, sizeof info->extracted_prompt, "%s", ands[j].text);
            break;
          }
        }
      }
      if (ands[i].neg_ref[0]) {
        for (int j = 0; j < n_ands; j++) {
          if (strcmp(ands[j].id, ands[i].neg_ref) == 0) {
            const char *nt = (ands[i].neg_slot == 1 && ands[j].text2[0]) ? ands[j].text2 : ands[j].text;
            if (nt[0]) snprintf(info->extracted_neg_prompt, sizeof info->extracted_neg_prompt, "%s", nt);
            break;
          }
        }
      }
    }
    // Fallback if API workflow didn't link sampler directly
    if (!info->extracted_prompt[0]) {
      for (int i = 0; i < n_ands; i++) {
        if (ands[i].text[0] && strlen(ands[i].text) > strlen(info->extracted_prompt)) {
          snprintf(info->extracted_prompt, sizeof info->extracted_prompt, "%s", ands[i].text);
        }
      }
    }
    free(ands);
  } else {
    // LiteGraph UI JSON v2: Parse top-level graph + ComfyUI 2026 definitions.subgraphs[] and flatten!
    lg_node_t *nodes = (lg_node_t *)calloc(LG_MAX_NODES, sizeof(lg_node_t));
    lg_link_t *links = (lg_link_t *)calloc(LG_MAX_LINKS, sizeof(lg_link_t));
    lg_subgraph_t *sgs = (lg_subgraph_t *)calloc(LG_MAX_SG, sizeof(lg_subgraph_t));
    int n_nodes = 0, n_links = 0, n_sgs = 0;

    if (nodes && links && sgs) {
      const char *top_nodes_e = top_nodes; skip_json_val(&top_nodes_e);
      const char *top_links_e = top_links; skip_json_val(&top_links_e);
      n_nodes = parse_lg_nodes(top_nodes, top_nodes_e, nodes, LG_MAX_NODES);
      n_links = parse_lg_links(top_links, top_links_e, links, LG_MAX_LINKS);

      // Parse ComfyUI 2026 definitions.subgraphs[] if present
      const char *defs = find_obj_key(root_s, root_e, "definitions");
      if (defs && *defs == '{') {
        const char *defs_e = defs; skip_json_val(&defs_e);
        const char *sg_arr = find_obj_key(defs, defs_e, "subgraphs");
        if (sg_arr && *sg_arr == '[') {
          const char *sg_arr_e = sg_arr; skip_json_val(&sg_arr_e);
          const char *sp = sg_arr + 1;
          while (sp < sg_arr_e && *sp != ']' && n_sgs < LG_MAX_SG) {
            skip_json_ws(&sp);
            if (*sp == ',') { sp++; continue; }
            if (*sp != '{') break;
            const char *ss = sp; skip_json_val(&sp); const char *se = sp;
            lg_subgraph_t *sg = &sgs[n_sgs];
            const char *idv = find_obj_key(ss, se, "id");
            if (idv && *idv == '"') { const char *iv = idv; grab_json_string(&iv, sg->uuid, sizeof sg->uuid); }
            const char *sins = find_obj_key(ss, se, "inputs");
            if (sins && *sins == '[') {
              const char *sins_e = sins; skip_json_val(&sins_e);
              const char *ip = sins + 1;
              while (ip < sins_e && *ip != ']' && sg->n_sg_in < 16) {
                skip_json_ws(&ip);
                if (*ip == ',') { ip++; continue; }
                if (*ip == '{') {
                  const char *is = ip; skip_json_val(&ip); const char *ie = ip;
                  const char *nm = find_obj_key(is, ie, "name");
                  if (nm && *nm == '"') {
                    const char *np2 = nm; grab_json_string(&np2, sg->in_names[sg->n_sg_in], sizeof sg->in_names[0]);
                  }
                  sg->n_sg_in++;
                } else skip_json_val(&ip);
              }
            }
            const char *sn = find_obj_key(ss, se, "nodes");
            const char *sl = find_obj_key(ss, se, "links");
            if (sn && *sn == '[') {
              const char *sn_e = sn; skip_json_val(&sn_e);
              sg->nodes = (lg_node_t *)calloc(128, sizeof(lg_node_t));
              if (sg->nodes) sg->n_nodes = parse_lg_nodes(sn, sn_e, sg->nodes, 128);
            }
            if (sl && *sl == '[') {
              const char *sl_e = sl; skip_json_val(&sl_e);
              sg->links = (lg_link_t *)calloc(512, sizeof(lg_link_t));
              if (sg->links) sg->n_links = parse_lg_links(sl, sl_e, sg->links, 512);
            }
            if (sg->uuid[0] && sg->n_nodes > 0) n_sgs++;
          }
        }
      }
      info->subgraph_count = n_sgs;

      // Flatten subgraph instances into main nodes[] & links[], rewiring inputNode(-10) to parent producers!
      int orig_n_nodes = n_nodes;
      int sg_inst_idx = 0;
      for (int i = 0; i < orig_n_nodes; i++) {
        const lg_subgraph_t *sg = NULL;
        for (int s = 0; s < n_sgs; s++) {
          if (strcmp(nodes[i].type, sgs[s].uuid) == 0) { sg = &sgs[s]; break; }
        }
        if (!sg) continue;
        sg_inst_idx++;
        int id_base   = sg_inst_idx * 10000;
        int link_base = sg_inst_idx * 100000;

        // Map subgraph input slot index k -> parent producer (from_node, from_slot)
        int parent_from_node[16], parent_from_slot[16];
        for (int k = 0; k < 16; k++) { parent_from_node[k] = -1; parent_from_slot[k] = 0; }
        for (int k = 0; k < sg->n_sg_in && k < 16; k++) {
          for (int pi = 0; pi < nodes[i].n_ins; pi++) {
            if (strcmp(nodes[i].in_names[pi], sg->in_names[k]) == 0) {
              const lg_link_t *plk = find_lg_link(links, n_links, nodes[i].in_link_ids[pi]);
              if (plk) {
                parent_from_node[k] = plk->from_node;
                parent_from_slot[k] = plk->from_slot;
              }
              break;
            }
          }
        }

        // Inline inner links (remapping node & link IDs; rewiring origin_id == -10 to parent producer!)
        for (int li = 0; li < sg->n_links && n_links < LG_MAX_LINKS; li++) {
          lg_link_t L = sg->links[li];
          L.id += link_base;
          if (L.from_node < 0) {
            if (L.from_slot >= 0 && L.from_slot < 16 && parent_from_node[L.from_slot] >= 0) {
              L.from_node = parent_from_node[L.from_slot];
              L.from_slot = parent_from_slot[L.from_slot];
            }
          } else {
            L.from_node += id_base;
          }
          if (L.to_node >= 0) L.to_node += id_base;
          links[n_links++] = L;
        }
        // Inline inner nodes
        for (int ni = 0; ni < sg->n_nodes && n_nodes < LG_MAX_NODES; ni++) {
          lg_node_t nd = sg->nodes[ni];
          nd.id += id_base;
          for (int ii = 0; ii < nd.n_ins; ii++) nd.in_link_ids[ii] += link_base;
          nodes[n_nodes++] = nd;
        }
      }

      // Pass 1: Detect modality, checkpoints, LoRAs, dims/frames/duration/bpm/steps/seed, LoadImage
      for (int i = 0; i < n_nodes; i++) {
        const lg_node_t *nd = &nodes[i];
        const char *nt = nd->type;
        detect_modality_from_type(nt, &has_music, &has_3d, &has_video);

        if (ci_has(nt, "LoadImage")) {
          info->needs_image_input = 1;
          if (nd->n_wstrs > 0 && !info->input_image_hint[0])
            snprintf(info->input_image_hint, sizeof info->input_image_hint, "%s", nd->wstrs[0]);
        }
        if ((ci_has(nt, "CheckpointLoader") || ci_has(nt, "UNETLoader") || ci_has(nt, "DiffusionModel")) && nd->n_wstrs > 0) {
          if (is_better_checkpoint(nd->wstrs[0], info->extracted_checkpoint, vault))
            snprintf(info->extracted_checkpoint, sizeof info->extracted_checkpoint, "%s", nd->wstrs[0]);
        }
        if ((ci_has(nt, "LoraLoader") || ci_has(nt, "LoraLoaderModelOnly")) && nd->n_wstrs > 0 && info->n_loras < 4) {
          snprintf(info->extracted_loras[info->n_loras], sizeof info->extracted_loras[0], "%s", nd->wstrs[0]);
          info->extracted_lora_strengths[info->n_loras] = (nd->n_wnums > 0) ? (float)nd->wnums[0] : 0.85f;
          info->n_loras++;
        }
        for (int ri = 0; ri < nd->n_rg_loras && info->n_loras < 4; ri++) {
          snprintf(info->extracted_loras[info->n_loras], sizeof info->extracted_loras[0], "%s", nd->rg_loras[ri]);
          info->extracted_lora_strengths[info->n_loras] = nd->rg_strengths[ri];
          info->n_loras++;
        }
        if (ci_has(nt, "TextEncodeAceStepAudio")) {
          if (nd->n_wstrs > 0 && nd->wstrs[0][0])
            snprintf(info->extracted_prompt, sizeof info->extracted_prompt, "%s", nd->wstrs[0]);
          if (nd->n_wstrs > 1 && nd->wstrs[1][0])
            snprintf(info->extracted_lyrics, sizeof info->extracted_lyrics, "%s", nd->wstrs[1]);
          if (nd->n_wnums >= 2 && nd->wnums[1] >= 40 && nd->wnums[1] <= 240)
            info->bpm = (int)nd->wnums[1];
          if (nd->n_wnums >= 3 && nd->wnums[2] >= 1 && nd->wnums[2] <= 600)
            info->duration_sec = (float)nd->wnums[2];
        }
        if (ci_has(nt, "EmptyAceStep") && ci_has(nt, "Audio") && nd->n_wnums >= 1) {
          info->duration_sec = (float)nd->wnums[0];
        } else if ((ci_has(nt, "Wan22ImageToVideoLatent") || ci_has(nt, "LatentVideo")) && nd->n_wnums >= 2) {
          info->width  = (int)nd->wnums[0];
          info->height = (int)nd->wnums[1];
          if (nd->n_wnums >= 3) info->frames = (int)nd->wnums[2];
        } else if (ci_has(nt, "Empty") && ci_has(nt, "Latent") && !ci_has(nt, "Structure") && nd->n_wnums >= 2) {
          info->width  = (int)nd->wnums[0];
          info->height = (int)nd->wnums[1];
        }
        if (ci_has(nt, "KSampler") && !ci_has(nt, "Select") && nd->n_wnums >= 2) {
          info->seed  = (uint32_t)nd->wnums[0];
          info->steps = (int)nd->wnums[1];
        }
        if ((ci_has(nt, "Flux2Scheduler") || ci_has(nt, "BasicScheduler")) && nd->n_wnums >= 1) {
          int st_v = (int)nd->wnums[0];
          if (st_v > 0 && st_v <= 100) info->steps = st_v;
          if (ci_has(nt, "Flux2Scheduler") && nd->n_wnums >= 3) {
            info->width = (int)nd->wnums[1];
            info->height = (int)nd->wnums[2];
          }
        }
        if (ci_has(nt, "RandomNoise") && nd->n_wnums >= 1 && nd->wnums[0] > 0) {
          info->seed = (uint32_t)((uint64_t)nd->wnums[0] & 0xFFFFFFFFu);
        }
      }

      // Pass 2: Backward link tracing from sampler/guider positive & negative inputs!
      for (int i = 0; i < n_nodes; i++) {
        const lg_node_t *nd = &nodes[i];
        const char *nt = nd->type;
        if (!ci_has(nt, "KSampler") && !ci_has(nt, "CFGGuider") && !ci_has(nt, "BasicGuider") && !ci_has(nt, "SamplerCustom"))
          continue;
        for (int ii = 0; ii < nd->n_ins; ii++) {
          const lg_link_t *lk = find_lg_link(links, n_links, nd->in_link_ids[ii]);
          if (!lk) continue;
          if (strcmp(nd->in_names[ii], "positive") == 0 || strcmp(nd->in_names[ii], "conditioning") == 0) {
            char pbuf[768] = {0};
            if (trace_lg_prompt(nodes, n_nodes, links, n_links, lk->from_node, lk->from_slot, pbuf, sizeof pbuf, 0) && pbuf[0]) {
              snprintf(info->extracted_prompt, sizeof info->extracted_prompt, "%s", pbuf);
            }
          } else if (strcmp(nd->in_names[ii], "negative") == 0) {
            char nbuf[768] = {0};
            if (trace_lg_prompt(nodes, n_nodes, links, n_links, lk->from_node, lk->from_slot, nbuf, sizeof nbuf, 0) && nbuf[0]) {
              snprintf(info->extracted_neg_prompt, sizeof info->extracted_neg_prompt, "%s", nbuf);
            }
          }
        }
      }

      // Pass 3: Fallback if no sampler link carried text (e.g. standalone encoders or PrimitiveStringMultiline)
      if (!info->extracted_prompt[0]) {
        for (int i = 0; i < n_nodes; i++) {
          const lg_node_t *nd = &nodes[i];
          const char *nt = nd->type;
          if (ci_has(nt, "MarkdownNote") || strcmp(nt, "Note") == 0) continue;
          if ((ci_has(nt, "TextEncodeQwenImage") || ci_has(nt, "CLIPTextEncode") ||
               ci_has(nt, "PrimitiveString") || ci_has(nt, "Prompt")) && nd->n_wstrs > 0 && nd->wstrs[0][0]) {
            if (strlen(nd->wstrs[0]) > strlen(info->extracted_prompt))
              snprintf(info->extracted_prompt, sizeof info->extracted_prompt, "%s", nd->wstrs[0]);
            if (ci_has(nt, "TextEncodeQwenImage") && nd->n_wstrs > 1 && !info->extracted_neg_prompt[0])
              snprintf(info->extracted_neg_prompt, sizeof info->extracted_neg_prompt, "%s", nd->wstrs[1]);
          }
        }
      }

      // Emit flattened canonical ComfyUI API JSON!
      int out_n = snprintf(out_api_json, out_api_cap, "{");
      for (int i = 0; i < n_nodes && (size_t)out_n + 512 < out_api_cap; i++) {
        const lg_node_t *nd = &nodes[i];
        // Skip subgraph wrapper shells now that their inner nodes are inlined!
        int is_sg_shell = 0;
        for (int s = 0; s < n_sgs; s++) if (strcmp(nd->type, sgs[s].uuid) == 0) { is_sg_shell = 1; break; }
        if (is_sg_shell) continue;

        char ect[192];
        json_esc(nd->type, ect, sizeof ect);
        out_n += snprintf(out_api_json + out_n, out_api_cap - (size_t)out_n,
            "%s\"%d\":{\"class_type\":\"%s\",\"inputs\":{",
            info->node_count ? "," : "", nd->id, ect);
        int first_in = 1;
        for (int k = 0; k < nd->n_ins; k++) {
          const lg_link_t *lk = find_lg_link(links, n_links, nd->in_link_ids[k]);
          if (lk && lk->from_node >= 0) {
            out_n += snprintf(out_api_json + out_n, out_api_cap - (size_t)out_n,
                "%s\"%s\":[\"%d\",%d]", first_in ? "" : ",", nd->in_names[k], lk->from_node, lk->from_slot);
            first_in = 0;
          }
        }
        if (nd->n_wstrs > 0 && nd->wstrs[0][0] && !ci_has(nd->type, "MarkdownNote")) {
          char ew[1024];
          json_esc(nd->wstrs[0], ew, sizeof ew);
          out_n += snprintf(out_api_json + out_n, out_api_cap - (size_t)out_n,
              "%s\"value\":\"%s\"", first_in ? "" : ",", ew);
        }
        out_n += snprintf(out_api_json + out_n, out_api_cap - (size_t)out_n, "}}");
        info->node_count++;
      }
      snprintf(out_api_json + out_n, out_api_cap - (size_t)out_n, "}");
    }

    for (int s = 0; s < n_sgs; s++) { free(sgs[s].nodes); free(sgs[s].links); }
    free(sgs); free(nodes); free(links);
  }

  // Final modality decision (priority: music > 3d > video > image)
  if (has_music)      snprintf(info->modality, sizeof info->modality, "music");
  else if (has_3d)    snprintf(info->modality, sizeof info->modality, "3d");
  else if (has_video) snprintf(info->modality, sizeof info->modality, "video");
  else                snprintf(info->modality, sizeof info->modality, "image");

  // Resolve checkpoints and LoRAs across ALL local + network storage roots (including model-family aliases!)
  if (vault) {
    if (info->extracted_checkpoint[0]) {
      const ec_storage_asset_t *a = ec_storage_find(vault, EC_ASSET_CHECKPOINT, info->extracted_checkpoint);
      if (a) snprintf(info->resolved_checkpoint, sizeof info->resolved_checkpoint, "%s", a->path);
    }
    for (int i = 0; i < info->n_loras; i++) {
      const ec_storage_asset_t *a = ec_storage_find(vault, EC_ASSET_LORA, info->extracted_loras[i]);
      if (a) snprintf(info->resolved_loras[i], sizeof info->resolved_loras[0], "%s", a->path);
    }
  }
  info->ok = (info->node_count > 0) ? 1 : 0;
  if (!info->ok) snprintf(info->error, sizeof info->error, "no valid ComfyUI nodes found");
  return info->ok ? 0 : -1;
}
