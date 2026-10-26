# fuzz.awk: N deterministic tokenizer fuzz strings, one per line, "\n" and "\t" written as escapes (t2.sh unescapes).
# awk -v N=400 -v SEED=7 -f fuzz.awk
BEGIN {
  srand(SEED)
  split("a b c x y z e i o u T H Q Z 0 1 2 7 9", al, " ")
  split("the of and to in is it you that he was for on are with as I his they be at one have this from", wd, " ")
  split(". , ; : ! ? - _ ( ) [ ] { } < > / \\ | @ # $ % ^ & * + = ~ ` \" '", pu, " ")
  split("'s 're 've 'll 'd 'm 't 'S 'RE", ct, " ")
  split("é ß ñ ü ø Ж я 中 文 日本 語 한국 ي ع ש ह ि ॐ € £ ¥ © ™ … — – « » ́ ̃ 𝔘 𝟙 😀 👍🏽 🇫🇷 ❤️ ✓ ½ ² Ⅻ", un, " ")
  split("<|im_start|> <|im_end|> <|endoftext|> <think> </think> <tool_call> <|vision_start|> <|im_start|>user", sp, " ")
  split(" |  |   |\\t|\\n|\\n\\n| \\n|\\r\\n|\\t\\t| \\t |    ", wsp, "|")
  for (n = 0; n < N; n++) {
    s = ""; k = 1 + int(rand() * 14)
    for (j = 0; j < k; j++) {
      r = rand()
      if (r < .25) s = s wd[1 + int(rand() * 25)]
      else if (r < .35) s = s pu[1 + int(rand() * 31)]
      else if (r < .42) s = s ct[1 + int(rand() * 9)]
      else if (r < .55) s = s un[1 + int(rand() * 43)]
      else if (r < .60) s = s sp[1 + int(rand() * 8)]
      else if (r < .70) s = s int(rand() * 10 ^ (1 + int(rand() * 9)))
      else if (r < .78) for (m = int(rand() * 6); m >= 0; m--) s = s al[1 + int(rand() * 19)]
      else if (r < .80) s = s toupper(wd[1 + int(rand() * 25)])
      s = s wsp[1 + int(rand() * 11)]
    }
    print s
  }
}
