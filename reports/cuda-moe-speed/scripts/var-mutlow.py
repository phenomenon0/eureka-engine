old = "        pl[q] = e < ne ? pv[e] : -1.0f;\n"
assert s.count(old) == 1
s = s.replace(old, "        pl[q] = e < ne ? 1.0f - pv[e] : -1.0f;  // MUTANT: the k lowest probabilities\n")
old = "          wt[j] = bv;\n        }\n        if (bix == mi) {"
assert s.count(old) == 1, s.count(old)
s = s.replace(old, "          wt[j] = pv[bix];\n        }\n        if (bix == mi) {")
