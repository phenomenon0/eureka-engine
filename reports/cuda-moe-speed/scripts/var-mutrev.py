old = "        acc[j] = moe_g(moe_g(acc[j], w0, xv[SX ? 0 : j][u][0]), w1, xv[SX ? 0 : j][u][1]);\n"
assert s.count(old) == 1
s = s.replace(old, "        acc[j] = SX ? moe_g(moe_g(acc[j], w1, xv[0][u][1]), w0, xv[0][u][0]) : moe_g(moe_g(acc[j], w0, xv[SX ? 0 : j][u][0]), w1, xv[SX ? 0 : j][u][1]);  // MUTANT: gate/up groups reversed in a unit\n")
