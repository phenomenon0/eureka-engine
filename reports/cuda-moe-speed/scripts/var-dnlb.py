s = s.replace("#define MOE_DN_NR 4\n#define MOE_DN_US 1\n", "#define MOE_DN_NR 4\n#define MOE_DN_US 1\n")
s = s.replace("extern \"C\" __global__ void __launch_bounds__(128) llb_moe_dn(", "extern \"C\" __global__ void __launch_bounds__(128, 8) llb_moe_dn(")
assert "__launch_bounds__(128, 8) llb_moe_dn" in s
