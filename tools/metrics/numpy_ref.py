# numpy reference values for test_metrics.c's percentile checks (linear method)
import numpy as np
cases = {
    "1..4 p50/p90": (np.arange(1, 5), [50, 90]),
    "1..100 p50/p95/p99": (np.arange(1, 101), [50, 95, 99]),
    "skew p50/p90": (np.array([0.010, 0.012, 0.013, 0.020, 0.100]), [50, 90]),
    "itl {3x 1/64, 1/32, 1/2} p50/p90": (np.array([.015625] * 3 + [.03125, .5]), [50, 90]),
    "window 477..1500 ms p50/p99": (np.arange(477, 1501) / 1000, [50, 99]),
}
for k, (a, qs) in cases.items():
    print(k, [float(np.percentile(a, q, method="linear")) for q in qs])
