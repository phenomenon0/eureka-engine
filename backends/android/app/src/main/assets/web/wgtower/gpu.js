// gpu.js: the adapter + device the tower pages use. Headless Chrome's GPU process can come up after the page's first
// requestAdapter (null), so ask again a few times before giving up.
export async function gpu_device() {
  let adapter = null;
  for (let i = 0; i < 20 && !adapter; i++) { adapter = await navigator.gpu?.requestAdapter({ powerPreference: "high-performance" }); if (!adapter) await new Promise((r) => setTimeout(r, 500)); }
  if (!adapter) throw new Error("no WebGPU adapter");
  const L = adapter.limits, sg = adapter.features.has("subgroups");
  const device = await adapter.requestDevice({ requiredFeatures: sg ? ["subgroups"] : [], requiredLimits: { maxStorageBufferBindingSize: L.maxStorageBufferBindingSize,
    maxBufferSize: L.maxBufferSize, maxComputeWorkgroupStorageSize: L.maxComputeWorkgroupStorageSize, maxStorageBuffersPerShaderStage: L.maxStorageBuffersPerShaderStage } });
  return { adapter, device, facts: { vendor: adapter.info.vendor, architecture: adapter.info.architecture, description: adapter.info.description,
    maxStorageBufferBindingSize: device.limits.maxStorageBufferBindingSize, maxBufferSize: device.limits.maxBufferSize,
    maxComputeInvocationsPerWorkgroup: device.limits.maxComputeInvocationsPerWorkgroup, maxComputeWorkgroupStorageSize: device.limits.maxComputeWorkgroupStorageSize } };
}
