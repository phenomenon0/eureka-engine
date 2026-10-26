# VENDOR: the in-tab runtime the web app loads next to itself (APP-FOLD 2/n)

Copied verbatim from the served site the served WEBGPU-E2E site bundle (the bundle live on the box since 2026-10-16 12:47, sha `dbd79976…`).
- `engine/` + `emit/`: WEB-ALLMODAL webapp's shipped engine (byte-equal per WEB-ALLMODAL/REPORT.md). The Mac test pages (smoke*.html, kcheck.html, drive-turn.js) are left out, as the box bundle leaves them out. `engine/app-worker.js` is WEBGPU-E2E's own (folded in 1/n).
- `trunk/`: bendweb-showcase branch `web-trunk-port` `e5a3783` `webapp/trunk/`. trunk_engine.js and trunk_kernels.js are equal; trunk_session.js carries WEBGPU-E2E P3's relative-path edit.
- Not here: `models/` (weights; HF or local symlinks), `models/qwen35-tokenizer.bin` (7.5 MB, no HF home yet), `senses/vendor/` (`scripts/senses/vendor.sh`).
- Owner of record stays bendweb. Replace these as a set and re-run the e2e matrix.

```
0b5185f77335186e094c431aa1fae6202c4196ada113be487d5a44a74c5c92c5  engine/app-worker.js
a47f970d27e3c8664043a98182a586632229b4d9dd7b742d7b202341ed7580e8  engine/bend-worker.js
edf92ae2f4cd5e9dcd88ab4bb984a28b5a52851087aeedbbc88e6ccc56962187  engine/gguf.js
476e1b2e78481f558debd66230b76ce3a7cddc7880132f4eb683c9fd371c7153  engine/ggufpack.js
7e8caf15da29991dfdc1b039c7d2b14953c49109afc896d8bd2ba84d22f1a840  engine/q35_engine.js
a8a7864f9d6344dfaa91e3bafa135d8e95680c051e2011887796cee2686dd5b5  engine/q35_plan.js
01b30a6bf900a535c4b35ab93a99816ffe5e6a3a3a9624d969c694f1b9179392  engine/q35_split.js
f79be8286ae7c5142950f97653fbab4ec018fbfcd2a5696627c4080cb7e73d67  engine/q35_step.js
7d0d34a7423fc18920a130463964ec5afed895e970afb5d4a0c83f7f3b098477  engine/tok.js
7b8a0f3e5028d8943359cf3c5445ab24dae664489d08012716a2054a9ab05703  engine/webgpu.js
b000ed7427d82eabb85d9b35e3dc518ffb820ca00a3a4596e6c3cbf90ec2780d  engine/webgpu_graph.js
7f0e3092c07fda8883271e32906c467a682e200b9c527e5ea05ca5f06f0d0c0d  engine/wire.js
b366c7eed7d2b69bdba56bbf056fcf3568eb864bd9058078dfeed706cb2b8e1d  engine/tok.wasm
e32941f3da34d6b05e96309695653361d9e3e9a3fcb35fa82152349e89e9082e  emit/k_attention.wgsl
f31b9b940ee3cb68b2e73ce17b8b200bdc7b6fa5c3f364d3d10ca16f93144484  emit/k_embed.wgsl
8824860b149c8a8f519ef9aba497ae33c956cb4d3088f90802eec537686f8d3e  emit/k_embed_q4k.wgsl
c3d6379f695822915a805005b6ae1c8d2927e61acd4c15eda67d4c914dc0e43f  emit/k_fuse_t.wgsl
729d515de2e55f78b05a2664fced1e94eb80b67a40250384b3bc9ccbf3369a39  emit/k_gemv.wgsl
53e15a2e9ee4de45f92eed2ffc4b7f4f45d34ea1e1128d17af30e5a7cdb12fc5  emit/k_gemv_fastE.wgsl
a87a1fb03d88eb4b93a172e3f1c48943883056c168d8c0c4fbcb63b90a326402  emit/k_gemv_fastEI.wgsl
cfdba20450c80e905157f3974f820db22bbe77eb82278fba2b3159b837865c43  emit/k_gemv_turbo.wgsl
29e0b497c1d836b491e0968feb435ab4362298a1038e1eed837edd38145a0271  emit/k_gemv_turbo_kev.wgsl
7b8c69d7eaa66e52c6b65a7e929f8ab2eaaf8bcde66d0b10b1a5dfbc59e44cd6  emit/k_gemv_turbo_sg.wgsl
35a5e51e09bda1c80ecbfbca3f2fffad3147eea5a641411f4415ff1d0d03fc6a  emit/k_gemv_turbo_sg_kev.wgsl
d0c1f93741ab0f22e78c766d29a7e6d25c6434e5794622b0cc47f309858e990d  emit/k_norms.wgsl
59c20989aa3b6b653859cbfc594160a87afb1d1cc636060d8618a946aa1b34f6  emit/k_norms_t.wgsl
86de48df9876d846d8001488512f8706d481e3369bb2e3a5fb87e45fd687737c  emit/k_recurrent.wgsl
5491de5734b141cef85342aeae33e4731f40774ceefa89955a5dac8df7be79df  emit/k_softmax.wgsl
419bc29b3b68401b0bc9dcb207bc715476fbc32207e0a24a7aa0ee3d8fa75fcd  emit/k_split.wgsl
d644b69af2a94f0c32a53dc9d184209d527cfafc5febd37c800a2ece38e451ec  emit/k_step.wgsl
adffd22f8dd842b530c03a4aa7c9d8e959802e3c63f9115a2e4c985d0fcd21a9  emit/k_top_t.wgsl
1c434b8bcb69effce7b35acb20351eb7a05cefe27ca46a1f7c63c89580b12178  trunk/trunk_engine.js
4f9524ee3d7293381bcd5991450c3b37d4ac4b0bf3b92d0db425fcbc0c7e298d  trunk/trunk_kernels.js
bb2c4b86f4f16567e4827cb03237977b385d812bcbffcb9d4675ef132ca68cdf  trunk/trunk_session.js
```
