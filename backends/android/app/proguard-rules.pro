# APKSIGN: onnxruntime's JNI side looks up its Java classes by name and the AAR ships no consumer rules.
-keep class ai.onnxruntime.** { *; }
