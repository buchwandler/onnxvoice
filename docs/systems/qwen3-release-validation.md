# Qwen3 real-model CPU validation

This record captures ONNXVoice's four-profile release gate. It verifies that each pinned WaveKat model executes through ONNX Runtime's CPU provider and returns audio satisfying the integration test's runtime assertions. It is not a numerical parity or perceptual-quality assessment.

## Environment

- Validation date: 2026-10-15
- OS: Linux x86_64 under WSL2, kernel `6.18.33.2-microsoft-standard-WSL2`
- CPU: Intel Core Ultra 7 355, 8 logical CPUs exposed to WSL
- RAM: 19 GiB reported by WSL; peak process memory was not measured
- Python: 3.13.15
- ONNX Runtime: 1.31.0
- Available providers: `AzureExecutionProvider`, `CPUExecutionProvider`; the test asserted `CPUExecutionProvider` was active for every profile

## Model sources

The bundle catalog pins these upstream source revisions:

| Bundle              | Repository                                | Revision                                   |
| ------------------- | ----------------------------------------- | ------------------------------------------ |
| `base-0.6b`         | `wavekat/Qwen3-TTS-0.6B-Base-ONNX`        | `177cd39e2ddab0d4d7ce5152a7abc2d382621158` |
| `voice-design-1.7b` | `wavekat/Qwen3-TTS-1.7B-VoiceDesign-ONNX` | `62c7863a68800d72bcee4f2931148c441571ace7` |

## Gate and results

The four pinned profiles were installed in the dedicated `.qwen3-release-cache`. The fail-closed command was:

```bash
ONNXVOICE_REQUIRE_QWEN3_REAL=1 \
ONNXVOICE_TEST_CACHE=.qwen3-release-cache \
.venv-qwen3-release/bin/python -m pytest -vv -m 'integration and resource_heavy' tests/test_qwen3_integration.py
```

| Profile               | Result |
| --------------------- | ------ |
| VoiceDesign 1.7B FP32 | Passed |
| VoiceDesign 1.7B INT4 | Passed |
| Base 0.6B FP32        | Passed |
| Base 0.6B INT4        | Passed |

Pytest reported **4 passed, 4 deselected, 0 skipped, 0 failed** in **609.63 seconds**. The four deselected tests are unit tests without the selected integration/resource-heavy markers. Each executed profile returned non-empty finite, non-silent mono `float32` audio at 24 kHz, with active `CPUExecutionProvider`.

The real models exposed a cache-output naming mismatch during initial execution: both `talker_decode` and `code_predictor` export `present_keys` and `present_values`. ONNXVoice's graph contract was corrected and offline regression coverage was added for both components.

No output-reference comparison, listening assessment, peak-memory measurement, latency, or real-time-factor measurement was performed. These results make no parity, perceptual-equivalence, performance, or non-CPU platform claim.
