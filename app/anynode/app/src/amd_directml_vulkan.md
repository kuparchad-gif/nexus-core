# AMD (Windows) Local Inference Options

Your GPU stack: **AMD Radeon RX 6900 XT** + iGPU. On Windows:
- Prefer **ONNX Runtime (DirectML)** for BERT/TinyLlama and small vision models.
- Or use **llama.cpp** built with **Vulkan** for TinyLlama/phi/llama-variants.

## ONNX Runtime DirectML
```powershell
pip install onnxruntime-directml onnx transformers optimum
# Example: export a TinyLlama/BERT model to ONNX and run with ORT (DML backend)
# Log errors to Loki
$env:LOKI_ENDPOINT="http://loki:3100"
python -m optimum.exporters.onnx --model tinyllama-1.1b-chat tinyllama_onnx/
if ($LASTEXITCODE -ne 0) {
    Invoke-RestMethod -Uri "$env:LOKI_ENDPOINT/loki/api/v1/push" -Method Post -Body @{
        streams = @(@{
            stream = @{job="onnx_export"}
            values = @(@( [string](Get-Date).ToUniversalTime().Ticks, "ONNX export failed: exit code $LASTEXITCODE" ))
        })
    } | Out-Null
}