# Compactify Gateway (Nexus shrinking module)

This is a drop-in **shrinking gateway** for Nexus. Models pass through this service when imported:
it **compresses (CompactifAI-inspired), converts to GGUF, and quantizes** to your target alias (e.g. `7b.gguf`).

## Add to `podman-compose.yaml`

```yaml
  compactify:
    build: ./cognikubes/compactify
    environment:
      WORKDIR: /work
      MODELDIR: /models
    volumes:
      - ./models:/models
    ports:
      - "8815:8815"
    profiles: ["base"]
    depends_on: ["nats"]
```

## Build & run

```powershell
# From C:\Projects\Stacks\nexus-metatron
podman compose -p nexus -f .\podman-compose.yaml up -d compactify
Invoke-WebRequest http://localhost:8815/alive -UseBasicParsing | Select-Object -Expand Content
```

## Submit a shrink job

```powershell
# HF example (Qwen3-8B → firmware alias 7b with Q3_K_M)
.\scripts\Invoke-Compactify.ps1 -SourceKind hf -Src "Qwen/Qwen2.5-8B-Instruct" -Alias 7b -Quant Q3_K_M

# HF example (Qwen3-14B → inference alias 14b with Q4_K_M)
.\scripts\Invoke-Compactify.ps1 -SourceKind hf -Src "Qwen/Qwen2.5-14B-Instruct" -Alias 14b -Quant Q4_K_M

# Local folder example
.\scripts\Invoke-Compactify.ps1 -SourceKind local -Src "models\src\qwen3-8b-instruct" -Alias 7b -Quant Q3_K_M
```

Outputs land in `models\{alias}.gguf`, ready for your existing `llama-7b` / `llama-14b` services.