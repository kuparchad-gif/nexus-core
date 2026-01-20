
param(
  [string]$Url = "http://localhost:1234",
  [string]$Model = "qwen/qwen3-4b-thinking-2507"
)

$body = @{
  model = $Model
  messages = @(@{ role="user"; content='Return only {"ok":true}.' })
  temperature = 0
  max_tokens = -1
  stream = $false
} | ConvertTo-Json -Depth 5

Invoke-RestMethod -Uri ($Url.TrimEnd('/') + "/v1/chat/completions") -Method Post -ContentType 'application/json' -Body $body
