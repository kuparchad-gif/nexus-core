In server/src/index.ts add:

  import council from './council'
  import fsapi from './fs'

After your auth/cors and before app.listen(), mount:

  app.use('/api/council', council)
  app.use('/api/fs', fsapi)

Env to add:
  FS_ROOT=/workspace         # where host files are mounted inside api container
  OPENAI_BASE_URL=...        # optional for external provider
  OPENAI_API_KEY=...         # optional for external provider
