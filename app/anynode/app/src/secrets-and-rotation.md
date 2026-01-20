# Secrets & Password Rotation

- Initial admin user: `admin` (set a bcrypt hash in env `ADMIN_PASSWORD_HASH`).
- Rotation script: `scripts/Initialize-Passwords.ps1` (forces change on first login by writing a one-time token file).
- All services read secrets from environment-first, then `docs/CONFIG/.env.local` (never commit real secrets).
- For Windows, you can store secrets securely via:
  - `ConvertTo-SecureString` + `Export-Clixml` for DPAPI-bound secure files per-machine.
  - Or use Windows Credential Manager.
- On first boot, scripts will detect placeholders and prompt/rotate automatically.
