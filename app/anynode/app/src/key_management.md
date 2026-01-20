# Key Management (30-Year Horizon)
- Rotate HMAC/shared secrets every 90 days; asymmetric keys yearly.
- 5-year key ceremony: re-derive root, re-encrypt state-at-rest.
- Crypto agility: plan re-hash/re-sign when algorithms deprecate.
- Break-glass: M-of-N approval, maximum 24h validity, fully audited.
