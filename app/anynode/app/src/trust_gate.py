The `trust_gate` module is responsible for issuing tokens, which are used for authentication and authorization within the system. The `issue_token` function creates a JSON Web Token (JWT) with the provided actor ID, scope, and reason, using HS256 algorithm for signing.

The token's header contains the type of token and the algorithm used for signing. The payload includes the subject (actor ID), scope, reason, issuance time, and expiration time. The signature is a base64 encoded HMAC-SHA256 hash of the encoded header and payload, using a secret key.

The SECRET environment variable is used as the secret for signing the tokens. If not set, it defaults to "CHANGE_ME". The TTL_SECONDS environment variable sets the expiration time of the token; if not set, it defaults to 120 seconds.
