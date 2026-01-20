# SOP: Route Change
1. Draft reason and scope (prefixes).
2. Lillith signs `intent.route.update`.
3. Core validates signature + reason; applies.
4. Core writes audit receipt; `/audit/verify` must remain valid.
5. Rollback plan documented before change.
