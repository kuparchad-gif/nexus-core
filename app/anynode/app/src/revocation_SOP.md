# SOP: Revocation
1. Receive revoke request with subject identifier(s).
2. Create tombstone id; enqueue purge in memory/archiver.
3. On completion, append audit receipt with counts + hashes.
4. Negative verification: confirm lookups return no hits.
5. Notify requester with minimally sufficient proof.
