# Git LFS Helpers — Quick Reference (2025-08-25)

## Typical fix after a rejected LFS push
```powershell
cd C:\Projects\Stacks\nexus-metatron
git lfs fetch --all
git lfs fsck
git lfs prune
git push
```

## Stop tracking a single file in LFS (future only)
```powershell
git lfs untrack "apps/xtts/examples/male.wav"
git add .gitattributes
git rm --cached "apps/xtts/examples/male.wav"
git commit -m "chore(lfs): stop tracking male.wav via LFS"
git push
```

> This *does not* rewrite history. Existing commits keep their LFS pointer for that file.
> New commits store it as normal Git (not recommended for large files).

## Fully remove from LFS (rewrite history)
```powershell
git lfs migrate export --include="apps/xtts/examples/male.wav"
git push --force-with-lease --all
git push --force-with-lease --tags
```

**Warning:** This rewrites history across all branches/tags. Coordinate with any collaborators.

## Normalize your policy
Keep `.gitattributes` checked in with sane defaults so large assets auto‑use LFS and code stays text.
