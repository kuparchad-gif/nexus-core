[MODE: ACCEPTANCE]
Return JSON only:
{
"smoke": ["curl http://localhost:<port>/alive == 200", "..."],
"post_checks": ["compose config OK", "no new open ports found"],
"rollback": ["remove overlay compose", "delete added services folder"]
}