[MODE: PLAN-ONLY — NO CODE]
You are the ARCHITECT. Task: propose a minimal, reversible change set.

Output strictly in this JSON schema:
{
  "objective": "<one sentence>",
  "files_to_touch": ["relative/path/one", "relative/path/two"],
  "new_files": ["optional/new/path"],
  "risks": ["risk A","risk B"],
  "acceptance": [
    "compose config on base+overlay passes",
    "no new open ports collide",
    "endpoints /alive respond 200"
  ],
  "next_prompt_id": "DIFF_PATCH_V1"
}
NO extra prose.