/**
 * Lightweight ToolSwitch shim:
 * - If tools provided and model lacks native function-calling, instruct model to emit {"call_tool":{...}} JSON.
 * - Parse strict JSON blocks, execute allowed tools via callback, loop once (simple stage-1).
 * - High-risk tools must be gated upstream by Viren (this shim only calls provided callback).
 */
export async function toolSwitch({messages, tools, callTool, modelCall}) {
  if(!tools || tools.length===0) {
    return await modelCall(messages); // no tools declared
  }

  // 1) Ask model whether to call a tool by prompting a deterministic system rule
  const sys = {
    role: "system",
    content:
`You may have access to tools. ONLY if a tool is needed, output a single JSON object on one line:
{"call_tool":{"name":"<toolName>","arguments":{...}}}
Otherwise, reply normally. Do not add extra text.`
  };
  const first = await modelCall([sys, ...messages]);
  const text = (first?.choices?.[0]?.message?.content || "").trim();

  let call;
  try {
    if(text.startsWith("{") && text.endsWith("}")){
      const obj = JSON.parse(text);
      call = obj?.call_tool;
    }
  } catch { /* not a tool call */ }

  if(!call) return first; // normal answer

  // Validate requested tool
  const tdef = tools.find(t => t.function?.name === call.name);
  if(!tdef) {
    return { choices:[{ message:{ role:"assistant", content:`Tool "${call.name}" not available.`}}]};
  }

  // 2) Execute tool via provided callback (Viren-gated upstream)
  const toolResult = await callTool(call.name, call.arguments || {});

  // 3) Give result back to model to produce final answer
  const toolMsg = { role:"tool", name:call.name, content: JSON.stringify(toolResult) };
  const final = await modelCall([ ...messages, {role:"assistant", content:text}, toolMsg ]);
  return final;
}
