const BASE = "http://localhost:8787/v1";

chrome.runtime.onInstalled.addListener(()=>{
  chrome.contextMenus.create({id:"ask_lilith", title:"Ask Lilith (Gateway)", contexts:["selection"]});
});

chrome.contextMenus.onClicked.addListener(async(info, tab)=>{
  if(info.menuItemId !== "ask_lilith") return;
  const [{result}] = await chrome.scripting.executeScript({
    target: {tabId: tab.id},
    func: () => window.getSelection()?.toString() || ""
  });
  if(!result) return;
  try{
    const r = await fetch(BASE + "/chat/completions", {
      method:"POST",
      headers:{ "content-type":"application/json" },
      body: JSON.stringify({
        model: "llama-3.2-3b-instruct",
        messages: [{role:"user", content: `From: ${location.href}\n\n` + result}]
      })
    });
    const j = await r.json();
    const text = j?.choices?.[0]?.message?.content || "No response.";
    chrome.notifications.create({
      type:"basic", iconUrl:"icon128.png",
      title:"Lilith Gateway", message: text.slice(0,220)
    });
  }catch(e){
    chrome.notifications.create({type:"basic", iconUrl:"icon128.png", title:"Lilith Gateway", message:"Error."});
  }
});
