type TTSOpts = {
  onWord?: (e?: any)=>void
  onEnd?: ()=>void
  rate?: number
  pitch?: number
  voiceName?: string
}

export function speakText(text: string, opts: TTSOpts = {}){
  if (!('speechSynthesis' in window)) {
    alert('SpeechSynthesis not supported in this browser.')
    opts.onEnd?.()
    return
  }
  window.speechSynthesis.cancel()

  const u = new SpeechSynthesisUtterance(text)
  u.lang = 'en-US'
  u.rate = opts.rate ?? 1.0
  u.pitch = opts.pitch ?? 1.0

  // pick preferred voice if available
  const voices = window.speechSynthesis.getVoices()
  if (opts.voiceName){
    const v = voices.find(v => v.name.includes(opts.voiceName!))
    if (v) u.voice = v
  }

  u.onboundary = (ev: any) => {
    if (ev.name === 'word' || ev.charIndex >= 0) {
      opts.onWord?.(ev)
    }
  }
  u.onend = ()=> opts.onEnd?.()
  window.speechSynthesis.speak(u)
}
