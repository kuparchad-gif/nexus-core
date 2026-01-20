import { useEffect, useState } from "react";
export function useHashRoute(defaultPath="/"){
  const get = () => (location.hash.replace(/^#/,"") || defaultPath);
  const [path,setPath] = useState(get());
  useEffect(()=>{ const h=()=>setPath(get()); window.addEventListener("hashchange",h); return ()=>window.removeEventListener("hashchange",h);},[]);
  const go = (p:string)=>{ if(!p.startsWith("/")) p="/"+p; location.hash = p; };
  return { path, go };
}