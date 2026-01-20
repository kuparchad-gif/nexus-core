export type Agent = { id:string; name:string; role:string; color?:string };
export const AGENTS: Agent[] = [
  { id:"director",      name:"Director",        role:"Big-picture orchestration, Solana strategy & alliances", color:"#3b82f6" },
  { id:"db",            name:"Database Manager",role:"Ingests/serves data to researchers & analysts",          color:"#0ea5e9" },
  { id:"research",      name:"Research Ops",    role:"Web crawlers; intel for strategists",                    color:"#22c55e" },
  { id:"strategists",   name:"Strategists",     role:"Plans, campaigns, council outputs",                      color:"#f59e0b" },
  { id:"architects",    name:"Architects",      role:"Systems design across Nexus/Viren",                      color:"#a855f7" },
  { id:"engineers",     name:"Engineers",       role:"Build & integrate services",                             color:"#06b6d4" },
  { id:"marketing",     name:"Marketing",       role:"Viral growth engines & funnels",                         color:"#ef4444" },
  { id:"avatars",       name:"Avatar Studio",   role:"AI avatars, faces of Aethereal",                         color:"#14b8a6" }
];