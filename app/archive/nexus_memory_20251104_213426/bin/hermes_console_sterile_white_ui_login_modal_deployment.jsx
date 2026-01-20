import React, { useEffect, useMemo, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter, DialogTrigger } from "@/components/ui/dialog";
import { Select, SelectTrigger, SelectContent, SelectItem, SelectValue } from "@/components/ui/select";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { ToastProvider, useToast } from "@/components/ui/use-toast";
import { Separator } from "@/components/ui/separator";
import { Textarea } from "@/components/ui/textarea";
import { Switch } from "@/components/ui/switch";
import { Loader2, Rocket, LogIn, Power, Server, ShieldCheck, PlugZap, Activity } from "lucide-react";

/**
 * Hermes Console – Sterile White UI
 * ------------------------------------------------------------
 * • Minimal, high-contrast on white. Aethereal accent (soft azure).
 * • Login modal, then a deployment modal ("Aethereal Bang" = animated glow).
 * • Health + Models panel, and a quick chat smoke-test.
 * • API base defaults to origin; override via window.HERMES_API or query ?api=
 */

// Theme tokens (tweak freely)
const THEME = {
  bg: "bg-white",
  fg: "text-gray-900",
  subtle: "text-gray-500",
  card: "bg-white/80",
  ring: "ring-1 ring-gray-200",
  // Aethereal accent (sterile, airy blue)
  accent: "from-sky-400/30 via-sky-500/20 to-indigo-500/10",
  accentSolid: "text-sky-600",
};

type HealthStatus = {
  platform?: string;
  status?: string;
  components?: Record<string, string>;
  timestamp?: string;
  llm_ready?: boolean;
};

type ModelItem = {
  id?: string;
  name?: string;
  model?: string;
  description?: string;
  type?: string;
};

type ChatChoice = { message?: { role: string; content: string } };

const apiBase = () => {
  const urlParam = new URLSearchParams(window.location.search).get("api");
  // @ts-ignore optional global override
  const global = (window as any).HERMES_API as string | undefined;
  return urlParam || global || `${window.location.origin}`;
};

function useInterval(fn: () => void, ms: number, enabled = true) {
  useEffect(() => {
    if (!enabled) return;
    const id = setInterval(fn, ms);
    return () => clearInterval(id);
  }, [fn, ms, enabled]);
}

function GlowBackdrop({ show }: { show: boolean }) {
  return (
    <AnimatePresence>
      {show && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="pointer-events-none fixed inset-0 -z-10"
        >
          <motion.div
            initial={{ scale: 0.9, opacity: 0.7 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ type: "spring", stiffness: 120, damping: 20 }}
            className={`absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 h-[60vmax] w-[60vmax] rounded-full blur-3xl bg-gradient-to-br ${THEME.accent}`}
          />
        </motion.div>
      )}
    </AnimatePresence>
  );
}

function LoginModal({ onAuthed }: { onAuthed: (token: string) => void }) {
  const [open, setOpen] = useState(true);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const { toast } = useToast();

  async function doLogin() {
    setBusy(true);
    try {
      // Placeholder: accept any non-empty creds. Replace with real auth call.
      if (!email || !password) throw new Error("Enter email & password");
      const fakeToken = btoa(`${email}:${Date.now()}`);
      localStorage.setItem("hermes.token", fakeToken);
      onAuthed(fakeToken);
      setOpen(false);
    } catch (e: any) {
      toast({ title: "Login failed", description: e.message, variant: "destructive" as any });
    } finally {
      setBusy(false);
    }
  }

  return (
    <Dialog open={open}>
      <GlowBackdrop show={open} />
      <DialogContent className={`sm:max-w-md ${THEME.card} ${THEME.ring} backdrop-blur-xl`}> 
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <ShieldCheck className="h-5 w-5 text-sky-600" />
            Hermes Console
          </DialogTitle>
          <DialogDescription>Sign in to deploy & monitor your nodes.</DialogDescription>
        </DialogHeader>
        <div className="grid gap-3">
          <div className="grid gap-1">
            <Label htmlFor="email">Email</Label>
            <Input id="email" type="email" value={email} onChange={e => setEmail(e.target.value)} placeholder="you@ae.dev" />
          </div>
          <div className="grid gap-1">
            <Label htmlFor="pw">Password</Label>
            <Input id="pw" type="password" value={password} onChange={e => setPassword(e.target.value)} />
          </div>
          <Button className="mt-2" onClick={doLogin} disabled={busy}>
            {busy ? <Loader2 className="mr-2 h-4 w-4 animate-spin"/> : <LogIn className="mr-2 h-4 w-4"/>}
            Sign In
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}

function TopBar({ onDeploy }: { onDeploy: () => void }) {
  return (
    <div className={`sticky top-0 z-20 ${THEME.bg} ${THEME.ring} backdrop-blur supports-[backdrop-filter]:bg-white/80`}> 
      <div className="mx-auto flex h-14 max-w-6xl items-center justify-between px-4">
        <div className="flex items-center gap-2">
          <div className="h-6 w-6 rounded-full bg-gradient-to-br from-sky-400 to-indigo-400"/>
          <span className="font-semibold tracking-tight">Æthereal – Hermes</span>
          <Badge variant="secondary" className="ml-2">Sterile White</Badge>
        </div>
        <div className="flex items-center gap-2">
          <Dialog>
            <DialogTrigger asChild>
              <Button variant="outline" className="gap-2"><PlugZap className="h-4 w-4"/> API</Button>
            </DialogTrigger>
            <DialogContent className={`${THEME.card} ${THEME.ring} backdrop-blur-xl`}>
              <DialogHeader>
                <DialogTitle>API Base</DialogTitle>
                <DialogDescription>Change the backend target for this console.</DialogDescription>
              </DialogHeader>
              <div className="grid gap-2">
                <Label>Effective</Label>
                <Input value={apiBase()} readOnly />
                <p className="text-sm text-gray-500">Override with <code>?api=https://host:11434</code> or <code>window.HERMES_API</code>.</p>
              </div>
            </DialogContent>
          </Dialog>
          <Button onClick={onDeploy} className="gap-2">
            <Rocket className="h-4 w-4"/>
            Deploy
          </Button>
        </div>
      </div>
    </div>
  );
}

function DeployModal({ open, onOpenChange }: { open: boolean; onOpenChange: (o: boolean) => void }) {
  const [platform, setPlatform] = useState<string>("cpu");
  const [busy, setBusy] = useState(false);
  const { toast } = useToast();

  async function triggerDeploy() {
    setBusy(true);
    try {
      // Optional: send a hint header with platform; server can ignore.
      const res = await fetch(`${apiBase()}/deploy`, { method: "POST", headers: { "x-platform": platform } });
      if (!res.ok) throw new Error(await res.text());
      toast({ title: "Deploy requested", description: `Platform: ${platform}` });
      onOpenChange(false);
    } catch (e: any) {
      toast({ title: "Deploy failed", description: e.message, variant: "destructive" as any });
    } finally {
      setBusy(false);
    }
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <GlowBackdrop show={open} />
      <DialogContent className={`${THEME.card} ${THEME.ring} backdrop-blur-xl`}>
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2"><Power className="h-5 w-5 text-sky-600"/> Modal Deployment</DialogTitle>
          <DialogDescription>Select target and fire the Aethereal Bang.</DialogDescription>
        </DialogHeader>
        <div className="grid gap-4">
          <div className="grid gap-1">
            <Label>Platform</Label>
            <Select value={platform} onValueChange={setPlatform}>
              <SelectTrigger className="w-full"><SelectValue placeholder="Choose platform"/></SelectTrigger>
              <SelectContent>
                <SelectItem value="cpu">CPU (Docker Compose)</SelectItem>
                <SelectItem value="windows-vulkan">Windows · Vulkan</SelectItem>
                <SelectItem value="linux-cuda">Linux · CUDA</SelectItem>
                <SelectItem value="linux-rocm">Linux · ROCm</SelectItem>
                <SelectItem value="linux-sycl">Linux · SYCL</SelectItem>
                <SelectItem value="macos-metal">macOS · Metal</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <Button onClick={triggerDeploy} disabled={busy} className="gap-2">
            {busy ? <Loader2 className="h-4 w-4 animate-spin"/> : <Rocket className="h-4 w-4"/>}
            Launch
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}

function HealthPanel() {
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [loading, setLoading] = useState(false);

  const fetchHealth = async () => {
    try {
      setLoading(true);
      const res = await fetch(`${apiBase()}/health`);
      setHealth(await res.json());
    } catch (_) {
      setHealth(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchHealth(); }, []);
  useInterval(fetchHealth, 5000, true);

  const overall = health?.status || "unknown";

  return (
    <Card className={`${THEME.card} ${THEME.ring}`}>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Activity className="h-5 w-5 text-sky-600"/>
          Health
          <Badge variant={overall === "healthy" ? "default" : overall === "degraded" ? "secondary" : "destructive"} className="ml-2 capitalize">{overall}</Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="grid gap-3 text-sm">
        <div className="grid grid-cols-2 gap-2">
          <div><span className="text-gray-500">Platform</span><div className="font-medium">{health?.platform ?? "—"}</div></div>
          <div><span className="text-gray-500">Timestamp</span><div className="font-medium">{health?.timestamp ?? "—"}</div></div>
        </div>
        <Separator/>
        <div className="grid gap-2">
          <div className="text-gray-500">Components</div>
          <div className="grid gap-1">
            {health?.components ? (
              Object.entries(health.components).map(([k, v]) => (
                <div key={k} className="flex items-center justify-between rounded-lg border border-gray-200 px-3 py-2">
                  <span className="font-medium">{k}</span>
                  <span className="text-gray-700">{String(v)}</span>
                </div>
              ))
            ) : (
              <div className="text-gray-400">No data</div>
            )}
          </div>
        </div>
        {loading && <div className="flex items-center gap-2 text-gray-500"><Loader2 className="h-4 w-4 animate-spin"/> Refreshing…</div>}
      </CardContent>
    </Card>
  );
}

function ModelsPanel() {
  const [models, setModels] = useState<ModelItem[]>([]);
  const [loading, setLoading] = useState(false);

  const fetchModels = async () => {
    try {
      setLoading(true);
      const res = await fetch(`${apiBase()}/v1/models`);
      const data = await res.json();
      const arr: any[] = data?.models || [];
      setModels(arr);
    } catch (_) {
      setModels([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchModels(); }, []);
  useInterval(fetchModels, 10000, true);

  return (
    <Card className={`${THEME.card} ${THEME.ring}`}>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Server className="h-5 w-5 text-sky-600"/>
          Models
        </CardTitle>
      </CardHeader>
      <CardContent className="grid gap-2 text-sm">
        {models.length === 0 && (
          <div className="text-gray-500">No models yet. Deploy and load one to begin.</div>
        )}
        {models.map((m, i) => (
          <div key={i} className="flex items-center justify-between rounded-lg border border-gray-200 px-3 py-2">
            <div className="truncate">
              <div className="font-medium truncate max-w-[44ch]">{m.name || m.id || m.model || "unknown"}</div>
              {m.description && <div className="text-gray-500 truncate max-w-[60ch]">{m.description}</div>}
            </div>
            <Badge variant="outline">{m.type || "model"}</Badge>
          </div>
        ))}
      </CardContent>
    </Card>
  );
}

function ChatSmokeTest() {
  const [prompt, setPrompt] = useState("Say 'hello from Hermes' in one short line.");
  const [resp, setResp] = useState<string>("");
  const [busy, setBusy] = useState(false);
  const { toast } = useToast();

  async function run() {
    setBusy(true);
    setResp("");
    try {
      const body = {
        model: "hermes-7b", // HermesOS proxies to your active model; adjust as needed
        messages: [
          { role: "system", content: "Be concise." },
          { role: "user", content: prompt }
        ],
        max_tokens: 64,
        temperature: 0.2,
        stream: false
      };
      const res = await fetch(`${apiBase()}/v1/chat/completions`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      const text = data?.choices?.[0]?.message?.content ?? JSON.stringify(data);
      setResp(String(text));
    } catch (e: any) {
      toast({ title: "Chat failed", description: e.message, variant: "destructive" as any });
    } finally {
      setBusy(false);
    }
  }

  return (
    <Card className={`${THEME.card} ${THEME.ring}`}>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <ShieldCheck className="h-5 w-5 text-sky-600"/>
          Smoke Test
        </CardTitle>
      </CardHeader>
      <CardContent className="grid gap-3">
        <Textarea value={prompt} onChange={(e) => setPrompt(e.target.value)} className="min-h-[84px]" />
        <div className="flex gap-2">
          <Button onClick={run} disabled={busy} className="gap-2">
            {busy ? <Loader2 className="h-4 w-4 animate-spin"/> : <Rocket className="h-4 w-4"/>}
            Send
          </Button>
          <Button variant="outline" onClick={() => setResp("")}>Clear</Button>
        </div>
        {resp && (
          <pre className="whitespace-pre-wrap rounded-lg border border-gray-200 bg-white p-3 text-sm text-gray-800">{resp}</pre>
        )}
      </CardContent>
    </Card>
  );
}

export default function HermesConsole() {
  const [token, setToken] = useState<string | null>(() => localStorage.getItem("hermes.token"));
  const [deployOpen, setDeployOpen] = useState(false);

  useEffect(() => {
    if (token) localStorage.setItem("hermes.token", token);
  }, [token]);

  return (
    <ToastProvider>
      <div className={`min-h-[100dvh] ${THEME.bg} ${THEME.fg}`}> 
        {!token && <LoginModal onAuthed={setToken} />}

        {/* Hero */}
        <section className="relative mx-auto max-w-6xl px-4 py-10">
          <div className="mb-6">
            <TopBar onDeploy={() => setDeployOpen(true)} />
          </div>

          <div className="relative overflow-hidden rounded-3xl border border-gray-200 p-8">
            <div className="pointer-events-none absolute inset-0 bg-gradient-to-b from-gray-50/60 to-white"/>
            <div className="relative grid gap-6 md:grid-cols-2">
              <div>
                <h1 className="mb-2 text-2xl font-semibold tracking-tight">Deploy. Orchestrate. Observe.</h1>
                <p className="max-w-prose text-gray-600">Sterile white console for the Hermes OS. Launch backends across CPU/GPU stacks, keep an eye on health, and sanity-check with a single prompt.</p>
                <div className="mt-4 flex gap-3">
                  <Button onClick={() => setDeployOpen(true)} className="gap-2">
                    <Rocket className="h-4 w-4"/> Aethereal Bang
                  </Button>
                  <Dialog>
                    <DialogTrigger asChild>
                      <Button variant="outline">What is this?</Button>
                    </DialogTrigger>
                    <DialogContent className={`${THEME.card} ${THEME.ring} backdrop-blur-xl`}>
                      <DialogHeader>
                        <DialogTitle>Console Primer</DialogTitle>
                        <DialogDescription>How the pieces click.</DialogDescription>
                      </DialogHeader>
                      <ul className="list-disc space-y-2 pl-5 text-sm text-gray-700">
                        <li><strong>Deploy</strong> triggers <code>POST /deploy</code> on your HermesOS.</li>
                        <li><strong>Health</strong> polls <code>GET /health</code> every 5s.</li>
                        <li><strong>Models</strong> polls <code>GET /v1/models</code> every 10s.</li>
                        <li><strong>Smoke Test</strong> posts to <code>/v1/chat/completions</code>.</li>
                        <li>Change API target via the API dialog in the header.</li>
                      </ul>
                    </DialogContent>
                  </Dialog>
                </div>
              </div>
              <div className="grid gap-4 md:grid-cols-2">
                <HealthPanel />
                <ModelsPanel />
              </div>
            </div>
          </div>

          <div className="mt-8">
            <ChatSmokeTest />
          </div>
        </section>

        <DeployModal open={deployOpen} onOpenChange={setDeployOpen} />
      </div>
    </ToastProvider>
  );
}
