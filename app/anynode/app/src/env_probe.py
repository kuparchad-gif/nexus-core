#!/usr/bin/env python3
import os, sys, json, urllib.request, urllib.error, socket, platform, pathlib, time, datetime, hashlib, hmac

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODES = json.loads((ROOT/'config/env.modes.json').read_text())

def is_k8s():
    return bool(os.environ.get('KUBERNETES_SERVICE_HOST')) or            (pathlib.Path('/var/run/secrets/kubernetes.io').exists())

def is_docker():
    if pathlib.Path('/.dockerenv').exists():
        return True
    try:
        with open('/proc/1/cgroup','r') as f:
            s = f.read()
            return ('docker' in s) or ('containerd' in s)
    except Exception:
        return False

def try_http(url, path='/health', timeout=0.7):
    if not url:
        return False
    target = url.rstrip('/') + path
    try:
        with urllib.request.urlopen(urllib.request.Request(target, method='GET'), timeout=timeout) as r:
            return 200 <= r.status < 500
    except Exception:
        return False

def pick(mode, explicit_env):
    a = explicit_env.get('ARCHIVER_URL') or MODES[mode]['ARCHIVER_URL']
    l = explicit_env.get('LOKI_URL')     or MODES[mode]['LOKI_URL']

    arch_cands = [a] + list({MODES[m]['ARCHIVER_URL'] for m in MODES})
    loki_cands = [l] + list({MODES[m]['LOKI_URL'] for m in MODES})

    arch = next((u for u in arch_cands if try_http(u, '/health') or try_http(u, '/policy') or try_http(u, '/')), a)
    loki = next((u for u in loki_cands if try_http(u, '/policy') or try_http(u, '/health') or try_http(u, '/')), l)

    return arch, loki

def get_ips():
    hn = socket.gethostname()
    ips = set()
    try:
        for fam in (socket.AF_INET, socket.AF_INET6):
            for res in socket.getaddrinfo(hn, None, fam, socket.SOCK_STREAM):
                addr = res[4][0]
                if not addr.startswith('::1') and not addr.startswith('127.'):
                    ips.add(addr)
    except Exception:
        pass
    return sorted(ips)

def sign_if_needed(msg_bytes: bytes) -> dict:
    headers = {'Content-Type': 'application/json'}
    token = os.environ.get('ARCHIVER_TOKEN')
    if token:
        headers['Authorization'] = f'Bearer {token}'
    key = os.environ.get('ARCHIVER_HMAC_KEY')
    if key:
        try:
            try_key = bytes.fromhex(key)
        except ValueError:
            try_key = key.encode('utf-8')
        headers['X-Signature'] = hmac.new(try_key, msg_bytes, hashlib.sha256).hexdigest()
    return headers

def emit_archiver(archiver_url: str, payload: dict):
    endpoint = archiver_url.rstrip('/') + '/events/env'
    body = json.dumps(payload, sort_keys=True).encode('utf-8')
    headers = sign_if_needed(body)
    try:
        req = urllib.request.Request(endpoint, data=body, headers=headers, method='POST')
        with urllib.request.urlopen(req, timeout=2.5) as r:
            return (200 <= r.status < 300, f'status={r.status}')
    except Exception as e:
        return (False, f'error={e}')

def main():
    out = pathlib.Path(sys.argv[sys.argv.index('--out')+1]) if '--out' in sys.argv else (ROOT/'sidecar/.env.resolved')
    report = '--report' in sys.argv

    explicit = {
        'ARCHIVER_URL': os.environ.get('ARCHIVER_URL'),
        'LOKI_URL': os.environ.get('LOKI_URL'),
    }

    if os.environ.get('ENV_MODE') in ('local','compose','k8s'):
        mode = os.environ['ENV_MODE']
    else:
        mode = 'k8s' if is_k8s() else ('compose' if is_docker() else 'local')

    arch, loki = pick(mode, explicit)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f'ENV_MODE={mode}\nARCHIVER_URL={arch}\nLOKI_URL={loki}\n', encoding='utf-8')

    now = datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc)
    payload = {
        'event': 'env_probe',
        'ts_epoch': time.time(),
        'ts_iso': now.isoformat().replace('+00:00','Z'),
        'node_name': platform.node(),
        'env_mode': mode,
        'archiver_url': arch,
        'loki_url': loki,
        'probes': {
            'archiver_alive': try_http(arch, '/health') or try_http(arch, '/policy') or try_http(arch, '/'),
            'loki_alive': try_http(loki, '/policy') or try_http(loki, '/health') or try_http(loki, '/'),
        },
        'host': {
            'os': platform.system(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
            'python': platform.python_version(),
        },
        'net': {
            'hostname': socket.gethostname(),
            'ips': get_ips(),
        },
        'repo_hint': 'meta-nexus',
        'source': 'scripts/env_probe.py'
    }

    msg = f'[env_probe] mode={mode} -> ARCHIVER_URL={arch}  LOKI_URL={loki}\nWrote {out}'
    if report and arch:
        ok, info = emit_archiver(arch, payload)
        msg += f'\nReport to Archiver: {'ok' if ok else 'fail'} ({info})'
    print(msg)

if __name__ == '__main__':
    main()
