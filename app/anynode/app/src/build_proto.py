import os, sys, subprocess
from pathlib import Path
root = Path(__file__).resolve().parents[2]
proto = root/'backend/common/proto/metatron.proto'
out = root/'backend/common/frames/gen'
out.mkdir(parents=True, exist_ok=True)
subprocess.check_call([sys.executable, '-m', 'grpc_tools.protoc', f'-I{proto.parent}', f'--python_out={out}', str(proto)])
print('Generated to', out)
