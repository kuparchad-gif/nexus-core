import sys, base64
from nim.codec import encode
if __name__ == "__main__":
    data = sys.stdin.buffer.read()
    out = encode(data, data_tiles_per_frame=48)
    sys.stdout.buffer.write(base64.b64encode(out))
