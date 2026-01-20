import sys, base64
from nim.codec import decode
if __name__ == "__main__":
    enc = base64.b64decode(sys.stdin.buffer.read())
    out = decode(enc, data_tiles_per_frame=48)
    sys.stdout.buffer.write(out)
