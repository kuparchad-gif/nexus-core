# C:\Projects\Stacks\nexus-metatron\backend\services\common\abe.py
import zlib, gzip, json
try:
    from nacl.signing import SigningKey, VerifyKey
    from nacl.exceptions import BadSignatureError
except Exception:  # soft import; allows running without nacl installed
    SigningKey = None
    VerifyKey  = None
    BadSignatureError = Exception

def pack_abe(header_dict:dict, payload_bytes:bytes, signing_key_hex:str|None=None):
    header_gz = gzip.compress(json.dumps(header_dict).encode())
    payload_gz= gzip.compress(payload_bytes)
    hcrc = zlib.crc32(header_gz) & 0xffffffff
    pcrc = zlib.crc32(payload_gz) & 0xffffffff
    sig=b""
    if signing_key_hex and SigningKey:
        sk = SigningKey(bytes.fromhex(signing_key_hex))
        sig = sk.sign(header_gz + payload_gz).signature
    return {
        "signature": sig,
        "header_crc32": hcrc,
        "payload_crc32": pcrc,
        "header": header_gz,
        "payload": payload_gz,
    }

def verify_abe(abe:dict, verify_key_hex:str|None=None):
    if (zlib.crc32(abe["header"]) & 0xffffffff) != abe["header_crc32"]:
        raise ValueError("Header CRC mismatch")
    if (zlib.crc32(abe["payload"]) & 0xffffffff) != abe["payload_crc32"]:
        raise ValueError("Payload CRC mismatch")
    if verify_key_hex and VerifyKey and abe.get("signature"):
        vk = VerifyKey(bytes.fromhex(verify_key_hex))
        try:
            vk.verify(abe["header"] + abe["payload"], abe["signature"])
        except BadSignatureError:
            raise ValueError("Signature verify failed")
    return True
