// Optional AEAD wrapper for end-to-end payload security (inside NIM).
import crypto from 'crypto';

export function seal(key, plaintext, aad = Buffer.from('NIM v0.1')) {
  // key: 32 bytes (AES-256-GCM)
  const iv = crypto.randomBytes(12);
  const cipher = crypto.createCipheriv('aes-256-gcm', key, iv, { authTagLength: 16 });
  cipher.setAAD(aad, { plaintextLength: plaintext.length });
  const ct = Buffer.concat([cipher.update(plaintext), cipher.final()]);
  const tag = cipher.getAuthTag();
  return Buffer.concat([iv, ct, tag]); // 12 + ct + 16
}

export function open(key, sealed, aad = Buffer.from('NIM v0.1')) {
  const iv = sealed.subarray(0,12);
  const tag = sealed.subarray(sealed.length-16);
  const ct  = sealed.subarray(12, sealed.length-16);
  const decipher = crypto.createDecipheriv('aes-256-gcm', key, iv, { authTagLength: 16 });
  decipher.setAAD(aad, { plaintextLength: ct.length });
  decipher.setAuthTag(tag);
  const pt = Buffer.concat([decipher.update(ct), decipher.final()]);
  return pt;
}
