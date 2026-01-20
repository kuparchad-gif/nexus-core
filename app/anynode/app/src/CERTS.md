# TLS/mTLS quickstart (dev)
```bash
# CA
openssl ecparam -name prime256v1 -genkey -noout -out ca.key
openssl req -x509 -new -key ca.key -days 3650 -sha256 -subj "/CN=Nexus-CA" -out ca.crt

# Server
openssl ecparam -name prime256v1 -genkey -noout -out nim-gw.key
openssl req -new -key nim-gw.key -subj "/CN=nim-gateway" -out nim-gw.csr
openssl x509 -req -in nim-gw.csr -CA ca.crt -CAkey ca.key -CAcreateserial -days 825 -sha256 -out nim-gw.crt

# Client
openssl ecparam -name prime256v1 -genkey -noout -out client.key
openssl req -new -key client.key -subj "/CN=nexus-client" -out client.csr
openssl x509 -req -in client.csr -CA ca.crt -CAkey ca.key -CAcreateserial -days 825 -sha256 -out client.crt
```
