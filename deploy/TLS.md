# HTTPS for IDS (port 9443)

Tilletia listens on **HTTP `:8000`** as before. Optionally, **Caddy** terminates TLS on
**`:9443`** and reverse-proxies to `127.0.0.1:8000`. There is no forced redirect from HTTP
to HTTPS.

Certificate: wildcard `*.agroprosperis.com` (PFX from IT: `ap_local_cert.7z`).

## URLs

| Mode | URL |
|---|---|
| Legacy (no SSL) | `http://<host-ip>:8000/` |
| IDS HTTPS | `https://<code>-ids.agroprosperis.com:9443/` |

Examples: `https://zhe-ids.agroprosperis.com:9443/`, `https://adb-ids.agroprosperis.com:9443/`.

DNS: each `*-ids.agroprosperis.com` A-record must point at the elevator PC IP.
`vlm-ai01` and `vlm-ai02` share hostname `vlm-ids.agroprosperis.com` — set the A-record to
the PC that should serve clients.

Host → hostname mapping lives in `deploy/ansible/inventory.ini` (`tilletia_ids_host=`).

## One-time setup on a host

1. Copy PFX and password (do **not** commit these to git):

```bash
sudo install -d -m 0700 /etc/tilletia-app/tls
sudo cp agroprosperis_local.pfx /etc/tilletia-app/tls/
# password is the single line from ap_local_cert.txt
echo -n 'YOUR_PFX_PASSWORD' | sudo tee /etc/tilletia-app/tls/pfx.password >/dev/null
sudo chmod 0600 /etc/tilletia-app/tls/agroprosperis_local.pfx /etc/tilletia-app/tls/pfx.password
```

2. Extract PEM files:

```bash
cd /opt/tilletia-app/deploy   # or your checkout
sudo ./install-tls.sh
```

3. Set IDS hostname (or use ansible below):

```bash
echo 'zhe-ids.agroprosperis.com' | sudo tee /etc/tilletia-app/ids-host
```

4. Ensure Caddyfile is installed and restart the stack:

```bash
sudo cp /opt/tilletia-app/config/Caddyfile /etc/tilletia-app/Caddyfile
sudo systemctl restart tilletia-app.service
```

5. Verify both modes:

```bash
curl -fsS -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8000/login
curl -fsSk -o /dev/null -w '%{http_code}\n' \
  --resolve zhe-ids.agroprosperis.com:9443:127.0.0.1 \
  https://zhe-ids.agroprosperis.com:9443/login
```

If TLS files or `ids-host` are missing, `start.sh` skips Caddy and only serves `:8000`.

## Ansible helpers

Write `ids-host` from inventory:

```bash
cd deploy/ansible
ansible -i inventory.ini zhe-ai01 --become --ask-become-pass -m copy \
  -a "content={{ tilletia_ids_host }}\ndest=/etc/tilletia-app/ids-host mode=0644"
```

Copy PFX + password to all hosts (adjust local paths):

```bash
ansible -i inventory.ini tilletia --become --ask-become-pass -m file \
  -a "path=/etc/tilletia-app/tls state=directory mode=0700"
ansible -i inventory.ini tilletia --become --ask-become-pass -m copy \
  -a "src=/Users/nesh/Downloads/ap_local_cert_unpacked/agroprosperis_local.pfx dest=/etc/tilletia-app/tls/agroprosperis_local.pfx mode=0600"
ansible -i inventory.ini tilletia --become --ask-become-pass -m copy \
  -a "content=YOUR_PFX_PASSWORD\ndest=/etc/tilletia-app/tls/pfx.password mode=0600"
ansible -i inventory.ini tilletia --become --ask-become-pass -m shell \
  -a "cd /opt/tilletia-app/deploy && ./install-tls.sh && cp ../config/Caddyfile /etc/tilletia-app/Caddyfile && systemctl restart tilletia-app.service"
```

Also write `ids-host` for the whole fleet:

```bash
ansible -i inventory.ini tilletia --become --ask-become-pass -m copy \
  -a "content={{ tilletia_ids_host }}\ndest=/etc/tilletia-app/ids-host mode=0644"
```

## Layout on disk

| Path | Purpose |
|---|---|
| `/etc/tilletia-app/tls/agroprosperis_local.pfx` | Source PFX (optional after PEM extract) |
| `/etc/tilletia-app/tls/pfx.password` | PFX password (mode 0600) |
| `/etc/tilletia-app/tls/fullchain.pem` | Cert chain for Caddy |
| `/etc/tilletia-app/tls/key.pem` | Private key for Caddy |
| `/etc/tilletia-app/ids-host` | e.g. `zhe-ids.agroprosperis.com` |
| `/etc/tilletia-app/Caddyfile` | Installed from `config/Caddyfile` |

Docker Compose profile `tls` starts container `tilletia-app-caddy` only when
`start.sh` detects PEM + `ids-host`.
