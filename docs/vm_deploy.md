# Deploying the prediction service on the VM

The VM is `pcom-vm` at RRZE: Ubuntu 24.04, IP `172.17.70.237`, with Docker installed. It is only reachable inside the FAU network, so from home connect eduVPN first.

The service is the FastAPI app in `service/main.py`. The Docker image (built from `service/Dockerfile`) contains only the files prediction needs.

**Before you start:** commit and push `service/`, `tests/data/round1_input.csv` and `tests/test_service.py`. The VM gets the code from GitHub, not from your laptop.

## 1. Log in

From PowerShell on the laptop:

```powershell
ssh -i "$HOME\.ssh\pcom-key.pem" ubuntu@172.17.70.237
```

Every command below runs on the VM.

## 2. Get the code

The first time:

```bash
cd ~
git clone https://github.com/mdobaidullahhaque/pcom-01-training-data.git pcom
cd ~/pcom
```

Every later time:

```bash
cd ~/pcom
git pull
```

Check that you have the commit you expect, and that the model arrived unchanged (the hash must match `model_sha256` in `models/pcom_entry_1/model_meta.json`):

```bash
git log --oneline -1
sha256sum models/pcom_entry_1/model.txt
grep model_sha256 models/pcom_entry_1/model_meta.json
```

If the repo is private, `git clone` asks for a username and password. Use a GitHub personal access token as the password, with read-only access to this repo.

## 3. Build the image

Run this from the repo root, not from `service/`:

```bash
cd ~/pcom
docker tag ml-prediction-api ml-prediction-api:previous 2>/dev/null || true   # keep the old image for rollback
docker build -f service/Dockerfile -t ml-prediction-api .
```

The build ends with `[self-check] OK: score=... output=[...]`. If the model cannot load or score, for example because of a wrong hash or missing `libgomp1`, the build fails at that step. The container you are running now keeps running in that case.

## 4. Replace the running container

Keep both port mappings and the restart policy:

```bash
docker stop ml-api
docker rm ml-api
docker run -d --name ml-api --restart unless-stopped -p 80:8000 -p 8000:8000 ml-prediction-api
```

## 5. Health check

On the VM:

```bash
docker ps --filter name=ml-api        # STATUS "Up", PORTS 0.0.0.0:80->8000, 0.0.0.0:8000->8000
curl -s http://localhost/health       # {"health":"up"}
curl -s http://localhost:8000/health  # {"health":"up"}
```

From the laptop, on the FAU network or eduVPN:

```powershell
Invoke-RestMethod http://172.17.70.237/health
Invoke-RestMethod http://172.17.70.237:8000/health
```

## 6. Run a full test with fake_compy

This runs a full round inside the container. The fake competition server runs next to the service: it serves the round 1 input, receives the upload, and checks it.

```bash
cd ~/pcom
docker cp service/fake_compy.py ml-api:/tmp/fake_compy.py
docker cp tests/data/round1_input.csv ml-api:/tmp/round1_input.csv
docker exec ml-api python /tmp/fake_compy.py --csv /tmp/round1_input.csv --port 9000 \
    --out /tmp --public-url http://127.0.0.1:9000 --trigger http://127.0.0.1:8000
```

You should see:

```
POST /predict -> {"status":"prediction started"}
Upload received 0.xx s after POST /predict
--- prediction.csv ---
target_ghsa_low_count_90d,target_ghsa_moderate_count_90d,target_ghsa_high_count_90d,target_ghsa_critical_count_90d,target_ghsa_unknown_count_90d
0.060606,0.363636,0.121212,0.030303,0.000000
----------------------
VALID: 1 row, 5 target headers, plain non-negative numbers
```

The exit code is 0 when the upload is valid. The files copied to `/tmp` are gone after the next redeploy.

**Optional: test through the published port as well.** Here the fake server runs on the VM itself (it needs only the standard library, so the system `python3` is enough), and the service reaches it through the Docker bridge at `172.17.0.1`:

```bash
python3 service/fake_compy.py --csv tests/data/round1_input.csv --port 9000 --out /tmp \
    --public-url http://172.17.0.1:9000 --trigger http://localhost:8000
```

If this one waits until it times out, a host firewall is probably blocking the container from reaching port 9000 (check `sudo ufw status`). The in-container test above is the one that matters.

## 7. Logs

```bash
docker logs --tail 50 ml-api
docker logs -f ml-api          # follow; Ctrl+C stops following, not the container
```

Each round writes these lines:

```
[predict] started: download=... upload=...
[predict] downloaded 16217 bytes in 0.01s
[run_model] rows=18 chosen=row 18 (newest published_at, published_at=2026-06-17T10:33:28.065Z) score=26.3859 output=[0.060606, 0.363636, 0.121212, 0.030303, 0.0] time=0.52s
[predict] uploaded (HTTP 200) in 0.60s total. Evaluation result: {...}
```

If you see `[run_model] FALLBACK all zeros (...)`, the input could not be used and zeros were uploaded. The reason is in the brackets.

## Rollback

If the new image misbehaves, go back to the previous one:

```bash
docker stop ml-api && docker rm ml-api
docker run -d --name ml-api --restart unless-stopped -p 80:8000 -p 8000:8000 ml-prediction-api:previous
```

## Cleanup (optional, later)

```bash
docker image prune -f   # removes untagged old images; keeps ml-prediction-api and :previous
```
