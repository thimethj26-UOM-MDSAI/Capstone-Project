# Moderation demo (social feed + scanner + moderator dashboard)

Run from the repo root, with the venv active:

    pip install fastapi uvicorn
    python -m moderation.calibrate --model "D:\path\to\xlmr"     # once: picks thresholds from VALIDATION data
    python -m moderation.run --model "D:\path\to\xlmr"

Then open  http://127.0.0.1:8000/  (social feed)  and  http://127.0.0.1:8000/admin  (moderator dashboard).
On the dashboard click "Load demo data" first. `--stub` is a keyword rule for testing the plumbing only.

Policy: p >= remove -> auto-removed + alert to author; flag <= p < remove -> visible but queued for a human;
p < flag -> published. Moderators can restore / approve / confirm removal; every step is in an audit log.
No authentication: local demo only.
