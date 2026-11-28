r"""Start the demo:  python -m moderation.run --model "D:\path\to\xlmr"      (testing only: --stub)"""
import argparse, os, uvicorn

ap = argparse.ArgumentParser()
ap.add_argument("--model", help="folder with the fine-tuned model")
ap.add_argument("--stub", action="store_true", help="TESTING ONLY: keyword rule instead of the model")
ap.add_argument("--port", type=int, default=8000)
a = ap.parse_args()
if a.stub:
    os.environ["MOD_STUB"] = "1"
elif a.model:
    os.environ["MODEL_PATH"] = a.model
else:
    ap.error("give --model PATH (or --stub for testing)")
print(f"\nFeed:       http://127.0.0.1:{a.port}/\nModerator:  http://127.0.0.1:{a.port}/admin\n")
uvicorn.run("moderation.server:app", host="127.0.0.1", port=a.port)
