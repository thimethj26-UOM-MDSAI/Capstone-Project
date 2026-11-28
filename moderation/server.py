"""Moderation back end: mock social network API + scanner + moderator actions. Run via moderation/run.py."""
import json, os, time
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from . import db
from .scanner import TransformerScorer, StubScorer, load_thresholds

USERS = ["Nimal", "Kamala", "Saman", "Dilini", "Ruwan"]
STATIC = Path(__file__).parent / "static"
app = FastAPI(title="Sinhala moderation demo")
S = {}  # scorer + thresholds, filled at startup


@app.on_event("startup")
def startup():
    db.init()
    if os.environ.get("MOD_STUB") == "1":
        S["scorer"] = StubScorer()
        print("\n*** WARNING: STUB scorer (keyword rule), not the real model. Testing only. ***\n")
    else:
        S["scorer"] = TransformerScorer(os.environ["MODEL_PATH"])
    S["th"] = load_thresholds()
    print("Scorer:", S["scorer"].name, "| thresholds:", S["th"])


class PostIn(BaseModel):
    author: str
    text: str


class CommentIn(PostIn):
    post_id: int


class ActionIn(BaseModel):
    action: str  # restore | confirm_removal | approve


def decide(prob):
    t = S["th"]
    return "removed" if prob >= t["remove"] else "flagged" if prob >= t["flag"] else "published"


def event(item_id, name, detail=""):
    db.x("INSERT INTO events(item_id,event,detail,ts) VALUES(?,?,?,?)", (item_id, name, detail, time.time()))


def alert(user, item_id, msg):
    db.x("INSERT INTO alerts(user,item_id,message,created_at) VALUES(?,?,?,?)", (user, item_id, msg, time.time()))


def scan_into(item_id, text, author, first=True, old_status=None):
    """Score text, store decision and explanation, emit event and alert. Returns new status."""
    r = S["scorer"].score(text)
    status = decide(r["prob"])
    db.x("UPDATE items SET status=?,prob=?,words=?,t_flag=?,t_remove=?,model=?,scanned_at=? WHERE id=?",
         (status, r["prob"], json.dumps(r["words"], ensure_ascii=False), S["th"]["flag"],
          S["th"]["remove"], S["scorer"].name, time.time(), item_id))
    event(item_id, "scanned" if first else "rescanned",
          f"prob={r['prob']:.3f} -> {status} (flag>={S['th']['flag']}, remove>={S['th']['remove']})")
    if status == "removed" and old_status != "removed":
        alert(author, item_id, "Your post/comment was removed automatically: our scanner found it very likely "
                               "to contain offensive language. A moderator can review it.")
    return status


def row(r):
    r["words"] = json.loads(r["words"]) if r.get("words") else []
    return r


def create(kind, author, text, parent_id=None):
    if author not in USERS:
        raise HTTPException(400, "Unknown user")
    text = text.strip()
    if not text or len(text) > 1000:
        raise HTTPException(400, "Text must be 1-1000 characters")
    iid = db.x("INSERT INTO items(kind,parent_id,author,text,created_at,status) VALUES(?,?,?,?,?,'pending')",
               (kind, parent_id, author, text, time.time()))
    event(iid, "submitted", f"{kind} by {author}")
    scan_into(iid, text, author)
    return row(db.q("SELECT * FROM items WHERE id=?", (iid,))[0])


@app.get("/api/users")
def users():
    return USERS


@app.get("/api/config")
def config():
    return {**S["th"], "model": S["scorer"].name}


@app.post("/api/posts")
def new_post(p: PostIn):
    return create("post", p.author, p.text)


@app.post("/api/comments")
def new_comment(c: CommentIn):
    parent = db.q("SELECT id,kind FROM items WHERE id=?", (c.post_id,))
    if not parent or parent[0]["kind"] != "post":
        raise HTTPException(404, "Post not found")
    return create("comment", c.author, c.text, c.post_id)


@app.get("/api/feed")
def feed(viewer: str):
    def visible(i):
        return i["status"] in ("published", "flagged", "restored") or i["author"] == viewer

    def slim(i):  # the feed never exposes scores or explanations
        return {k: i[k] for k in ("id", "kind", "author", "text", "created_at", "status")}
    items = [i for i in db.q("SELECT * FROM items ORDER BY created_at") if visible(i)]
    comments = {}
    for i in items:
        if i["kind"] == "comment":
            comments.setdefault(i["parent_id"], []).append(slim(i))
    posts = [dict(slim(i), comments=comments.get(i["id"], [])) for i in items if i["kind"] == "post"]
    return list(reversed(posts))


@app.get("/api/alerts")
def alerts(user: str, mark_read: bool = False):
    a = db.q("SELECT * FROM alerts WHERE user=? ORDER BY created_at DESC LIMIT 30", (user,))
    if mark_read:
        db.x("UPDATE alerts SET read=1 WHERE user=?", (user,))
    return a


@app.get("/api/admin/items")
def admin_items(status: str = "all"):
    sql = "SELECT * FROM items" + ("" if status == "all" else " WHERE status=?") + " ORDER BY created_at DESC"
    return [row(i) for i in db.q(sql, () if status == "all" else (status,))]


@app.get("/api/admin/items/{item_id}")
def admin_item(item_id: int):
    r = db.q("SELECT * FROM items WHERE id=?", (item_id,))
    if not r:
        raise HTTPException(404)
    return {**row(r[0]), "events": db.q("SELECT * FROM events WHERE item_id=? ORDER BY ts", (item_id,))}


@app.get("/api/admin/stats")
def stats():
    c = {s: 0 for s in ("published", "flagged", "removed", "restored")}
    for r in db.q("SELECT status, COUNT(*) n FROM items GROUP BY status"):
        c[r["status"]] = r["n"]
    overturned = db.q("SELECT COUNT(DISTINCT item_id) n FROM events WHERE event='restore'")[0]["n"]
    confirmed = db.q("SELECT COUNT(DISTINCT item_id) n FROM events WHERE event='confirm_removal'")[0]["n"]
    return {"total": sum(c.values()), **c, "overturned_by_moderators": overturned,
            "confirmed_by_moderators": confirmed, "thresholds": S["th"], "model": S["scorer"].name}


@app.post("/api/admin/items/{item_id}/action")
def action(item_id: int, a: ActionIn):
    r = db.q("SELECT * FROM items WHERE id=?", (item_id,))
    if not r:
        raise HTTPException(404)
    it = r[0]
    rules = {"restore": ("removed", "restored", "A moderator reviewed your removed post/comment and restored it."),
             "confirm_removal": ("flagged", "removed", "A moderator reviewed your post/comment and removed it for offensive language."),
             "approve": ("flagged", "published", None)}
    if a.action not in rules:
        raise HTTPException(400, "Unknown action")
    need, new, msg = rules[a.action]
    if a.action == "confirm_removal" and it["status"] not in ("flagged", "removed"):
        raise HTTPException(409, "Only flagged items can be confirmed")
    if a.action != "confirm_removal" and it["status"] != need:
        raise HTTPException(409, f"Action '{a.action}' needs an item with status '{need}'")
    db.x("UPDATE items SET status=?, reviewed=1 WHERE id=?", (new, item_id))
    event(item_id, a.action, f"{it['status']} -> {new} by moderator")
    if msg and it["status"] != new:
        alert(it["author"], item_id, msg)
    return admin_item(item_id)


@app.post("/api/admin/rescan")
def rescan():
    """Re-run the scanner over everything not yet reviewed by a human, with the current model + thresholds."""
    changed = 0
    for i in db.q("SELECT id,text,author,status FROM items WHERE reviewed=0"):
        new = scan_into(i["id"], i["text"], i["author"], first=False, old_status=i["status"])
        changed += new != i["status"]
    return {"scanned": len(db.q("SELECT id FROM items WHERE reviewed=0")), "changed": changed}


SEED = [
    ("Nimal", "අද දවස ගොඩක් ලස්සනයි, සුභ දවසක් වේවා!", [("Kamala", "ඔබේ උදව්වට බොහොම ස්තුතියි."),
                                                     ("Saman", "උඹ හරි මෝඩයෙක්, කිසි දෙයක් දන්නේ නෑ.")]),
    ("Kamala", "අපේ ක්‍රිකට් කණ්ඩායම අද හොඳට ක්‍රීඩා කළා.", [("Ruwan", "ඔය කට වහගෙන ඉන්න බං, උඹට පිස්සුද?"),
                                                       ("Dilini", "මේ චිත්‍රපටය ගොඩක් හොඳයි, බලන්න කියලා මම නිර්දේශ කරනවා.")]),
    ("Saman", "අද උදේ කොළඹ ප්‍රදේශයේ තද වැසි තිබුණා.", [("Dilini", "මගේ අයියා මෝඩ වැඩක් කරලා වැටුණා, හැබැයි ලොකු අවුලක් නෑ.")]),
    ("Dilini", "මේ කෑම හරිම නරකයි, ආයෙත් එන්නේ නෑ.", [("Ruwan", "උඹ වගේ නාකි බල්ලෙක් මම දැකලා නෑ.")]),
]


@app.post("/api/admin/seed")
def seed():
    if db.q("SELECT id FROM items LIMIT 1"):
        raise HTTPException(409, "Database already has content")
    for author, text, comments in SEED:
        p = create("post", author, text)
        for ca, ct in comments:
            create("comment", ca, ct, p["id"])
    return {"seeded": len(db.q("SELECT id FROM items"))}


@app.get("/")
def home():
    return FileResponse(STATIC / "index.html")


@app.get("/admin")
def admin_page():
    return FileResponse(STATIC / "admin.html")
