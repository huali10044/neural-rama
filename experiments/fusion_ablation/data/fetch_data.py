"""Download the TREC 2014 Contextual Suggestion input files and verify them. Not committed to the repo."""
import csv, hashlib, os, urllib.request
BASE = "https://raw.githubusercontent.com/tthonet/composite-contextual-suggestion/master/data"
SHA256 = {  # the exact files the reported results were produced from
    "examples2014.csv": "f008ff411298fc7f1ebfe5a89759a30589edc7bd8edd115103d7a9b6931992fc",
    "profiles2014-70.csv": "199a765b522d05c6ebc0519e374e5ac69d2ffaeb63f91d02dce6f34db04ecdf8",
    "profiles2014-100.csv": "69c84c1e9a21b19db5f679d47e285ed0c1d160c8b3914e2488a1ba130fa84ec0",
}
FILES = list(SHA256)

def main(dest="data"):
    os.makedirs(dest, exist_ok=True)
    for f in FILES:
        path = os.path.join(dest, f)
        if not os.path.exists(path):
            urllib.request.urlretrieve(f"{BASE}/{f}", path); print("downloaded", f)
        digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
        assert digest == SHA256[f], f"{f}: checksum mismatch — not the file the results were produced from"
    ex = list(csv.DictReader(open(os.path.join(dest, "examples2014.csv"), encoding="utf-8")))
    assert len(ex) == 100 and ex[1]["title"] == "Topolobampo/Frontera Grill", "examples2014.csv mismatch"
    def rows(f): return [r for r in csv.reader(open(os.path.join(dest, f))) if r and r[0].strip().isdigit()]
    p70, p100 = rows("profiles2014-70.csv"), rows("profiles2014-100.csv")
    u70, u100 = {r[0] for r in p70}, {r[0] for r in p100}
    assert (len(u70), len(p70)) == (183, 183 * 70), "profiles2014-70.csv mismatch"
    assert (len(u100), len(p100)) == (116, 116 * 100), "profiles2014-100.csv mismatch"
    assert not (u70 & u100), "user IDs overlap between files"
    ids = {r["id"] for r in ex}
    assert {r[1] for r in p70 + p100} <= ids, "profile references an unknown attraction"
    print("verified: checksums match; 100 examples, 183 x 70 + 116 x 100 ratings, 299 disjoint users")

if __name__ == "__main__":
    main()
