"""Correctness gates: the classical implementation must reproduce user 814's published models
(Li & Alonso 2014, Tables 1, 2 and 4). Requires data/ (python data/fetch_data.py)."""
import csv, os, pytest
from rama_core import build_user, category_elements, term_elements

D = "data"
pytestmark = pytest.mark.skipif(not os.path.exists(f"{D}/profiles2014-100.csv"), reason="run data/fetch_data.py")

def _load():
    ex = {int(r["id"]): r for r in csv.DictReader(open(f"{D}/examples2014.csv", encoding="utf-8"))}
    cats = {int(r["example_id"]): r["categories"].split("|")
            for r in csv.DictReader(open(f"{D}/examples2014_categories_reconstructed.csv", encoding="utf-8"))}
    u814 = [(int(r[1]), int(r[2])) for r in csv.reader(open(f"{D}/profiles2014-100.csv")) if r and r[0] == "814"]
    return ex, cats, u814

def test_general_model_reproduces_verified_weights():
    ex, cats, u814 = _load()
    G = build_user(u814, lambda e: category_elements(cats[e]))
    published = {"aquariums": 0.25, "culturalcenter": 0.25, "karaoke": 0.25, "rafting": 0.25, "zoos": 0.25,
                 "amusementparks": 0.5, "skiresorts": 0.5, "shoppingcenters": 0.5625,
                 "horsebackriding": -0.25, "wineries": -1 / 12, "churches": 0.34375}
    for c, w in published.items():
        assert G[c] == pytest.approx(w, abs=1e-4), c

def test_specific_model_close_to_table_2():
    ex, cats, u814 = _load()
    S = build_user(u814, lambda e: term_elements(ex[e]["title"], ex[e]["description"]))
    t2 = {"chicago": 0.302, "museum": 0.269, "art": 0.174, "fe": 0.17, "santa": 0.17, "illinois": 0.131,
          "history": 0.109, "world": 0.098, "american": 0.096, "gallery": 0.086, "pier": 0.083, "cafe": 0.073,
          "events": 0.072, "church": 0.072, "mile": 0.068, "mexico": 0.065, "side": 0.063, "goat": 0.06,
          "arts": 0.06, "collection": 0.059, "park": 0.058, "states": 0.058, "center": 0.055}
    mae = sum(abs(S.get(k, 0) - v) for k, v in t2.items()) / len(t2)
    top = set(sorted(S, key=S.get, reverse=True)[:23])
    assert mae <= 0.01 and len(top & set(t2)) >= 18
