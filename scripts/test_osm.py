import requests
import json

def test_ohsome():
    print("[*] Testing ohsome API...")
    url = "https://api.ohsome.org/v1/elements/geometry"
    params = {
        "bboxes": "85.25,27.95,85.35,28.05",
        "time": "2026-07-27T00:00:00Z",
        "filter": "highway=*",
        "properties": "tags"
    }
    headers = {"User-Agent": "FloodRescueAI-Hackathon2026/1.0 (academic research; contact: hackathon@example.org)"}
    try:
        r = requests.post(url, data=params, headers=headers, timeout=20)
        print("ohsome Status:", r.status_code)
        if r.status_code == 200:
            print("ohsome Features:", len(r.json().get("features", [])))
            return True
        else:
            print("ohsome body:", r.text[:200])
    except Exception as e:
        print("ohsome error:", e)
    return False

def test_overpass():
    print("[*] Testing Overpass historical API...")
    url = "https://overpass-api.de/api/interpreter"
    query = """[out:json][timeout:25][date:"2026-07-27T00:00:00Z"];
(
  way["highway"](27.95,85.25,28.05,85.35);
);
out body;
>;
out skel qt;"""
    try:
        r = requests.post(url, data={"data": query}, headers={"User-Agent": "FloodRescueAI/1.0"}, timeout=20)
        print("Overpass Status:", r.status_code)
        if r.status_code == 200:
            print("Overpass Elements:", len(r.json().get("elements", [])))
            return True
    except Exception as e:
        print("Overpass error:", e)
    return False

if __name__ == "__main__":
    ok_oh = test_ohsome()
    ok_op = test_overpass()
