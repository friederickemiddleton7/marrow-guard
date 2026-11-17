import json
m = json.load(open("/root/hermes-work/marrow-guard/model/profile.json"))
print("features:", len(m["features"]))
print("cap:", m["cap"], "threshold:", m["threshold"])
for i, p in enumerate(m["profiles"]):
    print(f"profile {i} members={p['members']}")
    print("   center:", [round(x, 2) for x in p["center"]])
    print("   scale :", [round(x, 2) for x in p["scale"]])
print("global scale:", [round(x, 2) for x in m["scale"]])
print("names:", m["features"])