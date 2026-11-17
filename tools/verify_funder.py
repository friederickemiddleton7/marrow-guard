import json, urllib.request

WALLET = "0x93e5e2fe82e2f7f23d8063cb039dfcb98806fe0d"
USDC = "0x833589fcd6edb6e08f4c7c32d4f71b54bda02913"
data = "0x70a08231" + "0" * 24 + WALLET[2:].lower()
body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "eth_call",
                   "params": [{"to": USDC, "data": data}, "latest"]}).encode()

for url in ["https://base-rpc.publicnode.com", "https://mainnet.base.org", "https://base.llamarpc.com"]:
    try:
        req = urllib.request.Request(url, body, {"Content-Type": "application/json",
                                                "User-Agent": "curl/8.5.0"})
        d = json.load(urllib.request.urlopen(req, timeout=25))
        print(url, "->", d.get("result"), "=", int(d["result"], 16) / 1e6, "USDC")
    except Exception as e:
        print(url, "ERR", repr(e)[:120])

# also block number for the snapshot context
try:
    req = urllib.request.Request("https://base-rpc.publicnode.com",
                                 json.dumps({"jsonrpc": "2.0", "id": 2, "method": "eth_blockNumber", "params": []}).encode(),
                                 {"Content-Type": "application/json", "User-Agent": "curl/8.5.0"})
    print("block now:", int(json.load(urllib.request.urlopen(req, timeout=20))["result"], 16))
except Exception as e:
    print("block ERR", repr(e)[:100])