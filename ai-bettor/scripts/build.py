"""Build all remaining AI BETTOR files."""
import os
WS = "C:/Users/User/Documents/Team-Bettor/ai-bettor"
def w(f,c):
    os.makedirs(os.path.dirname(WS+"/"+f),exist_ok=True)
    open(WS+"/"+f,"w",encoding="utf-8").write(c)
    print("  "+f)

print("=== BUILDING AI BETTOR ===")
