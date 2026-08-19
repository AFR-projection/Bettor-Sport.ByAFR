O=chr(60);C=chr(62);Q=chr(34);N=chr(10)
R=[]
a=lambda s="":R.append(s)
a(O+"!DOCTYPE html"+C)
a(O+"html lang="+Q+"en"+Q+C)
a(O+"head"+C)
a("<meta charset="+Q+"UTF-8"+Q+chr(62))
a("<meta name="+Q+"viewport"+Q+" content="+Q+"width=device-width,initial-scale=1"+Q+chr(62))
a(O+"title"+C+"AI BETTOR Dashboard"+O+"/title"+C)
a(O+"style"+C)
CSS="*{margin:0;padding:0;box-sizing:border-box}"
CSS+="body{font-family:system-ui,sans-serif;background:#0a0e17;color:#e5e7eb;min-height:100vh}"
CSS+=".hdr{background:#111827;padding:14px 24px;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid #374151}"
CSS+="h1{font-size:20px;background:linear-gradient(135deg,#3b82f6,#06b6d4);-webkit-background-clip:text;-webkit-text-fill-color:transparent}"
CSS+=".st{display:flex;gap:16px;font-size:11px;color:#9ca3af}"
CSS+=".nav{display:flex;gap:4px;padding:6px 24px;background:#111827;overflow-x:auto}"
CSS+="button{background:0 0;border:0;color:#9ca3af;padding:6px 12px;font-size:11px;cursor:pointer;border-radius:4px}"
CSS+="button:hover{color:#fff;background:#1f2937}"
CSS+=".main{padding:20px 24px;max-width:1400px;margin:0 auto}"
CSS+=".tab{display:none}.tab.on{display:block}"
CSS+=".card{background:#111827;border:1px solid #374151;border-radius:8px;padding:14px;margin:6px 0}"
CSS+="table{width:100%;border-collapse:collapse;font-size:11px}"
CSS+="th{color:#9ca3af;padding:6px;text-align:left;border-bottom:2px solid #374151}"
CSS+="td{padding:5px 6px;border-bottom:1px solid #1f2937}"
CSS+=".sc{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:8px;margin:10px 0}"
CSS+=".c1{background:#1f2937;border-radius:6px;padding:12px;text-align:center}"
CSS+=".pos{color:#10b981}.neg{color:#ef4444}"
CSS+=".bdg{display:inline-block;padding:2px 6px;border-radius:3px;font-size:9px;font-weight:600}.bdg.ok{background:rgba(16,185,129,.15);color:#10b981}.bdg.no{background:rgba(239,68,68,.15);color:#ef4444}"
CSS+=".spin{width:24px;height:24px;border:3px solid #374151;border-top-color:#3b82f6;border-radius:50%;animation:sp .7s linear infinite;margin:20px auto}"
CSS+="@keyframes sp{to{transform:rotate(360deg)}}"
a(CSS)
a(O+"/style"+C)
a(O+"/head"+C)
a(O+"body"+C)
a(O+"div class="+Q+"hdr"+C+O+"h1"+C+chr(9666)+" AI BETTOR"+O+"/h1"+C+O+"div class="+Q+"st"+C+O+"span"+C+"Online"+O+"/span"+O+"span id="+Q+"ts"+C+O+"/span"+C+O+"/div"+C+O+"/div"+C)
a(O+"div class="+Q+"nav"+C+" id="+Q+"nv"+C)
for i,(n,_) in enumerate([("Dashboard","0"),("Scanner","1"),("Matches","2"),("Predictions","3"),("Agents","4"),("Performance","5"),("Bankroll","6")]):
  c=" class="+Q+"on"+Q if i==0 else ""
  a(O+"button"+c+" onclick="+Q+"sw("+str(i)+")"+C+n+O+"/button"+C)
a(O+"/div"+C)
a(O+"div class="+Q+"main"+C+" id="+Q+"ct"+C)
for i in range(7):
  c=" class="+Q+"tab on"+Q if i==0 else " class="+Q+"tab"+Q
  a(O+"div id="+Q+"t"+str(i)+Q+c+C+O+"div id="+Q+"d"+str(i)+Q+" class="+Q+"card"+C+"Loading..."+O+"/div"+C+O+"/div"+C)
a(O+"/div"+C)
