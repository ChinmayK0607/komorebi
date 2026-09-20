import json,os,sys,time,urllib.request
pid=int(sys.argv[1]);expected=sys.argv[2] if len(sys.argv)>2 else 'base';deadline=time.monotonic()+1200
while time.monotonic()<deadline:
    os.kill(pid,0)
    try:
        with urllib.request.urlopen('http://127.0.0.1:8000/v1/models',timeout=3) as response:d=json.load(response)
        if any(m['id']==expected for m in d['data']):print(expected+' endpoint ready',flush=True);break
    except Exception:pass
    time.sleep(3)
else:raise RuntimeError(expected+' startup exceeded 20 minutes')
