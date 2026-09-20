"""Small OpenAI-compatible client usable with local serving or another provider."""
import json
import os
import time
import urllib.error
import urllib.request


def completion(endpoint,model,messages,max_tokens,temperature=.7,seed=None,schema=None):
    payload=dict(model=model,messages=messages,max_tokens=max_tokens,temperature=temperature,
                 chat_template_kwargs={'enable_thinking':False})
    if seed is not None:payload['seed']=seed
    if schema:payload['response_format']=dict(type='json_schema',json_schema=dict(name='verdict',schema=schema,strict=True))
    headers={'Content-Type':'application/json'}
    if os.environ.get('PAINTER_API_KEY'):headers['Authorization']='Bearer '+os.environ['PAINTER_API_KEY']
    request=urllib.request.Request(endpoint.rstrip('/')+'/chat/completions',data=json.dumps(payload).encode(),headers=headers)
    start=time.monotonic()
    try:
        with urllib.request.urlopen(request,timeout=900) as response:body=json.load(response)
        return dict(ok=True,response=body,seconds=time.monotonic()-start,model=model)
    except urllib.error.HTTPError as exc:
        return dict(ok=False,http_status=exc.code,error=exc.read().decode(errors='replace')[:4000],seconds=time.monotonic()-start,model=model)
    except Exception as exc:return dict(ok=False,error=f'{type(exc).__name__}: {exc}',seconds=time.monotonic()-start,model=model)
