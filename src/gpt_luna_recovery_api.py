"""Official GPT Luna recovery transport; capped retries delegated to episode runner."""
import os,json,threading,time,urllib.request,urllib.error,uuid
from pathlib import Path
from datetime import datetime,timezone
import run_current_multiround as control
ROOT=Path(__file__).resolve().parent
OUT=None
LOCK=threading.Lock()
COMMITTED=0.0
CONFIG=dict(model='gpt-6-luna',reasoning_effort='low',max_completion_tokens=8192,response_format={'type':'json_object'})
CAP_USD=.50

def call(messages,purpose='check'):
    global COMMITTED
    if control.STOP.is_set(): raise RuntimeError('Run already stopped')
    key=os.environ.get('OPENAI_API_KEY','').strip()
    if not key or any(ch.isspace() for ch in key): raise ValueError('Invalid credential format')
    size=sum(len(m['content'].encode()) for m in messages)
    if size>20000: control.stop('Input byte limit');raise ValueError('Input byte limit')
    reserve=((size+1024)*.125+8192*.5)/1e6
    ident=uuid.uuid4().hex
    with LOCK:
        if COMMITTED+reserve>CAP_USD: control.stop('Local USD spend cap');raise ValueError('Local USD spend cap')
        COMMITTED+=reserve
        with (OUT/'usd_budget.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(id=ident,purpose=purpose,status='reserved',usd=reserve))+'\n')
    payload=dict(CONFIG,messages=messages,stream=False,store=False)
    req=urllib.request.Request('https://api.openai.com/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
    start=time.perf_counter()
    try:
        with urllib.request.build_opener().open(req,timeout=90) as response: result=json.load(response)
    except Exception as exc:
        detail=exc.read(4096).decode(errors='replace') if isinstance(exc,urllib.error.HTTPError) else str(exc)
        detail=detail.replace(key,'[REDACTED]')
        status=exc.code if isinstance(exc,urllib.error.HTTPError) else None
        transient=(status in (408,409,429) or status is not None and status>=500 or status is None and isinstance(exc,(urllib.error.URLError,ConnectionError,TimeoutError)))
        if any(word in detail.lower() for word in ('insufficient_quota','invalid_api_key','must be verified')):transient=False
        with LOCK:
            with (OUT/'usd_budget.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(id=ident,status='failed_reservation_retained',retryable=transient,http_status=status,detail=detail))+'\n')
        if transient:
            # The episode runner performs at most two retries of identical input,
            # waiting 3 and 6 seconds. No valid answer is ever retried.
            raise RuntimeError('GPT network error: '+detail) from None
        control.stop('GPT request failed: '+detail)
        raise RuntimeError('GPT request failed: '+detail) from None
    usage=result.get('usage',{})
    inp=usage.get('prompt_tokens',0);out=usage.get('completion_tokens',0)
    details=usage.get('prompt_tokens_details',{})
    cached=details.get('cached_tokens',0);write=details.get('cache_write_tokens',0)
    upper=(inp*.125+out*.5)/1e6
    # Report list estimate separately; cache writes conservatively cost .125/M.
    nominal=(inp*.1+out*.5)/1e6
    aware=((inp-cached-write)*.1+cached*.01+write*.125+out*.5)/1e6
    with LOCK:
        COMMITTED+=upper-reserve
        with (OUT/'usd_budget.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(id=ident,status='settled',reserved_usd=reserve,conservative_usd=upper,nominal_usd=nominal,cache_aware_usd=aware))+'\n')
    return dict(request=payload,response=result,seconds=time.perf_counter()-start,estimated_list_price_usd=nominal,estimated_cache_aware_cost_usd=aware,transport_path='official_openai_existing_system_proxy')
