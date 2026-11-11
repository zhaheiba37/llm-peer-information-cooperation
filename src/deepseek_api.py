"""Official DeepSeek-only client; thread-safe conservative CNY budget ledger."""
from datetime import datetime, timezone, timedelta
import json
import os
from pathlib import Path
import threading
import time
import urllib.error
import urllib.request
import uuid

from qwen_api import ledger_committed

ROOT=Path(__file__).resolve().parent
CONFIG_PATH=ROOT/'deepseek_pilot_low_config.json'
LEDGER_LOCK=threading.Lock()


def load_key():
    value=os.environ.get('DEEPSEEK_API_KEY','').strip()
    source='environment'
    if not value:
        raise ValueError('Set DEEPSEEK_API_KEY in the environment before API execution.')
        source='local_file'
    if not value.startswith('sk-') or len(value)<15 or not value.isascii() or any(ch.isspace() for ch in value):
        raise ValueError('DeepSeek credential format invalid; no secret printed.')
    return value,source


def transmit(path,key,payload=None,timeout=120):
    if path not in ('/models','/user/balance','/chat/completions'):
        raise ValueError('Endpoint is not on the official allowlist.')
    request=urllib.request.Request('https://api.deepseek.com'+path,
        data=json.dumps(payload).encode('utf-8') if payload is not None else None,
        headers={'Authorization':'Bearer '+key,'Accept':'application/json','Content-Type':'application/json'},
        method='POST' if payload is not None else 'GET')
    try:
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request,timeout=timeout) as stream:result=json.load(stream)
    except urllib.error.HTTPError as exc:
        detail=exc.read(4096).decode('utf-8',errors='replace').replace(key,'[REDACTED]')
        raise RuntimeError(f'DeepSeek HTTP {exc.code}: {detail}') from None
    except urllib.error.URLError as exc:
        detail=str(exc.reason).replace(key,'[REDACTED]')
        raise RuntimeError('DeepSeek network error: '+detail) from None
    except Exception as exc:
        raise RuntimeError('DeepSeek connection error: '+type(exc).__name__) from None
    return json.loads(json.dumps(result,ensure_ascii=False).replace(key,'[REDACTED]'))


def get_balance():
    key,source=load_key()
    return dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),credential_source=source,
        endpoint='https://api.deepseek.com/user/balance',response=transmit('/user/balance',key,None,30))


def prices(config,when):
    beijing=when.astimezone(timezone(timedelta(hours=8)))
    minute=beijing.hour*60+beijing.minute
    # Treat weekday holiday peaks conservatively as peaks; never underestimate
    # holiday discounts. The actual run date (2026-11-05) is a normal Thursday.
    peak=beijing.weekday()<5 and (9*60<=minute<12*60 or 14*60<=minute<18*60)
    band='peak' if peak else 'off_peak'
    return band,config['price_cny_per_million'][band]


def estimate(usage,rates):
    inp=usage.get('prompt_tokens',0);out=usage.get('completion_tokens',0)
    upper=(inp*rates['input_cache_miss']+out*rates['output'])/1e6
    hit=usage.get('prompt_cache_hit_tokens');miss=usage.get('prompt_cache_miss_tokens')
    if type(hit) is int and type(miss) is int and hit>=0 and miss>=0 and hit+miss==inp:
        aware=(hit*rates['input_cache_hit']+miss*rates['input_cache_miss']+out*rates['output'])/1e6
    else:aware=upper
    return upper,aware


def call(messages,purpose='deepseek_check'):
    config=json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
    if config['base_url']!='https://api.deepseek.com' or config['model']!='deepseek-flash':
        raise ValueError('Unexpected DeepSeek endpoint/model; no request sent.')
    key,source=load_key()
    input_bytes=sum(len(x['content'].encode('utf-8')) for x in messages)
    if input_bytes>config['max_input_bytes_per_request']:raise ValueError('Input byte limit exceeded.')
    payload={k:config[k] for k in ('model','thinking','reasoning_effort','top_p','max_tokens','response_format')}
    payload.update(messages=messages,stream=False)
    # temperature is intentionally omitted: official thinking mode ignores it.
    filename=config['budget_ledger_file']
    if Path(filename).name!=filename:raise ValueError('Invalid ledger filename.')
    ledger=ROOT/filename
    worst=config['price_cny_per_million']['peak']
    reserve=((input_bytes+1024)*worst['input_cache_miss']+config['max_tokens']*worst['output'])/1e6
    entry=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),reservation_id=uuid.uuid4().hex,
        purpose=purpose,model=config['model'],reasoning_effort=config['reasoning_effort'],
        status='reserved',reserved_upper_cost_cny=reserve)
    with LEDGER_LOCK:
        entries=[json.loads(x) for x in ledger.read_text(encoding='utf-8').splitlines()] if ledger.exists() else []
        if ledger_committed(entries)+reserve>config['local_spend_cap_cny']:
            raise ValueError('DeepSeek local spend cap reached. No request sent.')
        with ledger.open('a',encoding='utf-8') as stream:stream.write(json.dumps(entry)+'\n')
    begin=datetime.now(timezone.utc);band,rates=prices(config,begin)
    started=time.perf_counter()
    result=transmit('/chat/completions',key,payload,config['timeout_seconds'])
    seconds=time.perf_counter()-started
    usage=result.get('usage',{})
    cost,aware=estimate(usage,rates)
    if all(type(usage.get(k)) is int and usage[k]>=0 for k in ('prompt_tokens','completion_tokens')):
        settle=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),reservation_id=entry['reservation_id'],
            status='settled',reserved_upper_cost_cny=0,released_reserved_cny=reserve,actual_cost_cny=cost,
            cache_aware_estimate_cny=aware,price_band_at_start=band)
        with LEDGER_LOCK:
            with ledger.open('a',encoding='utf-8') as stream:stream.write(json.dumps(settle)+'\n')
    safe=dict(request=payload,response=result,seconds=seconds,estimated_list_price_cny=cost,
        estimated_cache_aware_cost_cny=aware,price_band_at_start=band,rates_cny_per_million=rates,
        credential_source=source,transport_path='direct_official_deepseek',
        billing_note='Completion tokens already include reasoning, counted once. Weekday holiday peak rates conservatively overestimated; actual balance is separate.')
    return json.loads(json.dumps(safe,ensure_ascii=False).replace(key,'[REDACTED]'))


if __name__=='__main__':
    result=get_balance()
    (ROOT/'deepseek_balance_before_pilot_20261105.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result['response'],ensure_ascii=False,indent=2))
