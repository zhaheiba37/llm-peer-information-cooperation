"""Small official Bailian client; credentials never enter logs or exceptions."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
import threading
import uuid
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT/'qwen_api_config.json'
LEDGER_LOCK = threading.Lock()


def ledger_committed(entries):
    """Successful settled usage replaces its reservation; failures retain theirs."""
    reserved = sum(x.get('reserved_upper_cost_cny', 0) for x in entries)
    settlements = {}
    for row in entries:
        if row.get('status') == 'settled':
            settlements[row['reservation_id']] = row
    return reserved + sum(x['actual_cost_cny']-x['released_reserved_cny'] for x in settlements.values())


def load_key(config):
    value = os.environ.get('DASHSCOPE_API_KEY', '').strip()
    source = 'environment'
    if not value:
        raise ValueError('Set DASHSCOPE_API_KEY in the environment before API execution.')
        source = 'local_file'
    # Do not assume keys contain only hex/base64url characters: new key
    # formats may contain punctuation. Accept one bare printable credential.
    if (not value.startswith('sk-') or len(value) < 15 or not value.isascii()
            or any(ch.isspace() or ord(ch) < 33 or ord(ch) == 127 for ch in value)):
        raise ValueError('Credential format invalid or ambiguous; no secret was printed.')
    return value, source


def call(messages, purpose='connection_check'):
    config = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
    if config['base_url'] != 'https://dashscope.aliyuncs.com/compatible-mode/v1':
        raise ValueError('Unexpected endpoint; verify the region before sending credentials.')
    key, source = load_key(config)
    input_bytes = sum(len(x['content'].encode('utf-8')) for x in messages)
    if input_bytes > config['max_input_bytes_per_request']:
        raise ValueError('Request exceeds the local input limit.')
    ledger_name = config.get('budget_ledger_file', 'qwen_api_usage.jsonl')
    if Path(ledger_name).name != ledger_name:
        raise ValueError('Budget ledger must be a filename inside the client directory.')
    ledger = ROOT/ledger_name
    # Byte count plus message overhead is a deliberately conservative input bound.
    reserved = ((input_bytes + 1024)*config['price_cny_per_million_input']
                + config['max_tokens']*config['price_cny_per_million_output'])/1_000_000
    payload = {k: config[k] for k in ('model','enable_thinking','temperature','top_p','max_tokens')}
    if config.get('enable_thinking') and 'thinking_budget' in config:
        payload['thinking_budget'] = config['thinking_budget']
    if 'response_format' in config:
        payload['response_format'] = config['response_format']
    payload.update(messages=messages, stream=False)
    entry = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), purpose=purpose,
                 model=config['model'], credential_source=source,
                 reservation_id=uuid.uuid4().hex, reserved_upper_cost_cny=reserved, status='reserved')
    # Reserve before transmission; failed calls also consume the local allowance.
    # One runner process owns the ledger; its concurrent threads share this lock.
    with LEDGER_LOCK:
        previous = [json.loads(line) for line in ledger.read_text(encoding='utf-8').splitlines()] if ledger.exists() else []
        if ledger_committed(previous) + reserved > config['local_spend_cap_cny']:
            raise ValueError('Local spend cap reached. No request sent.')
        with ledger.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(entry)+'\n')
    request = urllib.request.Request(config['base_url']+'/chat/completions',
            data=json.dumps(payload).encode('utf-8'),
            headers={'Authorization':'Bearer '+key, 'Content-Type':'application/json'})
    start = time.perf_counter()
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=config['timeout_seconds']) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read(4096).decode('utf-8', errors='replace').replace(key,'[REDACTED]')
        raise RuntimeError(f'Bailian HTTP {exc.code}: {detail}') from None
    except urllib.error.URLError as exc:
        detail = str(exc.reason).replace(key,'[REDACTED]')
        raise RuntimeError(f'Bailian network error: {detail}') from None
    except Exception as exc:
        raise RuntimeError(f'Bailian connection error: {type(exc).__name__}') from None
    duration = time.perf_counter()-start
    usage = result.get('usage',{})
    cost = (usage.get('prompt_tokens',0)*config['price_cny_per_million_input']
            + usage.get('completion_tokens',0)*config['price_cny_per_million_output'])/1_000_000
    if config.get('settle_usage_budget') and all(type(usage.get(k)) is int and usage[k] >= 0 for k in ('prompt_tokens', 'completion_tokens')):
        settlement = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), status='settled',
            reservation_id=entry['reservation_id'], reserved_upper_cost_cny=0,
            released_reserved_cny=reserved, actual_cost_cny=cost)
        with LEDGER_LOCK:
            with ledger.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(settlement)+'\n')
    safe = dict(request=payload, response=result, seconds=duration,
                transport_path='direct_official_endpoint_no_system_proxy',
                estimated_list_price_cny=cost, credential_source=source,
                billing_note='Estimate at uncached list prices, not a balance query or billing receipt.')
    # Redact defensively before any result can be saved by the caller.
    return json.loads(json.dumps(safe,ensure_ascii=False).replace(key,'[REDACTED]'))


def smoke():
    import prompts_v7 as p
    item = p.packaged('payoff','D_LOW_N_LOW', contributions=dict(A=4,B=10,C=10,D=10))
    result = call(item['messages'])
    result['expected_not_for_model'] = item['expected_not_for_model']
    folder = ROOT/'runs'/('qwen_api_connection_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_UTC'))
    folder.mkdir(parents=True, exist_ok=False)
    (folder/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    choice = result['response']['choices'][0]
    print(json.dumps(dict(model=result['response'].get('model'),finish_reason=choice.get('finish_reason'),
          output=choice['message'].get('content'), usage=result['response'].get('usage'),
          expected=item['expected_not_for_model'],estimated_list_price_cny=result['estimated_list_price_cny'],
          seconds=result['seconds'],saved_to=str(folder)),ensure_ascii=True))


if __name__=='__main__':
    smoke()
