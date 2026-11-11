"""Prepare a versioned experiment schedule without calling any model."""
import json
import random
from pathlib import Path
import prompts_v7 as p

MODEL_SLOTS = ('GPT_family', 'Qwen_family', 'DeepSeek_family')
SCHEDULE_SEED = 20261104


def build_schedule():
    stages = {'main': [], 'self_max': [], 'numeric_anchor': [], 'optional_two_to_one': [], 'optional_own_history': []}
    lookup = {}

    def add(stage, slot, condition, rep, **settings):
        item = p.packaged('decision', condition, **settings)
        row = {'trial_id': f'{stage}/{slot}/{len(stages[stage])+1:05d}',
               'model_slot': slot, 'condition': condition, 'prompt_id': item['prompt_id'],
               'settings': item['settings'], 'replicate': rep}
        stages[stage].append(row)
        if stage == 'main':
            lookup[(slot, row['prompt_id'], rep)] = row['trial_id']
        return row

    for slot in MODEL_SLOTS:
        for template in p.RULES:
            for condition, (_, norm) in p.CONDITIONS.items():
                for own in (5,):
                    orders = ('history_first', 'statement_first') if norm != 'none' else ('history_first',)
                    for order in orders:
                        for rep in range(15 if norm != 'none' else 30):
                            add('main', slot, condition, rep, template=template, order=order, own_previous=own)
        for condition, (_, norm) in p.CONDITIONS.items():
            for own in (5,):
                orders = ('history_first', 'statement_first') if norm != 'none' else ('history_first',)
                for order in orders:
                    for rep in range(9 if norm != 'none' else 18):
                        settings = dict(template='T1', order=order, own_previous=own)
                        row = add('self_max', slot, condition, rep, goal='self_max', **settings)
                        baseline = p.packaged('decision', condition, **settings)
                        row['neutral_comparison_trial_id'] = lookup[(slot, baseline['prompt_id'], rep)]
        for condition in ('D_HIGH_N_ABSENT', 'D_LOW_N_ABSENT'):
            behavior = p.CONDITIONS[condition][0]
            positive = 'D_HIGH_N_HIGH' if behavior == 'high' else 'D_LOW_N_HIGH'
            negative = 'D_HIGH_N_LOW' if behavior == 'high' else 'D_LOW_N_LOW'
            for code in (0, 10):
                for own in (5,):
                    for order in ('history_first', 'statement_first'):
                        for rep in range(9):
                            settings = dict(template='T1', order=order, own_previous=own)
                            row = add('numeric_anchor', slot, condition, rep, anchor=code, **settings)
                            for key, comparison in [('norm_positive_trial_id', positive), ('norm_negative_trial_id', negative)]:
                                item = p.packaged('decision', comparison, **settings)
                                row[key] = lookup[(slot, item['prompt_id'], rep)]
        for condition, (_, norm) in p.CONDITIONS.items():
            for own in (0, 10):
                orders = ('history_first', 'statement_first') if norm != 'none' else ('history_first',)
                for order in orders:
                    for rep in range(6 if norm != 'none' else 12):
                        row = add('optional_own_history', slot, condition, rep, template='T1', order=order, own_previous=own)
                        baseline = p.packaged('decision', condition, template='T1', order=order, own_previous=5)
                        row['own5_comparison_trial_id'] = lookup[(slot, baseline['prompt_id'], rep)]
        for condition in ('D_HIGH_N_HIGH', 'D_HIGH_N_LOW', 'D_LOW_N_HIGH', 'D_LOW_N_LOW'):
            for actor in p.PEERS:
                for view in p.PEERS:
                    for order in ('history_first', 'statement_first'):
                        add('optional_two_to_one', slot, condition, 0, template='T1', order=order,
                            own_previous=5, minority_actor=actor, minority_view=view)

    episodes = []
    for slot in MODEL_SLOTS:
        for condition in ('D_LOW_N_HIGH', 'D_LOW_N_LOW', 'D_LOW_N_ABSENT'):
            norm = p.CONDITIONS[condition][1]
            for own in (5,):
                orders = ('history_first', 'statement_first') if norm != 'none' else ('history_first',)
                for order in orders:
                    for rep in range(6 if norm != 'none' else 12):
                        start = p.packaged('decision', condition, template='T1', order=order, own_previous=own)
                        episodes.append({'episode_id': f'continuation/{slot}/{len(episodes)+1:04d}',
                                         'model_slot': slot, 'condition': condition, 'own_previous': own,
                                         'order': order, 'replicate': rep, 'template': 'T1',
                                         'initial_prompt_id': start['prompt_id'],
                                         'decision_rounds': list(range(2, p.TOTAL_ROUNDS+1)),
                                         'peer_policy': 'B=C=D=0 in every continuation round',
                                         'history_update': 'Append actual A decisions and computed payoffs; no chat carryover.'})

    eligibility = []
    kinds = ('payoff', 'structure', 'information')
    vectors = ({'A': 0, 'B': 0, 'C': 0, 'D': 0},
               {'A': 4, 'B': 10, 'C': 10, 'D': 10},
               {'A': 4, 'B': 2, 'C': 5, 'D': 8})
    for slot in MODEL_SLOTS:
        for template_index, template in enumerate(p.RULES):
            for condition_index, (condition, (_, norm)) in enumerate(p.CONDITIONS.items()):
                for kind_index, kind in enumerate(kinds):
                    own = 5
                    total = template_index + condition_index + kind_index
                    order = 'statement_first' if norm != 'none' and total % 2 else 'history_first'
                    settings = dict(template=template, order=order, own_previous=own)
                    if kind == 'payoff':
                        settings['contributions'] = vectors[(template_index + condition_index) % 3]
                    item = p.packaged(kind, condition, **settings)
                    eligibility.append({'check_id': f'eligibility/{slot}/{len(eligibility)+1:04d}',
                                        'model_slot': slot, 'condition': condition, 'kind': kind,
                                        'prompt_id': item['prompt_id']})
    pilot = []
    for condition, (_, norm) in p.CONDITIONS.items():
        for own in (5,):
            orders = ('history_first', 'statement_first') if norm != 'none' else ('history_first',)
            for order in orders:
                for rep in range(6 if norm != 'none' else 12):
                    item = p.packaged('decision', condition, template='T1', order=order, own_previous=own)
                    pilot.append({'trial_id': f'pilot/{len(pilot)+1:03d}', 'condition': condition,
                                  'prompt_id': item['prompt_id'], 'replicate': rep})

    rng = random.Random(SCHEDULE_SEED)
    for name, rows in stages.items():
        rng.shuffle(rows)
        for index, row in enumerate(rows):
            row['request_order'] = index
            row['planned_sampling_seed'] = rng.randrange(1, 2**31)
    rng.shuffle(episodes)
    rng.shuffle(eligibility)
    rng.shuffle(pilot)
    counts = {name: len(rows) for name, rows in stages.items()}
    counts.update(default_decision_calls=sum(len(stages[k]) for k in ('main', 'self_max', 'numeric_anchor')),
                  optional_continuation_runs=len(episodes),
                  optional_continuation_calls=sum(len(x['decision_rounds']) for x in episodes))
    counts['eligibility_check_calls'] = len(eligibility)
    counts['pilot_decision_calls_one_debug_model'] = len(pilot)
    counts['default_calls_including_eligibility_not_pilot'] = counts['default_decision_calls'] + len(eligibility)
    counts['all_decision_calls_if_all_options_run'] = counts['default_decision_calls'] + counts['optional_two_to_one'] + counts['optional_own_history'] + counts['optional_continuation_calls']
    return {'version': p.VERSION, 'status': 'design_only_not_run_not_preregistered',
            'main_own_previous': 5, 'own_history_supplement_levels': [0, 10],
            'schedule_seed': SCHEDULE_SEED, 'model_slots': MODEL_SLOTS,
            'model_ids': 'Unset: exact official API IDs, modes and supported parameters must be frozen before collection.',
            'sampling_note': 'Planned sampling seeds are used only if the provider supports them; they do not guarantee reproducibility across APIs.',
            'counts': counts, 'stages': stages, 'eligibility_checks': eligibility,
            'pilot_one_debug_model': pilot, 'optional_continuation_runs': episodes}


def export():
    root = Path(__file__).resolve().parent.parent / '最小实验方案'
    payload = build_schedule()
    (root/'实验调用安排_v7_草案.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    p.export()
    print(json.dumps(payload['counts']))


if __name__ == '__main__':
    export()
