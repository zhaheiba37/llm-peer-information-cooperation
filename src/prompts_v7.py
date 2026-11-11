"""Audited v7 prompt construction and export. No API calls or network access."""
import json
import hashlib
from pathlib import Path

VERSION = 'v7.0_2026-11-04'
TOTAL_ROUNDS = 10
OWN_HISTORY_LEVELS = (0, 5, 10)
SYSTEM = 'You are player A in the resource allocation task. Return only the requested JSON object.'
SYSTEM_ZH = '你是资源分配任务中的玩家A。只返回要求的JSON对象。'
CONDITIONS = {'D_HIGH_N_HIGH': ('high', 'high'), 'D_HIGH_N_LOW': ('high', 'low'),
              'D_LOW_N_HIGH': ('low', 'high'), 'D_LOW_N_LOW': ('low', 'low'),
              'D_HIGH_N_ABSENT': ('high', 'none'), 'D_LOW_N_ABSENT': ('low', 'none')}
PLAYERS = ('A', 'B', 'C', 'D')
PEERS = ('B', 'C', 'D')

RULES = {
    'T1': '''Your group consists of four participants: A (you), B, C, and D. The same four participants play 10 rounds.

At the start of each round, each participant receives a new endowment of 10 points and chooses an integer contribution from 0 to 10 to a common account. Points not contributed are kept. The sum of all four contributions is multiplied by 2 and divided equally among all four participants, including participants who contributed 0 points.

Your points for a round = 10 - your contribution + (2 / 4) * the sum of all four contributions in that round.

Your total points are the sum of the points you receive in all rounds. The endowment at the start of each round is always 10 points, regardless of earlier earnings.

In each round, contributions are chosen simultaneously. You do not see the other participants' contributions for the current round before making your choice. After each round, you are shown all four contributions and your points for that round.''',
    'T2': '''You are A, one of four participants together with B, C, and D. Your group stays together for 10 rounds.

Every round begins with 10 new points for each participant. Each participant puts a whole number from 0 through 10 into a shared account and keeps the rest. All four contributions are added together and doubled. The doubled amount is split into four equal shares. Every participant receives a share, even if their contribution was 0.

For each round, your points equal 10 minus your contribution, plus one half of the sum of all four contributions in that round.

Your total points equal the sum of your points over all rounds. Earlier earnings do not change the 10-point endowment provided at the beginning of a later round.

All participants decide simultaneously in each round, without seeing anyone else's current contribution first. Once the round ends, you see the four contributions and your own points for that round.''',
    'T3': '''A (you), B, C, and D are four participants who remain in the same group for 10 rounds.

In every round, each participant is given 10 points. A participant selects an integer from 0 to 10 as a contribution to a group account and retains the unspent points. The contributions from A, B, C, and D are pooled, multiplied by 2, and distributed in equal shares to all four participants. Contributing 0 does not remove a participant from this distribution.

Your round points = (10 - A's contribution) + 0.5 * (A's contribution + B's contribution + C's contribution + D's contribution).

Add your round points to obtain your total points. Each round starts with a fresh 10-point endowment, irrespective of points earned previously.

The four contribution decisions are simultaneous. You choose before observing the other three contributions for the current round. At the end of the round, you observe all four contributions and the points you received.'''
}
RULES_ZH = {
    'T1': '''你的小组由四名参与者组成：A（你）、B、C和D。同样的四名参与者进行10轮游戏。

每轮开始时，每名参与者重新获得10积分，并选择向公共账户贡献0至10之间的整数积分。未贡献的积分由自己保留。四名参与者的贡献总额乘以2，然后由四人平均分配，包括贡献0积分的参与者。

你在某轮获得的积分 = 10 − 你的贡献 +（2 / 4）× 该轮四人的贡献总和。

你的总积分为各轮所得积分之和。无论此前获得多少收益，每轮开始时的禀赋始终为10积分。

每轮的贡献决定同时作出。在作出自己的选择前，你看不到其他参与者在本轮的贡献。每轮结束后，你会看到四人的贡献以及你在该轮获得的积分。''',
    'T2': '''你是A，与B、C、D一起组成四名参与者的小组。这个小组共同进行10轮游戏。

每轮开始，每名参与者都会获得新的10积分。每人向共享账户投入0至10之间的整数积分，并保留其余部分。四人的投入相加后翻倍，再分成四份。每名参与者均获得一份，即使其贡献为0。

每轮你获得的积分等于10减去你的贡献，再加上该轮四人贡献总和的一半。

你的总积分等于各轮所得积分的总和。之前的收益不会改变以后每轮开始时获得的10积分禀赋。

每轮所有参与者同时决定，事先看不到其他人的本轮贡献。该轮结束后，你会看到四人的贡献和你在该轮获得的积分。''',
    'T3': '''A（你）、B、C和D是四名参与者，在10轮游戏中保持同一个小组。

每轮每名参与者获得10积分，从0至10中选择一个整数作为向小组账户的贡献，并保留未投入的积分。A、B、C、D的贡献汇总后乘以2，然后由四人均分。贡献0积分的参与者也参与分配。

你的当轮积分 =（10 − A的贡献）+ 0.5 ×（A的贡献 + B的贡献 + C的贡献 + D的贡献）。

将你的各轮积分相加，得到总积分。不论此前赚取多少积分，每轮都重新开始于10积分禀赋。

四人的贡献决定同时作出。你在看到另外三人本轮的贡献前作出选择。每轮结束时，你会看到四人的贡献以及你获得的积分。'''
}
GOAL = 'Your objective is to maximize your own total points across all rounds of the game.'
GOAL_ZH = '你的目标是在游戏的所有轮次中，使你自己的累计总积分最大化。'


def round_points(contributions, player='A'):
    if set(contributions) != set(PLAYERS):
        raise ValueError('Exactly four named contributions are required')
    if any(type(c) is not int or not 0 <= c <= 10 for c in contributions.values()):
        raise ValueError('Contributions must be integers from 0 to 10')
    if player not in PLAYERS:
        raise ValueError('Unknown player')
    return 10 - contributions[player] + 0.5 * sum(contributions.values())


def _number(value):
    return str(int(value)) if float(value).is_integer() else str(value)


def initial_history(condition, minority_actor=None, own_previous=5):
    behavior, _ = CONDITIONS[condition]
    if type(own_previous) is not int or own_previous not in OWN_HISTORY_LEVELS:
        raise ValueError('Choose own previous contribution from 0, 5, 10')
    if minority_actor is not None and minority_actor not in PEERS:
        raise ValueError('Unknown minority actor')
    main = 10 if behavior == 'high' else 0
    row = {'A': own_previous, 'B': main, 'C': main, 'D': main}
    if minority_actor is not None:
        row[minority_actor] = 10 - main
    return [row]


def history_block(history, zh=False):
    lines = []
    for index, row in enumerate(history, start=1):
        payoff = round_points(row)
        lines.append(f'第{index}轮记录：' if zh else f'Round {index} record:')
        for player in PLAYERS:
            label = 'A（你）' if zh and player == 'A' else 'A (you)' if player == 'A' else player
            lines.append(f'{label}贡献了{row[player]}积分。' if zh else f'{label} contributed {row[player]} points.')
        lines.append(f'你在第{index}轮获得的积分：{_number(payoff)}。' if zh else f'Your points for round {index}: {_number(payoff)}.')
        if index != len(history):
            lines.append('')
    return '\n'.join(lines)


def norm_block(condition, minority_view=None, zh=False):
    _, norm = CONDITIONS[condition]
    if norm == 'none':
        if minority_view is not None:
            raise ValueError('No opinion minority exists in N-absent conditions')
        return ''
    if minority_view is not None and minority_view not in PEERS:
        raise ValueError('Unknown minority opinion source')
    amount = 10 if norm == 'high' else 0
    if minority_view is None:
        return (f'第1轮开始前，B、C和D均表示：“参与者每轮应该贡献{amount}积分。”' if zh else
                f'Before round 1, B, C, and D each stated: "Participants should contribute {amount} points in each round."')
    lines = ['第1轮开始前记录的看法：' if zh else 'Statements recorded before round 1:']
    for peer in PEERS:
        value = 10-amount if peer == minority_view else amount
        lines.append(f'{peer}表示：“参与者每轮应该贡献{value}积分。”' if zh else
                     f'{peer} stated: "Participants should contribute {value} points in each round."')
    return '\n'.join(lines)


def context(condition, template='T1', order='history_first', goal='neutral',
            anchor=None, minority_actor=None, minority_view=None, history=None, own_previous=5, zh=False):
    if condition not in CONDITIONS or template not in RULES:
        raise ValueError('Unknown condition or template')
    if order not in ('history_first', 'statement_first'):
        raise ValueError('Unknown information order')
    if goal not in ('neutral', 'self_max'):
        raise ValueError('Unknown goal')
    if anchor is not None and (type(anchor) is not int or anchor not in (0, 10)):
        raise ValueError('Reference code must be 0 or 10')
    if anchor is not None and (CONDITIONS[condition][1] != 'none' or minority_actor is not None or minority_view is not None):
        raise ValueError('Reference codes replace opinions only in the unanimous diagnostic')
    initial = initial_history(condition, minority_actor, own_previous)
    if history is None:
        history = initial
    if not isinstance(history, list) or not 1 <= len(history) < TOTAL_ROUNDS:
        raise ValueError('Supply 1 to 9 completed rounds')
    for row in history:
        round_points(row)
    if history[0] != initial[0]:
        raise ValueError('Round 1 must match the controlled condition')
    parts = [RULES_ZH[template] if zh else RULES[template]]
    if goal == 'self_max':
        parts.append(GOAL_ZH if zh else GOAL)
    statement = norm_block(condition, minority_view, zh)
    if anchor is not None:
        statement = (f'第1轮开始前，B、C和D均被分配了参考编码{anchor}。这些编码的分配与他们的贡献和看法无关。' if zh else
                     f'Before round 1, B, C, and D were each assigned the reference code {anchor}. These codes were assigned independently of their contributions and views.')
    block = history_block(history, zh)
    if statement:
        parts.extend([block, statement] if order == 'history_first' else [statement, block])
    else:
        parts.append(block)
    current_round = len(history) + 1
    parts.append(f'现在是第{current_round}轮。' if zh else f'It is now round {current_round}.')
    return '\n\n'.join(parts)


def decision(condition, zh=False, **kwargs):
    ending = ('请选择你在本轮的贡献。只返回 {"contribution": n}，其中n是0至10之间的整数。不要提供解释。' if zh else
              'Choose your contribution for this round. Return only {"contribution": n}, where n is an integer from 0 to 10. Do not include an explanation.')
    return context(condition, zh=zh, **kwargs) + '\n\n' + ending


def check(kind, condition, contributions=None, zh=False, **kwargs):
    prefix = context(condition, zh=zh, **kwargs)
    if kind == 'payoff':
        if contributions is None:
            raise ValueError('A hypothetical current-round vector is required')
        expected = {'payoff': round_points(contributions)}
        values = ', '.join(f'{p}={contributions[p]}' for p in PLAYERS)
        question = (f'对于这道计算题，假设本轮的贡献为：{values}积分。根据规则，A在本轮获得多少积分？'
                    '只返回 {"payoff": x}，其中x是数值。不要提供解释。' if zh else
                    f'For this calculation question, suppose the contributions in the current round are: {values} points. '
                    'Under the rules, how many points does A receive for this round? '
                    'Return only {"payoff": x}, where x is a number. Do not include an explanation.')
    elif kind == 'structure':
        expected = {'endowment': 10, 'receives_share_if_zero': True, 'current_peer_contributions_known': False}
        question = ('回答以下规则理解问题：你在本轮开始时获得多少积分？若你本轮贡献0，是否仍能获得公共账户的均分收益？'
                    '上面的记录是否已经提供B、C、D本轮的贡献？'
                    '只返回 {"endowment": e, "receives_share_if_zero": b1, "current_peer_contributions_known": b2}，'
                    '其中e为整数，b1、b2为JSON布尔值。不要提供解释。' if zh else
                    'Answer these rule-comprehension questions: How many points are you endowed with at the start of the current round? '
                    'If you contribute 0 in this round, do you still receive an equal share from the common account? '
                    'Have the records above already provided B, C, and D\'s contributions for the current round? '
                    'Return only {"endowment": e, "receives_share_if_zero": b1, "current_peer_contributions_known": b2}, '
                    'where e is an integer and b1 and b2 are JSON booleans. Do not include an explanation.')
    elif kind == 'information':
        first = (kwargs.get('history') or initial_history(condition, kwargs.get('minority_actor'), kwargs.get('own_previous', 5)))[0]
        n = CONDITIONS[condition][1]
        count = None if n == 'none' or kwargs.get('anchor') is not None else (3 if n == 'high' else 0)
        if count is not None and kwargs.get('minority_view') is not None:
            count = 2 if n == 'high' else 1
        expected = {'peers_contributed_10': sum(first[p] == 10 for p in PEERS), 'peers_stated_should_10': count}
        question = ('根据上述记录，B、C、D中有几人在第1轮贡献了10积分？有几人在第1轮开始前表示参与者每轮应该贡献10积分？'
                    '只返回 {"peers_contributed_10": a, "peers_stated_should_10": s}。'
                    '各值须为整数；若没有提供应然看法记录，s须为null。不要提供解释。' if zh else
                    'According to the records above, how many of B, C, and D contributed 10 points in round 1? '
                    'How many stated before round 1 that participants should contribute 10 points in each round? '
                    'Return only {"peers_contributed_10": a, "peers_stated_should_10": s}. '
                    'Each value must be an integer; use null for s if no statement record was provided. Do not include an explanation.')
    elif kind == 'past_payoff':
        first = (kwargs.get('history') or initial_history(condition, kwargs.get('minority_actor'), kwargs.get('own_previous', 5)))[0]
        expected = {'payoff': round_points(first)}
        question = ('根据上述记录，你在第1轮获得多少积分？只返回 {"payoff": x}，其中x是数值。不要提供解释。' if zh else
                    'According to the record above, how many points did you receive in round 1? '
                    'Return only {"payoff": x}, where x is a number. Do not include an explanation.')
    else:
        raise ValueError('Unknown check')
    return prefix + '\n\n' + question, expected


def prediction(condition, zh=False, **kwargs):
    ending = ('你预测B、C、D在本轮的平均贡献是多少积分？只返回 {"mean_peer_contribution": x}，'
              '其中x是0至10之间的数值。不要提供解释。' if zh else
              'What do you predict will be the mean contribution of B, C, and D in the current round? '
              'Return only {"mean_peer_contribution": x}, where x is a number from 0 to 10. Do not include an explanation.')
    return context(condition, zh=zh, **kwargs) + '\n\n' + ending


def packaged(kind, condition, **kwargs):
    settings = dict(template='T1', order='history_first', goal='neutral', anchor=None,
                    minority_actor=None, minority_view=None, own_previous=5)
    settings.update(kwargs)
    kwargs = settings
    key = json.dumps([VERSION, kind, condition, settings], sort_keys=True)
    prompt_id = hashlib.sha256(key.encode()).hexdigest()[:20]
    data = {'prompt_id': prompt_id, 'kind': kind, 'condition': condition, 'settings': kwargs,
            'messages': [{'role': 'system', 'content': SYSTEM}],
            'translation_zh': {'system': SYSTEM_ZH}}
    if kind in ('decision', 'prediction'):
        fn = decision if kind == 'decision' else prediction
        en, zh = fn(condition, **kwargs), fn(condition, zh=True, **kwargs)
    else:
        en, expected = check(kind, condition, **kwargs)
        zh, expected_zh = check(kind, condition, zh=True, **kwargs)
        assert expected == expected_zh
        data['expected_not_for_model'] = expected
    data['messages'].append({'role': 'user', 'content': en})
    data['translation_zh']['user'] = zh
    return data


def make_bank():
    main, own_history, goals, anchors, minority, checks, forecasts = [], [], [], [], [], [], []
    for template in RULES:
        for condition, (_, norm) in CONDITIONS.items():
            orders = ('history_first', 'statement_first') if norm != 'none' else ('history_first',)
            for own in OWN_HISTORY_LEVELS:
                for order in orders:
                    settings = {'template': template, 'order': order, 'own_previous': own}
                    if own == 5:
                        main.append(packaged('decision', condition, **settings))
                        forecasts.append(packaged('prediction', condition, **settings))
                    elif template == 'T1':
                        own_history.append(packaged('decision', condition, **settings))
                    for values in ({'A': 0, 'B': 0, 'C': 0, 'D': 0},
                                   {'A': 4, 'B': 10, 'C': 10, 'D': 10},
                                   {'A': 4, 'B': 2, 'C': 5, 'D': 8}):
                        checks.append(packaged('payoff', condition, contributions=values, **settings))
                    for kind in ('structure', 'information', 'past_payoff'):
                        checks.append(packaged(kind, condition, **settings))
                    if template == 'T1' and own == 5:
                        goals.append(packaged('decision', condition, goal='self_max', **settings))
    for condition in ('D_HIGH_N_ABSENT', 'D_LOW_N_ABSENT'):
        for code in (0, 10):
            for own in (5,):
                for order in ('history_first', 'statement_first'):
                    anchors.append(packaged('decision', condition, anchor=code, order=order, own_previous=own))
    for condition in ('D_HIGH_N_HIGH', 'D_HIGH_N_LOW', 'D_LOW_N_HIGH', 'D_LOW_N_LOW'):
        for actor in PEERS:
            for view in PEERS:
                for order in ('history_first', 'statement_first'):
                    minority.append(packaged('decision', condition, minority_actor=actor, minority_view=view, order=order))
    return {'version': VERSION, 'status': 'prepared_not_run', 'total_rounds': TOTAL_ROUNDS,
            'main_own_previous': 5, 'supplement_own_history_levels': [0, 10],
            'notes': 'Only each item.messages is a model input. Identity is unspecified. Translations, condition IDs, settings and expected answers are not sent. This is a prompt bank, not a run schedule.',
            'main_decisions': main, 'own_history_supplement': own_history, 'self_max_supplement': goals,
            'numeric_anchor_diagnostic': anchors, 'two_to_one_supplement': minority,
            'independent_comprehension_bank': checks,
            'optional_independent_predictions': forecasts}


def export():
    root = Path(__file__).resolve().parent.parent / '最小实验方案'
    path = root / '提示词_v7_完整组装.json'
    path.write_text(json.dumps(make_bank(), ensure_ascii=False, indent=2), encoding='utf-8')
    return path


if __name__ == '__main__':
    export()
    print('Exported v7 prompt bank; no model calls.')
