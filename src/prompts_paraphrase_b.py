"""Frozen meaning-preserving wording B; numerical rules and history unchanged."""
import run_from_round1 as r
import prompts_v11_public_observation as v
BASE_MESSAGES=r.messages
RULES="""The four participants in your group are A (you), B, C, and D. This group remains unchanged throughout the 10 rounds.

Each round begins with a fresh allocation of 10 points to every participant. Each participant chooses a whole-number contribution between 0 and 10 points for the common account and retains any points not contributed. The contributions of all four participants are combined. This total is multiplied by two and then divided equally among all four participants, including those who contributed 0.

Your payoff for a round consists of the points you retain and your equal share of the common account:
Your round payoff = (10 - your contribution) + (your contribution + B's contribution + C's contribution + D's contribution) / 2.

Your overall payoff is obtained by adding your payoffs across rounds. The fresh 10-point allocation at the start of a round is unaffected by payoffs from earlier rounds.

Contribution decisions are made simultaneously. Before deciding, no participant knows how much any of the other participants will contribute in the current round. At the end of each round, all four participants see the contribution made by each participant, and each participant learns their own payoff for that round."""
RULES_ZH="""你的组内有四名参与者：A（你）、B、C和D。10轮游戏期间，小组成员保持不变。

每轮开始时，每人重新获配10积分。每人选择向公共账户贡献0至10之间的整数积分，并保留未贡献部分。四人的贡献合并，总额乘以2，然后由四人均分，包括贡献0的人。

你每轮的收益由保留积分和从公共账户均分的积分组成：
你的当轮所得 =（10 − 你的贡献）+（你的贡献 + B的贡献 + C的贡献 + D的贡献）/ 2。

你的总收益为各轮收益相加。以前轮次的收益不影响每轮开始时新分配的10积分。

各人同时作出贡献决定。决定前，任何人均不知道其他参与者本轮将贡献多少。每轮结束，四人都看到每个人的贡献，并各自得知自己的当轮收益。"""

def rewrite(text):
    text=text.replace(v.RULES,RULES)
    for t in range(1,11):
        for n in (0,10):
            text=text.replace(f'In round {t}, each participant should contribute {n} points.',f'Each participant should put {n} points into the common account in round {t}.')
    return text

def messages(condition,history,rules_text=None):
    original=BASE_MESSAGES(condition,history,v.RULES)
    return [dict(original[0]),dict(role='user',content=rewrite(original[1]['content']))]
