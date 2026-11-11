"""Candidate comprehension revision; diagnostic only, not a new formal design."""
import prompts_v7 as p

RULES = '''Your group has four participants: A (you), B, C, and D. The same group plays 10 rounds.

At the start of every round, each participant receives 10 new points. Each chooses an integer from 0 to 10 to put into a common account and keeps the rest. The four contributions are added together, doubled, and split equally among all four participants, including anyone who contributed 0.

Your round payoff is the points you kept plus your share from the common account:
Your round payoff = (10 - your contribution) + (your contribution + B's contribution + C's contribution + D's contribution) / 2.

Your total payoff is the sum of your round payoffs. Earlier payoffs do not change the 10 new points received each round.

Everyone chooses simultaneously. You do not know B, C, or D's current-round contributions before choosing. After each round, you see all four contributions and your round payoff.'''

RULES_ZH = '''小组由四名参与者组成：A（你）、B、C和D。同一个小组进行10轮游戏。

每轮开始时，每人获得新的10积分。每人选择向公共账户投入0至10之间的整数积分，并保留其余积分。四人的贡献相加、翻倍，然后由四人均分，包括贡献0的参与者。

你的当轮所得等于你保留的积分加上你从公共账户分得的积分：
你的当轮所得 =（10 − 你的贡献）+（你的贡献 + B的贡献 + C的贡献 + D的贡献）/ 2。

你的总所得等于各轮所得相加。过去的收益不会改变每轮重新获得的10积分。

所有人同时选择。作出选择前，你不知道B、C、D本轮的贡献。每轮结束后，你会看到四人的贡献和你当轮所得。'''

VECTORS = [dict(A=4,B=10,C=10,D=10),dict(A=4,B=2,C=5,D=8),
           dict(A=2,B=4,C=6,D=8),dict(A=8,B=2,C=4,D=6)]


def context(condition, clear=True, zh=False):
    text=p.context(condition,zh=zh)
    if clear:
        text=text.replace(p.RULES_ZH['T1'] if zh else p.RULES['T1'],RULES_ZH if zh else RULES,1)
    return text


def payoff_question(values, direct=False, zh=False):
    if not direct:
        text,_=p.check('payoff','D_LOW_N_LOW',contributions=values,zh=zh)
        return text.removeprefix(p.context('D_LOW_N_LOW',zh=zh)+'\n\n')
    table='\n'.join(f'{k}: {values[k]}' for k in p.PLAYERS)
    if zh:
        return ('下面是一道独立的收益计算题，不是让你选择贡献。\n'
                '对于本题，第2轮的假设贡献如下（单位：积分）：\n'+table+
                '\n每人这轮开始时获得10积分。按照游戏规则，A在第2轮获得多少积分？'
                '只返回 {"payoff": x}，其中x为数值。不要提供解释。')
    return ('This is a payoff calculation question, not a request to choose a contribution.\n'
            'For this question, the hypothetical contributions in round 2 are (in points):\n'+table+
            '\nEach participant starts this round with 10 points. Under the game rules, what is A\'s payoff for round 2? '
            'Return only {"payoff": x}, where x is a number. Do not include an explanation.')


def tasks():
    rows=[]
    def add(key,category,condition,user,expected,**details):
        rows.append(dict(task_id=key,category=category,condition=condition,**details,
                         messages=[dict(role='system',content=p.SYSTEM),dict(role='user',content=user)],
                         expected_not_for_model=expected))
    # Full-history low-behavior/low-opinion background held fixed in the 2x2.
    for clear in (False,True):
        for direct in (False,True):
            category=f'rules_{"clear" if clear else "old"}__question_{"direct" if direct else "old"}'
            for index,vector in enumerate(VECTORS):
                add(f'{category}/{index}',category,'D_LOW_N_LOW',
                    context('D_LOW_N_LOW',clear)+'\n\n'+payoff_question(vector,direct),
                    {'payoff':p.round_points(vector)},vector=vector,held_out=index>=2)
    for condition in p.CONDITIONS:
        question=('Do the records above include a statement by B about how much participants should contribute? '
                  'Return only {"record_present": true} or {"record_present": false}. Do not include an explanation.')
        add('statement_present/'+condition,'statement_present',condition,
            context(condition)+'\n\n'+question,{'record_present':p.CONDITIONS[condition][1]!='none'})
    for condition in ('D_LOW_N_LOW','D_LOW_N_HIGH'):
        question=('According to B\'s recorded statement, how many points should each participant contribute per round? '
                  'Return only {"amount": n}, where n is an integer. Do not include an explanation.')
        add('stated_amount/'+condition,'stated_amount',condition,context(condition)+'\n\n'+question,
            {'amount':0 if p.CONDITIONS[condition][1]=='low' else 10})
    for key,question,expected in [
        ('endowment','How many new points do you receive at the start of round 2? Return only {"points": n}, where n is an integer.',{'points':10}),
        ('share','If you contribute 0 in a round, do you still receive an equal share of the common account? Return only {"receives_share": true} or {"receives_share": false}.',{'receives_share':True}),
        ('current_known','Have the records above provided B, C, and D\'s actual contributions for round 2? Return only {"known": true} or {"known": false}.',{'known':False})]:
        add(key,'basic_rule','D_LOW_N_LOW',context('D_LOW_N_LOW')+'\n\n'+question,expected)
    return rows
