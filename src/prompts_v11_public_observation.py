"""Versioned public-observation clarification. Earlier rules retained unchanged."""
import check_revision_r1 as old

OLD_FINAL="Everyone chooses simultaneously. You do not know B, C, or D's current-round contributions before choosing. After each round, you see all four contributions and your round payoff."
NEW_FINAL="Everyone chooses simultaneously. Before choosing, no participant knows the other participants’ current-round contributions. After each round, all four participants observe everyone’s contributions, and each participant learns their own round payoff."
NEW_FINAL_ZH='所有人同时选择。作出选择前，任何参与者都不知道其他参与者本轮的贡献。每轮结束后，四名参与者都能看到所有人的贡献，并各自得知自己的当轮收益。'
assert old.RULES.count(OLD_FINAL)==1
RULES=old.RULES.replace(OLD_FINAL,NEW_FINAL)
RULES_ZH=old.RULES_ZH.rsplit('\n\n',1)[0]+'\n\n'+NEW_FINAL_ZH
