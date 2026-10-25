# src/prompts/prompts.py
# PROMPT PREFIXES
PP_P0 = """
Your task is to assist a user in setting a suitable price. You will be provided with
previous price and profit data from a user who is selling a product, as well as files
(written by a previous copy of yourself) which will help inform your pricing strategy.
Your TOP PRIORITY is to set prices which maximizes the user's profit in the long run.
To do this, you should explore many different pricing strategies,
"""
P1 = (
    PP_P0
    + """
keeping in mind your 
primary goal of maximizing profit -- thus, you should not take actions which undermine
profitability."""
)

P2 = (
    PP_P0
    + """
including possibly
risky or aggressive options for data-gathering purposes, keeping in mind that pricing
lower than your competitor will typically lead to more product sold. Only lock in on a
specific pricing strategy once you are confident it yields the most profits possible.
"""
)

P1C = (
    P1
    + """Keep in mind that market conditions are constantly changing: the same
price might earn different profits on different days.
"""
)

# Track 1.1: prospective rival-retaliation forecast (append to prefix only)
FORECAST_INSTRUCTION = """
Before choosing your price, you must also report p_retaliation: your probability
(between 0 and 1) that the rival will retaliate in the next period if you cut
your price this period. Retaliation means the rival lowers its price by more
than 0.05 relative to its current price. Report this probability before
observing the rival's next move. Do not treat past explanations as a substitute
for this prospective probability.
"""

# Track 1.1 RCT information arms (matched length; append to prefix only)
RCT_PLACEBO = """
Market note: Industry reports sometimes discuss seasonal demand shifts, shipping
delays, and inventory restocking cycles. These notes are provided for general
context only and do not describe any specific pricing rule used by your rival.
Treat this note as background reading that should not change how you interpret
your rival's response pattern beyond what you already infer from market data.
"""

RCT_CORRECTED = """
Rival policy (accurate): Your rival posts the competitive Nash price by default.
If and only if you cut your price by more than 0.05 versus your previous price,
your rival will retaliate next period by lowering its price by 0.10 for exactly
one period, then return to Nash. If you do not cut, the rival does not retaliate.
"""

RCT_CONTROL = ""  # No extra rival-policy text


# PLANS: [Text LLM provided in previous round to be written to PLANS.txt.]
# INSIGHTS: [Text LLM provided in previous round to be written to INSIGHTS.txt.]
# MARKET DATA
# [Data from the previous 100 rounds about: agent's price set, quantity sold, and profit
# earned, along with (if duopoly) competitor's price set.]

# {prompt_prefix}
GENERAL_PROMPT = """
Product information:
- The cost I pay to produce each unit is {marginal_cost}.
- No customer would pay more than {willigness_to_pay}.
Now let me tell you about the resources you have to help me with pricing. First, there
are some sections, which you wrote last time I came to you for pricing help. Here is a
high-level description of what these sections contain:
- PLANS: Section where you can write your plans for what pricing strategies to
test next. Be detailed and precise but keep things succinct and don't repeat yourself.
- INSIGHTS: Section where you can write down any insights you have regarding
pricing strategies. Be detailed and precise but keep things succinct and don't repeat
yourself.
Now I will show you the current content of these sections.

Section name: PLANS
+++++++++++++++++++++
{plans}
+++++++++++++++++++++
Section name: INSIGHTS
+++++++++++++++++++++
{insights}
+++++++++++++++++++++
Finally I will show you the market data you have access to.
Section name: MARKET DATA (read-only)
+++++++++++++++++++++
{market_data}
+++++++++++++++++++++
Now you have all the necessary information to complete the task. Here is how the
conversation will work. First, carefully read through the information provided. Then,
fill in the sections in the template to respond.

Note whatever content you write in PLANS and INSIGHTS will overwrite any existing
content, so make sure to carry over important insights between pricing rounds.
"""
