# Reflection — Day 17 (≤ 200 words)

1. **The flywheel.** Day 13 emitted agent traces; today you turned them into an
   eval set and DPO pairs that Day 22 will train on. Which step in
   `traces → Bronze → datasets` would break most silently in production if you
   got it wrong — and how would you detect it?

2. **Decontamination.** Your run dropped 2 of 3 preference pairs because their
   prompts were in the eval set. What concretely goes wrong if you *skip* this
   step and train on those pairs? How would the lie show up in your metrics?

3. **Point-in-time.** The naive join leaked a future `lifetime_spend` into the
   training row. Describe one feature in a system you know that would be
   dangerous to join without an `ASOF`/point-in-time guard.

4. **Graph vs vector.** From `kg_demo.py`, name one question the knowledge graph
   answers well that flat chunk retrieval (`embed.py`) would struggle with, and
   one where the graph is overkill.

---

1. **Decontamination** breaks most silently: the pipeline still runs and outputs look fine, but train prompts overlap eval. Detect by diffing prompts in `eval_golden.jsonl` vs `preference_pairs.jsonl`, or by eval scores jumping after fine-tune without gains on fresh holdout prompts.

2. Skipping it lets the model memorize eval answers (e.g. widget/gadget questions). Offline eval accuracy rises, but the agent fails on new phrasings—the benchmark improvement is fake leakage, not real capability.

3. **Lifetime spend at transaction time** for fraud/churn: joining the latest total instead of an ASOF value leaks future purchases into training; offline AUC looks great, production does not.

4. KG wins on multi-hop *“Where does a widget ship from?”* (widget → accessory → Hanoi). Vector is enough for *“How long can I return a widget?”*—one chunk already holds the answer.
