# Smarter offline assistant

The online Claude option has been removed. Instead, the existing offline
assistant (the ⌘ button, no internet or API key needed) can now answer a new
kind of question: "what should I do next?"

Ask it any of:
- "what should i do next?"
- "any suggestions?"
- "how do i improve this dataset?"
- "what's wrong with my data?"
- "help me clean this"
- "where do i start?"

It looks at the current dataset and returns a prioritized, numbered checklist
covering: duplicate rows, heavy/moderate missing data, constant columns,
ID-like columns worth dropping before training, outliers, skewed numeric
columns, near-perfectly correlated column pairs, and a class-balance note if
it can guess a target column. Each item that maps to a real cleaning command
includes it in backticks, e.g. `drop column customer_id` -- type or paste
that into the same box to actually run it (through the same command engine
the offline assistant already used).

All of this is deterministic pandas/statistics under the hood, not a model:
no internet, no API key, no per-message cost, same as everything else in the
offline assistant.

## Files touched
- `backend/assistant.py` -- adds `_next_steps_report` and two small helpers
  (`_looks_like_id_column`, `_looks_like_target_column`), a new dispatch
  branch for the trigger phrases above, a wider typo-correction keyword set,
  and updated hint text in two existing fallback messages.
- `frontend/index.html` -- one CSS line (`white-space: pre-wrap` on
  `.fc-msg`) so the numbered list actually shows line breaks instead of
  running together. No markup or JS added.
- Everything from the Claude feature (`backend/claude_assistant.py`, the
  ✦ button/panel, the `run.py` changes) has been removed.
