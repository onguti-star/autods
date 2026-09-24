# Claude assistant (optional, online)

A second, additional assistant that sits next to AutoDS's existing offline
one. It coexists with it rather than replacing it:

|                     | Offline assistant (⌘, existing) | Claude assistant (✦, new) |
|---------------------|----------------------------------|------------------------------|
| Needs internet      | No                               | Yes                          |
| Needs an API key    | No                               | Yes (your own)                |
| Sees                | The data directly (runs rule-based commands on it) | Only a text profile: shape, dtypes, missing %, stats, top categories, correlations -- never raw rows |
| Can edit the data   | Directly                         | Only by suggesting a command, which you click to run through the SAME offline command engine |

Nothing about the offline assistant changed. If you never add a Claude API
key, the ✦ button just shows a settings prompt and nothing is ever sent
anywhere.

## Setup

1. Get a key at https://console.anthropic.com/settings/keys (this is a
   personal Anthropic account, separate from AutoDS -- normal API pricing
   applies per message).
2. Click the ✦ button (bottom-right, next to the other floating buttons),
   then ⚙ settings, paste the key, Save. It's saved to a file in the app's
   data directory (`claude_api_key.txt`) and is never sent back to the
   browser except as `sk-...xxxx`.
3. Load a dataset. A "Send summary to Claude" card appears with the profile
   already built -- click it to start the conversation, or just type a
   question.

## Suggested actions

When Claude proposes something concrete, it shows up as a clickable chip
under its reply:
- **▶ Run** -- a cleaning instruction, sent to the exact same `/api/ask`
  command engine the offline assistant uses, after you confirm. Claude never
  touches the dataframe itself.
- **→ Open** -- scrolls to a tab.
- **📊 Chart** -- scrolls to Visualise and pre-fills the chart-type and
  column pickers (you still click Plot).

## Changing the model

Default is `claude-sonnet-5`. Override with an environment variable before
starting AutoDS, e.g. `AUTODS_CLAUDE_MODEL=claude-opus-5-5`.

## Files this feature adds or touches

- `backend/claude_assistant.py` -- new. All the `/api/claude/*` routes.
- `frontend/index.html` -- adds the ✦ button/panel; nothing existing removed.
- `run.py`, `desktop.py` -- both now also mount `claude_assistant`'s routes
  (same pattern already used for the desktop app's own routes). `run.py
  --dev` (hot-reload) does not load it; run without `--dev` to use it.
- `main.py` -- untouched.

## Limitations of this first version

- Chat history resets on page reload (kept in the browser tab's memory only,
  not saved to disk).
- One conversation per dataset tab, not per chart or per panel.
- No streaming -- the reply appears all at once after a few seconds.
