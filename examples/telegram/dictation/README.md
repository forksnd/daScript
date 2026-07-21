# dictation / Cadmus - local Telegram dictation and conversation bot

Cadmus transcribes Telegram voice notes through a local `dasllama-server`, cleans the transcript,
and answers questions using the current chat's locally stored history. The Telegram bot contains no
models; the server owns ASR and LLM inference.

Cadmus requests automatic oldest-turn truncation from the chat endpoint, which preserves the system
prompt, database tools, and newest turns while reserving the configured output budget. It refuses
to post a response returned with `finish_reason: "length"`, so a cut-off answer is never presented
as complete.

For each voice message the bot downloads the audio, converts it to 16 kHz mono WAV with ffmpeg,
calls `/v1/audio/transcriptions`, then applies the editable `dictation_prompt` through
`/v1/chat/completions`. The cleaned transcript is recorded as the user's message. The bot's
transcript-delivery reply is deliberately not recorded as an assistant turn.
Before acknowledging a live Telegram audio update, the bot serializes it into a durable SQLite
queue. It then processes that queue one message at a time, with capped exponential retry backoff;
an interrupted job returns to pending when the bot restarts. This queue is independent of the
Telegram Desktop export backlog and the server's transient ASR request queue.

Every inbound text, caption, voice, and audio message visible to the bot is retained in local SQLite.
Real Cadmus answers are retained in the same history. Conversational answers are disabled by default;
set `reply_when_mentioned = true` to answer a leading `@bot` username or wake name, replies to real
answers, and private-chat messages. A reference to Cadmus later in a message is ignored. `/help` and
`/summary` remain available either way. Replies to bot-authored transcript deliveries are treated as
attribution to the speaker and never trigger Cadmus.

Cadmus has four current-chat-only history tools: full-text search, first/last/count timelines,
message context, and chronological browsing. Results include Telegram message IDs and post links.
The runtime supplies the current chat ID, so the model cannot request another channel's history.
Each addressed question uses two stages: a small retrieval-controller prompt returns one validated
JSON action, the bot executes that action, and a tool-disabled completion writes the final answer
from the returned evidence. This avoids relying on optional model-native tool-call formats and makes
failure to consult required chat history deterministic.
Grounded answers retain their source message IDs. A follow-up such as "what were we discussing
there?" automatically loads five messages before and after those sources before generation.
When `BRAVE_SEARCH_API_KEY` is set, Cadmus also has a compact Brave web-search tool for current or
external facts, including details needed to supplement freshly retrieved chat context. Web excerpts
are marked as untrusted, and source links are appended to the answer.
Permanent audio-format failures and summary/conversation failures are reported in the originating
Telegram chat with a `⚠` marker. Transient live-audio download or inference failures stay in the
durable queue, are logged under `dictation.queue`, and retry without spamming the chat.

## Requirements

- A running `dasllama-server` with both an instruct model and an ASR model.
- ffmpeg on `PATH`, or an absolute `ffmpeg` path in `dictation.toml`.
- A Telegram token from `@BotFather`.
- Optional: a Brave Search API key in `BRAVE_SEARCH_API_KEY` for public web search.

## Setup and run from source

```text
bin/daslang utils/daspkg/main.das -- install --root examples/telegram/dictation
bin/daslang -project_root examples/telegram/dictation examples/telegram/dictation/main.das
```

The bot itself does not require JIT. Edit `dictation.toml` first. `TELEGRAM_BOT_TOKEN` overrides
`bot_token`; `BRAVE_SEARCH_API_KEY` enables Brave Search and has no config-file fallback. Relative
database and log paths resolve beside the executable in a release and beside `main.das` during
development.

The database is created automatically. Migration 1 creates message/state storage, migration 2 adds
the FTS5 index, migration 3 records outgoing transcript-delivery IDs, migration 4 adds resumable
Telegram-export media checkpoints, and migration 5 records the database sources behind Cadmus
answers. Migration 6 adds the durable live-audio queue, and migration 7 checkpoints successful live
ASR results so cleanup retries do not retranscribe the same audio. Migration 8 rebuilds the FTS5
index with typed, unindexed chat/message keys so only message text is tokenized. Telegram update
offsets are persisted, and `(chat_id, message_id)` makes replayed updates and edits idempotent.
History is retained indefinitely.

Before sending an answer, Cadmus validates every model-written HTTP(S) link against exact URLs
returned by Brave Search or chat-history tools in the current turn. An unverified link triggers up to
two corrected completion attempts; if it remains ungrounded, Cadmus sends an error instead of the
answer. Successful Brave calls are recorded under the `cadmus.web` log category.

## Conversation and summary commands

```text
/help
/summary
/summary me
/summary yesterday
/summary 2026-07-13
/summary @username 2026-07-13
```

Bare `/summary` (and `today`) means the rolling previous 24 hours. `yesterday` and an explicit
`YYYY-MM-DD` use local calendar days. `me` and `@username` filter by Telegram user ID after resolving
the participant inside this chat. Forum topics currently share their parent chat's history.

Anyone who can address the bot in a chat can request a chat-wide or per-user summary. Long histories
are summarized in chunks and reduced into one answer.

## Telegram group setup

Commands work without registering them, but the Telegram command menu is configured manually:

1. Open `@BotFather`, run `/setcommands`, and select this bot.
2. Paste:

   ```text
   summary - Summarize recent conversation
   help - Show Cadmus commands and usage
   ```

3. Run `/setprivacy`, select the bot, and choose `Disable` so it can retain ordinary group messages.
4. Remove and re-add the bot to each group, or make it an administrator, for the privacy change to
   take effect.

## Important configuration

| Key | Meaning |
|---|---|
| `server` | dasllama-server base URL |
| `bot_token` | BotFather token; the environment variable wins |
| `dictation_prompt` | Transcript-repair prompt; `{language}` is replaced per message |
| `assistant_name`, `assistant_aliases` | Display identity and comma-separated group wake names |
| `reply_when_mentioned` | Reply when directly addressed at the start, replied to, or in private; default `false` |
| `history_chat_username` | Optional public chat username for post links; private chats use `t.me/c` links |
| `assistant_prompt` | Editable conversational personality, separate from trusted runtime rules |
| `database` | Local SQLite history path |
| `recent_messages`, `history_search_results` | Recent context and per-tool FTS result limits |
| `history_search_languages` | Comma-separated languages for independent strict FTS query variants; this does not translate or embed stored messages |
| `web_search_enabled`, `web_search_results`, `web_search_timeout`, `web_search_max_calls` | Brave Search controls; key comes from the environment |
| `summary_chunk_messages`, `summary_max_messages` | Summary reduction limits |
| `max_tokens`, `temp` | Dictation cleanup generation settings |
| `assistant_max_tokens`, `assistant_temp` | Conversational generation settings |
| `language`, `min_chars`, `llm_timeout` | ASR and request controls |
| `ffmpeg`, `poll_timeout`, `log_file`, `log_level` | Process and logging controls |
| `keep_audio_dir` | Optional archive for original audio; empty keeps none |

The legacy `prompt` key is still accepted for deployed configs, but new configs should use
`dictation_prompt`. Keeping it separate from `assistant_prompt` prevents conversational personality
from changing dictation.

## Live control and prompt testing

The bundled watchdog serves a localhost-only Cadmus control page at
`http://127.0.0.1:8091/`. It shows bot/server state and recent structured activity, and provides:

- live switches for transcription, cleanup, assistant replies, history retrieval, summaries,
  web search, and background history import;
- per-stage temperature, top-k, top-p, min-p, presence/frequency/repeat penalties, token budget,
  and thinking controls;
- live prompt overrides for cleanup, retrieval planning, assistant answers, and summaries;
- a direct model playground which never writes chat history or sends a Telegram message; and
- a model picker backed by dasllama-server's existing config/restart API.

Live values are stored in `cadmus-runtime.json` next to the executable and are reloaded before new
work without restarting Cadmus. The release does not own this file, so it survives upgrades. An
empty prompt override selects that stage's built-in prompt, which the page shows as placeholder
text. Thinking defaults to off for every bot stage; enable it deliberately per stage or in the
playground.

Cadmus owns its own defaults: at startup it writes `cadmus-defaults.json` (feature flags, per-stage
generation settings, and built-in prompt text) beside the executable, and the watchdog and control
page read it from there rather than restating those values. Both fall back to their own defaults
while the file is absent, so a first run before the bot has ever started still works.

The control API is same-origin only. It requires a loopback `Host`, rejects a foreign `Origin`, and
requires `Content-Type: application/json` on writes, so a page in another tab cannot drive the bot
through the control port; violations return 403.

## Importing older Telegram history

The HTTP Bot API cannot fetch arbitrary pre-existing chat or channel history. `getUpdates` exposes
only the pending update queue, and Telegram may remove older bot messages after processing. The
MTProto `messages.getHistory` method does provide history, but Telegram marks it as user-only, so the
bot token cannot use it.

A practical backfill therefore uses Telegram Desktop's HTML export. The importer is incremental:
completed `messages*.html` pages are safe to re-run, the newest page is skipped while the export is
growing, and message upserts are idempotent. HTML exports expose display names but not stable Telegram
sender IDs, so imported senders receive deterministic negative IDs based on their display names.

Dry-run the current stable pages:

```text
bin/daslang examples/telegram/dictation/import_history.das -- \
  --root <telegram-export> --database cadmus.db --chat-id <telegram-chat-id>
```

Catalog them, then raw-transcribe queued audio without LLM cleanup:

```text
bin/daslang examples/telegram/dictation/import_history.das -- \
  --root <telegram-export> --database cadmus.db --chat-id <telegram-chat-id> \
  --apply --transcribe --server http://127.0.0.1:8080
```

Raw ASR runs two concurrent requests by default, smallest files first. It stores the result in both `Content` and
`RawTranscript`, updates FTS immediately, and records `pending`, `done`, or `failed` durably. Use
`--workers N` to match the server's ASR worker count, `--retry-failed` to retry failures, and
`--max-audio N` for a bounded batch. Re-run the same command as
Telegram Desktop produces more pages; use `--include-last-page` only after the export has finished.

## Release

```text
bin/daslang utils/daspkg/main.das -- release --root examples/telegram/dictation --out <staging>
```

The release contains a standalone executable, the shared watchdog (`watchdog.py` from
`utils/watchdog/`, plus `watchdog_control.py` and `watchdog.json`), `control.html`, runtime DLLs,
required shared modules, and initializes the config template only when `dictation.toml` is absent.
Re-releasing preserves the deployed `dictation.toml`; `cadmus.db` and logs are not release-owned and
are preserved as well.

The watchdog is the shared one — it finds the bundle's single executable on its own, and
`watchdog.json` supplies only what discovery cannot know: the name logs are keyed on (`cadmus`),
the graceful-stop file, and that this program has no HTTP health endpoint. The control page lives in
`watchdog_control.py`, a plugin the watchdog loads if present; it owns `dictation.toml`, the prompt
set, generation defaults and the activity feed. See `utils/watchdog/README.md`.

It shows a Windows notification after a crash, saves a crash bundle, and restarts with bounded
backoff. Start it from the released directory with no arguments:

```powershell
Set-Location E:/dictation-bot
python watchdog.py
```

Then open `http://127.0.0.1:8091/` for the debugging console.

Pressing Ctrl+C or creating `logs/cadmus.stop` requests a lifecycle-safe shutdown. Cadmus finishes
its current poll or request, flushes its logger, and exits normally; the watchdog treats exit code 0
as intentional and does not restart it. On Windows the watchdog uses the effective WER LocalDumps
policy for `dictation-bot.exe` when one is installed and logs `wer_not_ready` when it is absent.
