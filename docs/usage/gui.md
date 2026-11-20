# Web GUI

`mini-gui` starts a small browser-based interface for mini-SWE-agent. It uses
Server-Sent Events (SSE) to stream the model response token by token. Assistant
messages read as a plain transcript instead of a stack of step cards, and each
bash command is a quiet one-line row showing only the command and its exit
status. A run has explicit **Run started** and **Run finished** markers.

```bash
mini-gui
```

The command prints a URL such as `http://127.0.0.1:8000` and opens it in your
default browser. The text box starts as a compact single-line composer. Enter a
task and press **Send**; while a run is active, the button changes to **Stop**.
After a run finishes or is stopped, the button switches back to **Send** and you
can keep typing follow-up messages in the same conversation. Follow-up turns are
appended to the existing transcript: the model receives all earlier user
instructions, tool calls, and observations as context instead of starting a new
conversation.

## Image input

Use the **+** button to the left of the task input to add one or more
screenshots or diagrams, or paste/drop an image directly into the task box. Attachments are
shown as thumbnails before the run starts and can be removed individually.
You can also send an image without any text; the agent will be asked to analyze
the attached image. The images are included in the first user message, stored
with the conversation, and shown when a conversation is reloaded.

Image input requires a multimodal model. Up to 8 images can be attached per
conversation, with a 20 MB limit per image.

## Conversations

Every run started through the GUI appears in the **Conversations** sidebar.
Click a conversation to load its task, messages, commands, status, and cost
again. Conversations that are still running keep streaming while selected.
Use **New** to clear the view and start another task.

Simple greetings and small-talk messages (for example, "hi", "hello", or
"thanks") are answered immediately instead of being sent through the bash
tool loop. Messages that contain an actual request, such as
"hi, please fix the failing tests", still start a normal agent run. The same
guard is used by the Python agent classes, so short conversations cannot
accidentally trigger repository searches.

Each conversation shows the first eight characters of its conversation ID and
its workspace directory next to its name in the sidebar. The workspace is taken
from the environment's ``cwd`` setting; if that is unset, the GUI process's
current working directory is used.

The selected conversation is remembered in the URL and browser session storage.
If you refresh the page while an agent is running, the GUI reconnects to that
session and replays its events, so the run remains visible and can still be
stopped. The agent itself keeps running on the server while the page is closed.

Conversations are persisted in SQLite as they run. The default database is
`gui_conversations.db` in mini-SWE-agent's global configuration directory. On
startup, the GUI restores the conversation list, summaries, and streamed events
from this database, so conversations remain available after restarting the
server. Each conversation update is written immediately, and runs interrupted by
a server restart are marked as `Interrupted`.

The exact database path is printed when `mini-gui` starts, shown in the GUI
header, and returned by `GET /api/config` and `GET /api/database`. The location
depends on your operating system:

| Platform | Default path |
| --- | --- |
| macOS | `~/Library/Application Support/mini-swe-agent/gui_conversations.db` |
| Linux | `${XDG_CONFIG_HOME:-~/.config}/mini-swe-agent/gui_conversations.db` |
| Windows | under `%LOCALAPPDATA%` (the exact path is printed at startup) |

Set the `MSWEA_GLOBAL_CONFIG_DIR` environment variable to move the whole
configuration directory, or use `--database PATH` / `MSWEA_GUI_DB_PATH` to
choose only the database file.

A conversation can be downloaded as a self-contained JSON trajectory,
including its event stream, token/cost summary, and the latest agent trajectory
when available, from `/api/conversations/<session-id>/export`.


## Settings and model switching

Use the **Settings** button in the top-right corner to open the settings
dialog. The dialog is split into a few modules: **General** shows the active
model, workspace, and conversation database; **Models** manages model
providers and models; **Conversation** explains the conversation controls; and
**About** contains version information.

In the **Models** module you can add, edit, disable, and delete providers, and
manage each provider's models. Click a provider row to expand its model list.
Each model has an enabled toggle, and each provider's expanded section has its
own **Add model** action. Only enabled models from enabled providers appear in
the model switcher at the top of the conversation. Provider definitions support
an API-key environment variable, a custom base URL, and an optional locally
stored API key.

After saving a provider, click **Load models** in its settings row (or
**Fetch & load models** in the provider form). The GUI calls the provider's
model-list endpoint, adds every model that is not already configured under that
provider, marks the new models as enabled, and refreshes the conversation model
selector. Loaded model IDs are qualified with the provider prefix LiteLLM
requires (for example, `gpt-4o-mini` loaded from OpenAI becomes
`openai/gpt-4o-mini`). Models already present are left unchanged, so loading
the same provider again is safe. Deleting a provider also deletes all models
that belong to it.

### Where to enter an API key

The provider form has two API-key fields with different purposes:

- **API key (optional, stored locally)** — paste the actual secret API key
  here. The GUI stores it in `gui_settings.json` and passes it to the model
  when a run starts. This is the field to use if you want to configure the key
  entirely from the GUI.
- **API key environment variable (variable name)** — enter only the *name* of
  a shell environment variable, such as `OPENAI_API_KEY`. This field must not
  contain the secret key itself. Use it when the key is already exported in
  the environment that starts `mini-gui`; in that case the stored API-key field
  can be left blank.

If both are configured, the locally stored API key is passed to the model and
is also exposed through the named environment variable for model backends that
resolve credentials from the environment. Typical configurations therefore
look like either:

```text
API key environment variable: OPENAI_API_KEY
API key (optional, stored locally): sk-...
```

or, when using only the shell environment:

```bash
export OPENAI_API_KEY=sk-...
mini-gui
# Leave the stored API key field blank.
```

The conversation toolbar shows a **Model** selector in place of the old status
pill. Each option names the model's provider on the right, and the provider is
also shown as a badge beside the current selection. Choosing a different model
updates the active model and applies it to the next run in the current
conversation. Run status remains available in the Run panel and in each
conversation's sidebar row. Settings are stored as
`gui_settings.json` next to the conversation database; override the location
with `MSWEA_GUI_SETTINGS_PATH`.

The GUI API also exposes provider/model management for local tooling:

| Endpoint | Description |
| --- | --- |
| `GET /api/settings` | Providers, models, enabled models, and the active model |
| `POST /api/providers` | Create a provider |
| `PATCH /api/providers/<id>` | Update or enable/disable a provider |
| `DELETE /api/providers/<id>` | Delete a provider and its models |
| `GET /api/providers/<id>/models` | Query the provider and load any new models |
| `POST /api/providers/<id>/load-models` | Query the provider and load any new models |
| `POST /api/models` | Create a model |
| `PATCH /api/models/<model-name>` | Update or enable/disable a model |
| `DELETE /api/models/<model-name>` | Delete a model |
| `POST /api/settings/model` | Switch the active model |

## Conversation states

Every conversation has one of five canonical states:

| State | Meaning |
| --- | --- |
| `running` | The agent is still working. |
| `submitted` | The agent finished by submitting a result (`Submitted`). |
| `stopped` | The run was stopped by the user. |
| `failed` | The run ended with an error or a limit was exceeded. |
| `interrupted` | The server stopped before the run finished. |

The raw agent `exit_status` remains available as `status` and `exit_status` in
the conversation metadata. The normalized state is available as `state`.
`GET /api/conversations` also returns `states`, `state_count`, and
`state_counts`, and `GET /api/conversations/statuses` returns the same state
list and counts on their own.

## Options

The GUI accepts the same main model/configuration options as `mini`:

```bash
mini-gui --model openai/gpt-5.4-mini
mini-gui -c mini.yaml -m anthropic/claude-sonnet-4-5-20250929
mini-gui --host 0.0.0.0 --port 8080 --no-browser
```

Useful flags:

| Flag | Description |
| --- | --- |
| `-m`, `--model` | Model name |
| `-c`, `--config` | Config file(s) or `key=value` overrides |
| `-t`, `--task` | Prefill the task text box |
| `-o`, `--output` | Trajectory output path |
| `--database`, `--db` | SQLite conversation database path |
| `--host` / `--port` | Bind address (the default is local-only `127.0.0.1:8000`) |
| `--browser` / `--no-browser` | Open or do not open a browser |

The GUI runs the agent non-interactively (equivalent to yolo mode), so it does
not ask for command confirmation in the browser. Only expose it on an untrusted
network if you understand that anyone who can reach it can ask the agent to run
commands in the configured environment.

## Activity timeline

The right-hand **Activity** panel is a compact, turn-aware ledger of the run,
modelled after the trajectory view in DeepSeek Harness. Instead of an
undifferentiated log, it shows the events in the order they happened:

- **Run started** marks the task and starts the elapsed timer.
- **Step N** rows group one model turn. A numbered marker identifies the step,
  and the row is updated as the model thinks, writes a response, or requests a
  command.
- **Bash command** rows appear under the step that requested them and show the
  command and exit status. The full command text, duration, and captured output
  are shown when the Activity view is expanded.
- **Run finished** reports the final exit status, submission preview, cost, and
  token totals.

The header summarises the number of model steps, tool runs, errors, and elapsed
time. The colour strip is a compact overview of the same timeline: blue segments
are model turns, amber segments are tool runs, green is a successful finish, and
red marks an error. The compact panel never scrolls; it keeps the most recent
activity rows that fit in the available space and drops older rows from view.
Use **Expand** in the Activity header to turn the right rail into a full-width
timeline over the conversation area. In that view every activity is shown as a
readable row with its command and result details, and the timeline scrolls
vertically. It follows the newest activity until you scroll up to inspect older
entries. Press **Collapse** or Escape to return to the conversation. On narrower
windows where the right rail is hidden, use **Activity** in the conversation
toolbar to open the timeline. Every conversation also retains this timeline when
selected again.

## Streaming

Streaming is implemented by `LitellmModel.stream_query()`, which requests a
stream from the provider, emits `assistant_delta` events through an optional
agent callback, and then feeds the assembled response through the normal action
parsing and execution path. Other model implementations that do not support
streaming still work: the GUI displays their final assistant message when the
query returns. After every model call, the run panel shows cumulative input and
output token usage, and each assistant message reports the tokens consumed by
that step.

The transcript keeps tool calls deliberately lightweight: a command is shown as
one `$ command` line with its exit status, without numbered badges, nested cards,
or inline output. Click any tool line to open Activity Expand and jump to its
entry, where the full command, timing, token/cost metadata, and captured output
are available. Long output stays out of the conversation transcript so it does
not overwhelm the messages that matter.
