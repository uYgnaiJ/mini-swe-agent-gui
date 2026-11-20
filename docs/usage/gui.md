# Web GUI

`mini-gui` starts a small browser-based interface for mini-SWE-agent. It uses
Server-Sent Events (SSE) to stream the model response token by token, and shows
each bash command and its output as the agent works.

```bash
mini-gui
```

The command prints a URL such as `http://127.0.0.1:8000` and opens it in your
default browser. Enter a task in the text box and press **Run agent**.

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
| `--host` / `--port` | Bind address (the default is local-only `127.0.0.1:8000`) |
| `--browser` / `--no-browser` | Open or do not open a browser |

The GUI runs the agent non-interactively (equivalent to yolo mode), so it does
not ask for command confirmation in the browser. Only expose it on an untrusted
network if you understand that anyone who can reach it can ask the agent to run
commands in the configured environment.

## Streaming

Streaming is implemented by `LitellmModel.stream_query()`, which requests a
stream from the provider, emits `assistant_delta` events through an optional
agent callback, and then feeds the assembled response through the normal action
parsing and execution path. Other model implementations that do not support
streaming still work: the GUI displays their final assistant message when the
query returns.
