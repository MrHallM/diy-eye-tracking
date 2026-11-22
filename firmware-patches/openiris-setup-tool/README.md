# OpenIris setup tool fixes (24 Oct 2026)

Two bugs in `tools/setup_openiris.py` (OpenIris-ESPIDF), hit while setting up the two eye boards on Windows.
Both are upstream bugs, not ours; applied on our `combo` branch (`8d34dfc`, `883f2f7`).

| patch | bug | effect |
|---|---|---|
| 0001 | `valid_port()` tests `sys.platform == "windows"`, but Python on Windows reports `"win32"` | every `--port COMx` is rejected with "Port must be in /dev/tty", so the tool can't run on Windows at all |
| 0002 | `get_response_error()` assumes a `results` key, but `send_command()` returns `{"error": ...}` on a timeout | a timed-out command (e.g. `connect_wifi` with a wrong password) crashes the tool with `KeyError: 'results'` instead of printing the error |

Apply: `git am 000*.patch` in an OpenIris-ESPIDF checkout. Candidate upstream PR (small, no behaviour change on Linux/macOS).

Launcher: `tools/setup_eye_board.bat` (finds the XIAO's COM port by USB id 303A).
