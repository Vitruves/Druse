# Console and headless mode

Druse's two command surfaces: the terminal inside the app window, and the `druse` command that runs the same language from a shell with no window.

Both run [DruseScript](DruseScript.md), the command language that covers every step of the pipeline. This page documents the two surfaces — keys, flags, exit codes, installation; the language itself and the full command reference are in [DruseScript](DruseScript.md).

Contents: [In-app terminal](#in-app-terminal) · [Headless mode](#headless-mode) · [Installing the druse command](#installing-the-druse-command) · [Shell completions](#shell-completions) · [One language, two surfaces](#one-language-two-surfaces) · [First session](#first-session) · [Command groups](#command-groups)

## In-app terminal

Open it with **⌃\`** (View ▸ Show Terminal), or with the **Terminal** tab in the bottom bar. The same shortcut hides it again, and the panel is resizable by dragging its top edge (80–500 pt).

The terminal is a real terminal surface, not a log view. It shows only what you typed and what those commands printed; **activity-log entries never appear in it** — they have their own **Logs** tab (**⇧⌘Y**), with its own search, level and category filters. The bottom bar carries a warning and error counter that jumps to the matching entries in Logs. The two surfaces share the bar but are independent panes, so switching between them, hiding the panel or changing workflow step loses nothing: the session is created once per window, and its scrollback, history, the line being edited and any running command all survive.

### Keys

| Key | Action |
|---|---|
| Return | Run the line. With the completion list open, accept the highlighted candidate instead. |
| Tab | Complete the word at the caret (the caret must be at the end of the line). One candidate is inserted directly; several extend the line to their common prefix and open a list. |
| ↑ / ↓ | Move in the open completion list (wrapping); with no list, walk the command history. Inside a multi-line input they move the caret as usual. |
| ⇧Tab | Move backwards in the completion list. |
| Esc | Close the completion list. |
| → (also End, ⌘→) | At the end of the line, accept the whole inline suggestion. |
| ⌥→ | At the end of the line, accept the next word of the inline suggestion. |
| ⌃R | Reverse history search. Type to filter, ⌃R again for the next older match, Return runs the match, ⌫ shortens the query, any other key drops the match onto the line for editing. |
| ⌃G | Cancel the reverse search and restore the line being edited. |
| ⇧Return (also ⌥Return) | Continue the input on a new line; the whole input then runs as one script. Pasted multi-line text behaves the same way. |
| ⌃C | Cancel the running command, printing `^C`. With nothing running, abandon the line being typed and start a fresh prompt. |
| ⌘K, ⌃L | Clear the scrollback (the `clear` command does the same). History and the line being edited are kept. |
| ⌃A, ⌘← | Jump to the start of the input — never into the prompt. |

The caret cannot walk back into the `druse›` prompt, and typing or pasting while the selection sits in the scrollback moves the caret into the input first.

### Completion and suggestions

Tab completion is driven by the command registry and the session state, so it offers commands, subcommands, option names, enum option values, `$variables` and file paths (directory listings are capped at 200 entries so completing in a large folder stays instant). The floating list shows an icon for the candidate's kind, the candidate itself, and the registry's one-line description; a click accepts a row. Accepting a command, a subcommand, an `option=` or a folder keeps the list open on what can follow.

The inline suggestion is fish-style grey text after the caret: the most recent history line that extends what you typed, otherwise the common completion from the registry. Turn it off in Settings ▸ Console, or with `settings set autosuggest=off`.

`complete "<partial line>"` prints exactly what Tab would offer, which is useful inside scripts and when checking a command name:

```
complete "dock "
complete "prep pro"
complete "load ~/Doc"
```

### History

Submitted lines are remembered across launches and shared by ↑/↓, ⌃R, the inline suggestion and the `history` command (`history`, `history 50`, `history clear`). An immediate repeat is stored once. How many lines are kept comes from Settings ▸ Console (`settings set history=N`, 10–10000, default 500) and takes effect on the next command without a restart. The in-app history and the `druse` prompt's history are separate lists — the terminal's lives in the app's preferences, the prompt's in `~/.druse/console_history` — but both honour the same length setting.

### Panel buttons

While the Terminal surface is visible, the right of the bottom bar carries:

| Button | Action |
|---|---|
| **Run Script…** | Opens a file panel for a `.druse` (or plain text) file and runs it, announcing its start and outcome in the scrollback. Disabled while a command is running. |
| **Stop** | Appears only while a command runs; same as ⌃C. |
| **Copy All** | Copies the whole scrollback as plain text, with `druse›` before commands and `warning:` / `error:` before diagnostics — the same prefixes headless mode writes to stderr. |
| **Clear** | Clears the scrollback (⌘K). |

Opening a `.druse` file from Finder, or with `open -a Druse script.druse`, also runs it: the file is queued, the Terminal surface is shown, and the queued scripts run one after another.

## Headless mode

The app executable runs DruseScript with no window, no scene and **no Dock icon** when any of `--headless`, `-c`, `-e`, `--eval`, `-i`, `--interactive` or `--completions` is on its command line. The decision is made in `App/main.swift` before any `NSApplication` or SwiftUI scene exists, and the process sets its activation policy to `prohibited`.

```bash
/Applications/Druse.app/Contents/MacOS/Druse --headless script.druse
```

Installing the `druse` command (see [below](#installing-the-druse-command)) gives you a short form — `druse` is exactly `Druse.app/Contents/MacOS/Druse --headless "$@"`. The rest of this page uses it.

### Invocation forms

```bash
druse script.druse                     # run a script file, then exit
druse script.druse --log-level info    # flags may also follow a script path
druse dock preset=fast                 # one statement given as shell words
druse -c "fetch 1hsg; prep; pocket detect; dock preset=fast; results top=5"
druse                                  # interactive prompt (REPL) on a terminal
druse -i -c 'fetch 1HSG; prep'         # run first, then open the prompt with everything it loaded
cat steps.druse | druse                # stdin that is not a terminal runs as one script
druse - < steps.druse                  # the same, written explicitly
```

The first non-flag word is treated as a **script** when it ends in `.druse`, contains a `/`, or names an existing file that is not a command verb — so `druse status` still runs the `status` command in a folder that happens to hold a file called `status`. Otherwise it is a **command**, and every word after it belongs to that one statement (which is why flags must precede a command word, while they may follow a script path). Words the shell already split and unquoted are re-quoted so the statement re-parses to the same words: `druse load "my protein.pdb" name=x` works.

A script, `-c` and `-` are mutually exclusive; giving two of them is a usage error. Repeating `-c` is allowed — the sources are joined with newlines and run as one script.

### Modes

**One-shot** — a command, a script file, `-c`, or piped stdin. Each run starts from an empty session and exits when it is done, so loading, preparation and docking must happen in the same script or `-c` source. Command output goes to **stdout**; errors, warnings, echoed statements and the activity-log mirror go to **stderr**, so results can be piped or redirected without diagnostics mixed in:

```bash
druse run.druse > report.txt 2> run.log
```

Output is coloured and wrapped to the terminal width only when the stream is a terminal; it is plain when piped or redirected. Colour is also suppressed when `NO_COLOR` is set or `TERM=dumb`. The live progress line of a long command is drawn only on an interactive stderr, so logs and captured output stay free of control characters.

**Interactive** — `druse` alone on a terminal, or any form plus `-i`. A `druse› ` prompt keeps the session between statements, with line editing, history and Tab completion. A line ending in `\` continues on a `   … ` prompt. `exit`, `quit` or ⌃D leaves; ⌃C cancels the running command, and a second ⌃C exits.

History for the prompt is persisted in **`~/.druse/console_history`** (under `$HOME` as the shell sets it), trimmed to the Settings ▸ Console history length. Line editing is provided by macOS's own libedit (`/usr/lib/libedit.3.dylib`, loaded with `dlopen`), so the usual Emacs keys, ↑/↓ and ⌃R all work, and `~/.editrc` is honoured — prefix bindings there with `druse:`. If libedit cannot be loaded, the prompt falls back to plain line reads without editing or completion.

A syntax error is reported before anything runs, so a typo on line 40 never leaves a half-executed script behind.

### Flags

| Flag | Value | Effect |
|---|---|---|
| `-c`, `-e`, `--eval` | `source` | Run the given statements (`;` or newline separated). Repeatable. |
| `-i`, `--interactive` | — | After the script, `-c` or command has run, stay in the interactive prompt with its state. |
| `--var` | `name=value` | Predefine `$name`. Repeatable. |
| `--cores` | `N` | CPU cores for this run, overriding Settings ▸ Performance. Must be at least 1; clamped to the cores this Mac makes available. |
| `--log-level` | `debug`, `info`, `warn`, `error`, `none` | Activity-log verbosity mirrored to stderr. Default from Settings ▸ Console, normally `warn`. `warning` is accepted as an alias of `warn`. |
| `--echo` | — | Echo each statement to stderr as `druse› <statement>` before running it. |
| `--completions` | `fish`, `zsh`, `bash` | Print that shell's completion script to stdout and exit. |
| `-h`, `--help` | — | Print the usage — invocation forms, the two modes, every command by category, the flags, examples and exit codes — and exit. |
| `--headless` | — | Force headless mode (implied by the flags above). |
| `-` | — | Read the statements from stdin. |

A script file's own folder becomes `$script_dir`, and relative paths in it resolve against that folder. For `-c` and for a command given as words, relative paths resolve against the current working directory.

### Exit codes

| Code | Meaning |
|---|---|
| `0` | Success — every statement ran. |
| `1` | A command failed. With `set onerror=continue`, at least one statement failed. |
| `2` | Usage error (bad flag, unreadable script, two sources given) or a syntax error. Nothing ran. |
| `130` | Interrupted. ⌃C cancels the running command; a second ⌃C exits with this code. |

With `-i`, the exit code is that of the interactive session, normally `0`.

### Settings in headless runs

Headless runs share the app's `UserDefaults` domain, so a change made in Settings applies to the next terminal run and vice versa — see [Settings](Settings.md). `--cores` and `--log-level` override the stored values for one process only; `settings set …` in a script changes them for good.

## Installing the `druse` command

From the app: **Settings ▸ Command Line**. It installs the command, reports where it is installed, and runs your login shell once to tell you whether that folder is on its PATH — printing the exact line to add if it is not. The same is available as `cli` commands in either surface:

```
cli status                   # what is installed where, and whether it is current
cli install                  # ~/.local/bin/druse — no password
cli install location=system  # /usr/local/bin/druse — asks for an administrator password
cli uninstall                # removes only a wrapper Druse installed
cli uninstall location=system
```

| Location | Path | Notes |
|---|---|---|
| `user` (default) | `~/.local/bin/druse` | No administrator rights; may need adding to PATH |
| `system` | `/usr/local/bin/druse` | On the default PATH of every shell; asks for an administrator password, so it works from the app only |

What gets installed is a small bash wrapper, written with mode `0755` so no `chmod +x` is needed. It records the bundle path it was installed from and honours a `DRUSE_APP` environment override:

```bash
app="${DRUSE_APP:-/Applications/Druse.app}"
exec "$app/Contents/MacOS/Druse" --headless "$@"
```

After moving `Druse.app`, install again to update the recorded path; `cli status` reports a wrapper pointing at another bundle as stale. A `druse` file Druse did not write is never replaced or removed — the installer reports it and leaves it alone.

Putting `~/.local/bin` on PATH, per shell:

| Shell | Line | Where |
|---|---|---|
| fish | `fish_add_path ~/.local/bin` | run once; fish remembers it |
| zsh | `export PATH="$HOME/.local/bin:$PATH"` | `~/.zshrc` |
| bash | `export PATH="$HOME/.local/bin:$PATH"` | `~/.bash_profile` |

From a source checkout with no installed app, `Tools/druse` is a wrapper that locates a bundle by itself: `$DRUSE_APP`, then `/Applications/Druse.app`, `~/Applications/Druse.app`, then the newest Debug or Release build under the checkout's `build/`, `~/Library/Developer/Xcode/DerivedData`, or any `:`-separated folders in `$DRUSE_DERIVED_DATA`. Make it executable once and symlink it into your PATH. See [Installation](Installation.md).

## Shell completions

The completion scripts are generated from the same command registry the in-app terminal uses, so they complete commands, subcommands, options and their enum values identically, plus `.druse` files. Install them from **Settings ▸ Shell Completions**, or with `cli`:

```
cli completions install              # for your login shell
cli completions install shell=all    # fish, zsh and bash
cli completions print zsh            # to stdout, to redirect yourself
cli completions uninstall
```

`druse --completions <shell>` prints the same script without the app running:

```bash
druse --completions fish > ~/.config/fish/completions/druse.fish
```

| Shell | Installed to | Setup |
|---|---|---|
| fish | `~/.config/fish/completions/druse.fish` | Loaded automatically |
| zsh | `~/.zfunc/_druse` | Add to `~/.zshrc` before any `compinit`: `fpath=(~/.zfunc $fpath); autoload -Uz compinit && compinit` |
| bash | `~/.local/share/bash-completion/completions/druse` | Loaded by bash-completion 2 (`brew install bash-completion@2`); otherwise `source` the file from `~/.bash_profile` |

Each generated script carries a signature line, so Druse recognises its own files: `cli status` and Settings ▸ Shell Completions report an installed script generated by an older build as outdated, and a file Druse did not generate is never replaced or removed. Re-run `cli completions install` after updating Druse to pick up new commands.

## One language, two surfaces

The in-app terminal and `druse` run the same interpreter against the same command registry; only the mode differs. The one visible difference concerns the 3D view.

Commands whose only effect is on the 3D view are **skipped with a warning** in headless mode:

```
warning: 'view fit' only affects the 3D view — ignored in headless mode
```

These are the forms marked *GUI only* in `help` output: every form of `hide`, `show` and `center`; the camera and overlay forms of `view` (`view fit`, `view reset`, `view rotate`, `view zoom`, `view grid`, `view interactions`); `subset show` and `subset hide`; and `pose show`. Everything that changes state rather than pixels — including `view mode=`, `view surface`, `view clip` and `view status` — runs normally. A pose that cannot be shown can still be inspected: `results`, `interactions pose=N` and `contacts pose=N` report it as text.

**`view snapshot` works headless**, rendering the scene offscreen to a `.png`, `.jpg` or `.tiff`:

```
view snapshot figures/pose1.png width=2400 height=1600 fit=pocket theme=light
```

Without a viewport to measure, the headless default size is 1600 × 1200 and the default framing is `fit=all`. This makes a script a complete figure pipeline — see [Visualization](Visualization.md).

Two smaller differences: the activity log goes to the Logs panel in the app but is mirrored to stderr headless (at `--log-level`, or `log level <level>` for the rest of the run), and relative paths resolve against the running script's folder in both, falling back to the current working directory headless and the home folder in the app.

## First session

Install the command and check it:

```bash
# in the app's terminal (⌃`)
cli install
cli completions install
```

```bash
# in your shell
druse --help
druse version
```

Then run a complete study in one go — fetch a structure, prepare it, find the pocket, dock a ligand and write the ranking:

```bash
druse -c 'fetch 1HSG
          prep
          pocket detect
          ligand smiles "CC(=O)Oc1ccccc1C(=O)O" name=aspirin
          dock preset=fast
          results top=5
          export results /tmp/1hsg.csv'
```

Or do the same interactively and keep the loaded state to poke at:

```bash
druse -i -c 'fetch 1HSG; prep; pocket detect'
```

```
druse› status
druse› pocket list
druse› help dock
druse› dock preset=fast
druse› results top=5
druse› interactions pose=1
druse› view snapshot /tmp/pose1.png fit=pocket
druse› exit
```

Useful habits at the prompt: `help` lists every command by category, `help <command>` prints every form of one command with its arguments, types, ranges, defaults and examples, `apropos <word>` searches all descriptions, and `status` says what is currently loaded.

Runnable scripts live in [`Examples/`](../Examples) and are described in [DruseScript](DruseScript.md#examples):

```bash
druse Examples/dock_one_ligand.druse
druse Examples/batch_dock_and_export.druse --cores 8 --log-level info
```

## Command groups

Every command exists in both surfaces. `help` prints this same grouping; the links go to the full reference.

| Group | Commands | Topic |
|---|---|---|
| Session | [`help`](DruseScript.md#help), [`apropos`](DruseScript.md#apropos), [`complete`](DruseScript.md#complete), [`set`](DruseScript.md#set), [`unset`](DruseScript.md#unset), [`vars`](DruseScript.md#vars), [`echo`](DruseScript.md#echo), [`log`](DruseScript.md#log), [`history`](DruseScript.md#history), [`wait`](DruseScript.md#wait), [`run`](DruseScript.md#run), [`status`](DruseScript.md#status), [`version`](DruseScript.md#version), [`clear`](DruseScript.md#clear) | [DruseScript](DruseScript.md#syntax) |
| Structures | [`load`](DruseScript.md#load), [`fetch`](DruseScript.md#fetch), [`project`](DruseScript.md#project), [`chain`](DruseScript.md#chain), [`residue`](DruseScript.md#residue), [`protein`](DruseScript.md#protein) | [Preparation](Preparation.md) |
| Selection | [`select`](DruseScript.md#select), [`selection`](DruseScript.md#selection), [`subset`](DruseScript.md#subset) | [Visualization](Visualization.md) |
| Protein preparation | [`prep`](DruseScript.md#prep) | [Preparation](Preparation.md) |
| Ligands | [`ligand`](DruseScript.md#ligand) | [Ligands](Ligands.md) |
| Docking | [`pocket`](DruseScript.md#pocket), [`dock`](DruseScript.md#dock), [`results`](DruseScript.md#results), [`pose`](DruseScript.md#pose), [`flex`](DruseScript.md#flex), [`constraint`](DruseScript.md#constraint), [`refine`](DruseScript.md#refine), [`rescore`](DruseScript.md#rescore), [`strain`](DruseScript.md#strain), [`screen`](DruseScript.md#screen) | [BindingSite](BindingSite.md), [Docking](Docking.md), [Scoring](Scoring.md), [VirtualScreening](VirtualScreening.md) |
| Analysis | [`interactions`](DruseScript.md#interactions), [`contacts`](DruseScript.md#contacts), [`rmsd`](DruseScript.md#rmsd), [`fingerprint`](DruseScript.md#fingerprint), [`validity`](DruseScript.md#validity) | [Analysis](Analysis.md) |
| Export | [`export`](DruseScript.md#export) | [Analysis](Analysis.md) |
| Lead optimization | [`leadopt`](DruseScript.md#leadopt) | [LeadOptimization](LeadOptimization.md) |
| View | [`hide`](DruseScript.md#hide), [`show`](DruseScript.md#show), [`center`](DruseScript.md#center), [`view`](DruseScript.md#view) | [Visualization](Visualization.md) |
| Settings | [`settings`](DruseScript.md#settings), [`cli`](DruseScript.md#cli) | [Settings](Settings.md) |

---

See also: [DruseScript](DruseScript.md) for the language and the full command reference · [Settings](Settings.md) · [Installation](Installation.md) · [README](../README.md)
