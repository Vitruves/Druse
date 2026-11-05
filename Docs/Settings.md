# Settings

The Settings window, pane by pane, and the `settings` and `cli` commands that reach the same values from a script.

Open Settings from the Druse menu (⌘,). The window has a fixed size for every pane, so switching tabs never makes it jump; a pane taller than the window scrolls.

Contents: [Where settings live](#where-settings-live) · [General](#general) · [Viewport](#viewport) · [Docking](#docking) · [Logging](#logging) · [Performance](#performance) · [Console](#console) · [Command Line](#command-line) · [Shell Completions](#shell-completions) · [From a script](#from-a-script)

## Where settings live

The panes draw on three different stores, which is worth knowing because only one of them is reachable from DruseScript.

| Store | Contents | Reachable from a script |
|---|---|---|
| `AppSettings` | CPU cores, headless log level, console history length, autosuggestions, PDB cache folder, project folder | Yes — `settings` and `settings set` |
| App preferences (`@AppStorage`) | Appearance, docking confirmation, advanced-section state, file logging, log folder and retention | No |
| Workspace state | Viewport sliders and grid-box appearance | Partly — the equivalent `view` commands ([Visualization](Visualization.md)) |

Everything in the first two stores lives in the app's `UserDefaults` domain, which `Druse --headless` shares. A change made in Settings therefore applies to the next terminal run, and a `settings set` from a script shows up in the window — see [Console](Console.md).

## General

| Setting | Values | Default |
|---|---|---|
| Appearance | Automatic (follows the macOS system appearance), Light, Dark | Automatic |
| PDB download cache | Any folder; **Choose…** and **Reset** buttons | `~/.druse/pdb_cache` |
| Project folder | Any folder; the folder the Save Project panel opens in | `~/Documents` |

Downloaded PDB and mmCIF entries are kept in the cache for 7 days. If the project folder no longer exists, the save panel falls back to its own last location.

Both folders are also `settings set pdb-cache=` and `settings set project-dir=`.

## Viewport

Display defaults for the 3D view. These write through to the live renderer, so the effect is immediate.

| Setting | Range | Note |
|---|---|---|
| Gradient intensity | 0 – 1 | Background gradient. Set it to 0 for a plain white (light appearance) or black (dark appearance) backdrop, which is what you want for figures. |
| Surface opacity | 0.1 – 1 | Opacity of the molecular surface. |
| Grid box line thickness | 1 – 8, step 0.5 | Line width of the docking box wireframe. |
| Grid box colour | Colour picker, with opacity | Colour of the docking box wireframe. |

The same values are reachable per session as `view background <opacity>`, `view surface opacity=`, and `view grid width= color=` — see [Visualization](Visualization.md).

## Docking

| Setting | Effect | Default |
|---|---|---|
| Confirm before docking | Shows a review sheet with the protein, ligands, site and method before each run. Turn it off to start docking straight from **Run**. | On |
| Show advanced settings | Opens the Docking step with its Advanced section expanded. Expanding or collapsing the section in the step changes this setting too. | Off |

Docking parameters themselves are not software settings: they are session state, changed with `dock set` and shown with `dock config`. See [Docking](Docking.md).

## Logging

Controls the activity log's file output. The log *display* threshold is separate and belongs to the session (`log level <level>`); see [Console](Console.md).

| Setting | Effect | Default |
|---|---|---|
| Write logs to file | Each session creates a timestamped log file holding all console output. Toggling it starts or stops file logging immediately. | On |
| Log directory | Where those files go, as an editable path with **Choose…**, **Reset to Default** and **Open in Finder**. | `~/.druse/logs` |
| Keep logs for | 1 – 90 days. Files older than this are deleted automatically on launch. | 7 days |

The pane also shows the current session's log file with a **Reveal** button, the number and total size of `.log` files in the directory, and a **Prune Old Logs Now** button that applies the retention immediately. `log file` prints the path of the current session's log file from either console surface.

## Performance

### CPU

A single core budget, applied for real rather than advisory: it caps the C++ core's TBB global thread limit and the OpenMM CPU platform, and bounds Swift fan-out (ligand preparation, pocket-detection probes, pose clustering, GFN2 pose refinement) through an internal worker count.

| Row | Meaning |
|---|---|
| CPU cores | Slider plus stepper, `1` – the logical CPUs macOS currently lets the process use, shown as *"N of M"*. |
| In effect | The thread limit the C++ core reported after the last change. |
| Default | Performance cores − 1 (at least 1), so a performance core stays free for the interface and the window server while a long job runs. Shows a **Use N Cores** button when the current value differs. |

Changes apply immediately. A terminal run can override the budget for that process only with `druse --cores N`; the stored setting is untouched. `settings set cores=N` changes the stored value and clears any session override.

### This Mac

Read-only, for reference when choosing a budget:

| Row | Content |
|---|---|
| Processor | Total logical cores, split into performance and efficiency cores when the machine reports both (for example `8 cores (4 performance + 4 efficiency)`). |
| GPU | Metal device name, whether its memory is unified or discrete, and its recommended maximum working-set size. |
| Memory | Physical memory. |

The same three lines appear under **Machine** in the output of the `settings` command.

## Console

| Setting | Values | Default |
|---|---|---|
| History length | 10 – 10000 commands, stepped by 50 | 500 |
| Suggest completions while typing | On / off — the inline grey suggestion in the in-app terminal | On |
| Log level (Terminal Runs) | `debug`, `info`, `warn`, `error`, `none` | `warn` |

History length applies to both the in-app terminal and the `druse` prompt's `~/.druse/console_history`, and takes effect on the next command without a restart.

The **Terminal Runs** log level is the default verbosity of activity-log lines that `druse` mirrors to stderr. Each level shows itself and everything more severe. Override it per run with `druse --log-level <level>`, or inside a script with `log level <level>`. It does not affect the app, where activity-log entries go to the Logs panel (⇧⌘Y) instead.

In script form: `settings set history=1000 autosuggest=off log-level=info`.

## Command Line

Installs the `druse` terminal command, which runs DruseScript with no window.

| Row | Path | Notes |
|---|---|---|
| For me | `~/.local/bin/druse` | No password needed; may need adding to PATH |
| For all users | `/usr/local/bin/druse` | On the default PATH of every shell; asks for an administrator password, so it is available from the app only |

Each row reports its state and offers the matching button:

| State | Shown as | Button |
|---|---|---|
| Not installed | *Not installed* | **Install** |
| Installed for this bundle | *Installed: \<path\>* | **Uninstall** |
| Installed for another bundle | *Points to another Druse.app (\<path\>)* | **Update**, **Uninstall** |
| A file Druse did not write | *\<path\> exists but was not installed by Druse* | none — the file is left alone |

The **Your Shell** section runs your login shell (from the account database, not `$SHELL`) once as a login shell to read its PATH, and reports whether `~/.local/bin` is on it. If it is not, it shows the line to add and the file to add it to, with a **Copy** button:

| Shell | Line | Where |
|---|---|---|
| fish | `fish_add_path ~/.local/bin` | run once; fish remembers it |
| zsh | `export PATH="$HOME/.local/bin:$PATH"` | `~/.zshrc` |
| bash | `export PATH="$HOME/.local/bin:$PATH"` | `~/.bash_profile` |

Because PATH is read from a login shell that is not interactive, a PATH set only in `~/.zshrc` or `~/.bashrc` is not seen here; fish reads `config.fish` in every mode.

In script form: `cli status`, `cli install`, `cli install location=system`, `cli uninstall`. See [Console](Console.md).

## Shell Completions

Installs Tab completion for `druse`, generated from the console's command registry, so it completes commands, subcommands, options, their enum values and `.druse` files exactly as the in-app terminal does.

| Shell | Installed to | Setup hint shown once installed |
|---|---|---|
| fish | `~/.config/fish/completions/druse.fish` | none — loaded automatically |
| zsh | `~/.zfunc/_druse` | Add to `~/.zshrc`, before any `compinit`: `fpath=(~/.zfunc $fpath); autoload -Uz compinit && compinit` |
| bash | `~/.local/share/bash-completion/completions/druse` | Loaded automatically with bash-completion 2 (`brew install bash-completion@2`); otherwise `source` the file from `~/.bash_profile` |

Your login shell is marked *(your shell)*. Each row reports one of four states:

| State | Shown as | Button |
|---|---|---|
| Not installed | *Not installed* | **Install** |
| Current | *Installed* | **Uninstall** |
| Generated by an older Druse | *Installed, from an older Druse: update it* | **Update**, **Uninstall** |
| A file Druse did not generate | *Another file is there (not generated by Druse); left alone* | none |

Install again after updating Druse to pick up new commands.

In script form: `cli completions install`, `cli completions install shell=all`, `cli completions print zsh`, `cli completions uninstall`.

## From a script

`settings` prints every value with its default and description, the session flags, and the machine summary. `settings set` changes one or more values — they are saved and shared with the GUI and later terminal runs. `settings reset [name]` restores one setting, or all of them.

```
settings
settings set cores=4
settings set log-level=info history=1000
settings set pdb-cache=~/Data/pdb project-dir=~/Projects/druse
settings reset cores
settings reset
```

| `settings set` option | Type and bounds | Default | Pane |
|---|---|---|---|
| `cores=N` | integer, 1 – the cores this Mac makes available | performance cores − 1 | Performance |
| `log-level=` | `debug`, `info`, `warn`, `error`, `none` | `warn` | Console |
| `history=N` | integer, 10 – 10000 | 500 | Console |
| `autosuggest=` | `on` / `off` | `on` | Console |
| `pdb-cache=` | path | `~/.druse/pdb_cache` | General |
| `project-dir=` | path | `~/Documents` | General |

Session flags are separate from stored settings and are listed by `settings` alongside them:

| Flag | Values | Default |
|---|---|---|
| `set onerror=stop\|continue` | `stop` ends a script at the first failing statement; `continue` reports it and runs the next one | `stop` |
| `log level <level>` | Log threshold for this session (Logs panel in the app, stderr headless) | the stored `log-level` |
| `--cores N` | Core budget for one headless process, overriding the stored `cores` | the stored `cores` |

---

See also: [Console](Console.md) · [DruseScript](DruseScript.md#settings) · [Installation](Installation.md) · [Visualization](Visualization.md) · [README](../README.md)
