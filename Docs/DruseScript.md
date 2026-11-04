# DruseScript

DruseScript is the command language of Druse. Every step of the GUI pipeline (loading, protein preparation, ligand preparation, pocket detection, docking, analysis, export, lead optimization, view control, settings) has a console command, so a whole study can be typed in the in-app terminal, saved as a `.druse` script, or run from the macOS command line with no window.

DruseScript is a console language, not a programming language: one command per line, no `if`, no loops, no blocks. Batch work is done by dedicated commands (`ligand prep all`, `dock batch`, `screen`, `leadopt dock`).

Contents: [Quick start](#quick-start) · [Syntax](#syntax) · [Session settings](#session-settings) · [In-app terminal](#in-app-terminal) · [Command line (headless)](#command-line-headless) · [Installing `druse` and shell completions](#installing-druse-and-shell-completions) · [Settings](#settings) · [Examples](#examples) · [Command reference](#command-reference)

## Quick start

```
load ~/targets/1add_protein.pdb
load ~/targets/1add_ligand.sdf
prep waters=remove
pocket box fit around=ligand padding=6
dock preset=standard
results top=5
export poses ~/Desktop/poses.sdf top=5
```

Type `help` for every command by category, `help dock` for every form of one command with its arguments, types, ranges, defaults and examples, and `apropos <word>` to search all descriptions (`apropos tautomer`).

## Syntax

A statement is `verb [subcommand…] [arguments…] [key=value…]`.

| Element | Rule | Example |
|---|---|---|
| Statement end | A newline or `;` outside quotes | `prep missing; prep repair` |
| Line continuation | `\` as the last character of a line joins the next line | `dock batch ligands=prepared \`<br>`preset=fast` |
| Comment | `#` at the start of a word, to the end of the line (`a#b` stays one word) | `prep  # full pipeline` |
| Quoted text | `"…"` keeps spaces; escapes `\"`, `\\`, `\n`, `\t`; cannot span lines | `ligand smiles "CC(=O)Oc1ccccc1C(=O)O" name=aspirin` |
| Options | `key=value` with no spaces around `=`; the value may be quoted (`name="my ligand"`); an empty value is written `key=""` | `dock set population=150 runs=5` |
| Booleans | `on` / `off` (also `true`/`false`, `yes`/`no`, `1`/`0`) | `ligand smiles "c1ccccc1" prep=off` |
| Variables | `set name = value` defines `$name`; `$name` or `${name}` is substituted in words and quoted strings at run time | `set ph = 6.5` then `prep protonate ph=$ph` |
| Case | Command and option names are case-insensitive; values keep their case | `DOCK PRESET=fast` |
| Paths | `~` expands to your home folder; relative paths resolve against the running script's folder (`$script_dir`), or the current folder for `--eval` and the terminal | `load ../data/1add_protein.pdb` |
| Ligands | A name or the 1-based number shown by `ligand list`; commands taking several accept names, numbers, ranges (`2-5`) and `all`, `prepared`, `unprepared`, `failed` | `ligand show 3`, `ligand prep 2-4`, `ligand remove failed` |
| Residues | `[chain:]number`, `[chain:]first-last`, `[chain:]NAME` or `[chain:]NAMEnumber`, several separated by spaces | `select residue A:25-30 HIS57`, `flex add A:HIS17 A:ASP19` |

Errors are reported as `file:line:column: message`. An unknown command or option gets a did-you-mean suggestion (`unknown command 'dok' — did you mean 'dock'?`). A syntax error stops the script before anything runs.

## Session settings

| Statement | Effect |
|---|---|
| `set onerror=stop` | Default: a failed statement ends the script (headless exit code 1). |
| `set onerror=continue` | Report a failed statement and run the next one; the script still exits 1 at the end. |
| `set script_dir=<path>` | Folder relative paths resolve against; scripts set it to their own folder automatically. |
| `log level debug\|info\|warn\|error\|none` | Log threshold: entries shown in the Logs panel (GUI) or mirrored to stderr (headless). `log level` alone shows the current level. |
| `unset <name>` | Remove a variable; `unset onerror` restores `stop`. |
| `vars` | List variables and session settings. |
| `run <file>` | Run another script; it honours `onerror` like the caller. |

Persistent software settings (CPU cores, default headless log level, history length, autosuggestions, folders) are changed with `settings set` and are shared by the GUI, the terminal and headless runs; see [Settings](#settings).

## In-app terminal

Open it with the Terminal tab of the bottom bar, View ▸ Show Terminal or ⌃\`. The terminal shows only commands and their output; the activity log has its own Logs tab (⇧⌘Y). A long command shows a live progress line.

| Key | Action |
|---|---|
| Return | Run the line (or accept the highlighted completion when the list is open) |
| Tab | Complete the command, subcommand, option name, option value, ligand, residue, chain or file path at the caret. With several candidates a list opens. |
| ↑ / ↓, ⇧Tab | Move in the open completion list (wraps around); without a list, walk the command history |
| Esc | Close the completion list |
| → (or ⌘→, End) at the end of the line | Accept the whole grey autosuggestion |
| ⌥→ at the end of the line | Accept the next word of the autosuggestion |
| ⌃R | Reverse history search: type to filter, ⌃R again for the next older match, Return runs the match, ⌫ shortens the query, any other key puts the match on the line for editing, ⌃G or ⌃C cancels |
| ⇧Return (or ⌥Return) | Continue on a new line; the whole input then runs as one script (pasted multi-line text runs the same way) |
| ⌃C | Cancel the running command and print `^C`; with nothing running, abandon the line being typed |
| ⌘K, ⌃L | Clear the screen (`clear` does the same); history and the line being edited are kept |

Autosuggestions are fish-style grey text after the caret: the most recent history line that extends what you typed, otherwise the common completion from the command registry. Settings ▸ Console turns them off and sets how many history lines are kept (`settings set autosuggest=off history=1000`). `complete "<partial line>"` prints the candidates Tab would offer, and `history` lists the remembered lines.

Commands that only affect the 3D view (`hide`, `show`, `center`, `view …`) are accepted headless but have no visible effect there, except `view snapshot`, which renders offscreen.

## Command line (headless)

`druse` (installed below) is `Druse.app/Contents/MacOS/Druse --headless`. It runs DruseScript with no window and no Dock icon.

```
druse script.druse                          # run a script
druse script.druse --log-level info         # flags may follow the script path
druse --eval "fetch 1hsg; prep; pocket detect; dock preset=fast; results"
druse dock preset=fast                      # one statement given as shell words
druse                                       # interactive prompt (line editing, history, Tab completion)
cat steps.druse | druse                     # stdin that is not a terminal is read as one script
druse --var receptor=~/t/1hsg.pdb run.druse # predefine $receptor
```

A first word that is not a `.druse` file or an existing file is a command; the remaining words form one statement.

| Flag | Value | Effect |
|---|---|---|
| `-e`, `--eval` | `source` | Run the given statements (`;` or newline separated) |
| `--var` | `name=value` | Predefine `$name` (repeatable). A `set` of the same name in the script replaces it. |
| `--cores` | `N` (1 – cores of this Mac) | CPU cores for this run, overriding Settings ▸ Performance |
| `--log-level` | `debug\|info\|warn\|error\|none` | Activity log mirrored to stderr (default from Settings, normally `warn`) |
| `--echo` | — | Echo each statement to stderr before running it |
| `--completions` | `fish\|zsh\|bash` | Print the shell completion script and exit |
| `-h`, `--help` | — | Show the usage and exit |

Command output goes to stdout; log lines, warnings and errors go to stderr, so `druse run.druse > report.txt` keeps the tables and `2> log.txt` keeps the diagnostics.

| Exit code | Meaning |
|---|---|
| `0` | Every statement succeeded |
| `1` | A command failed (with `set onerror=continue`, at least one failed) |
| `2` | Usage error (bad flag) or syntax error in the script; nothing was run |
| `130` | Interrupted: Ctrl-C cancels the running command, a second Ctrl-C exits |

## Installing `druse` and shell completions

From the app: Settings ▸ Command Line installs the `druse` command and shows whether its folder is on your shell's PATH (with the line to add if not); Settings ▸ Shell Completions installs Tab completion. The same is available as commands:

```
cli install                        # ~/.local/bin/druse, no password
cli install location=system        # /usr/local/bin/druse, administrator prompt (GUI only)
cli completions install            # for your login shell
cli completions install shell=all  # fish, zsh and bash
cli status                         # what is installed where, and whether it is current
cli uninstall                      # removes only files Druse installed
```

| Shell | Completion file | Notes |
|---|---|---|
| fish | `~/.config/fish/completions/druse.fish` | Loaded automatically |
| zsh | `~/.zfunc/_druse` | Needs `fpath=(~/.zfunc $fpath)` before `compinit` in `~/.zshrc` |
| bash | `~/.local/share/bash-completion/completions/druse` | Loaded by bash-completion 2 |

`~/.local/bin` must be on your PATH (`fish_add_path ~/.local/bin` in fish; `export PATH="$HOME/.local/bin:$PATH"` in `~/.zshrc` or `~/.bash_profile`). The completion scripts are generated from the command registry: they complete commands, subcommands, options and enum values exactly as the in-app terminal does. Re-run `cli completions install` after updating Druse (`cli status` reports an outdated script).

From a source checkout without an installed app, `Tools/druse` is a wrapper that finds `Druse.app` in `$DRUSE_APP`, `/Applications`, `~/Applications` or the newest local build; `chmod +x Tools/druse` and symlink it into your PATH.

## Settings

| Setting | `settings set` option | Default | Where in the GUI |
|---|---|---|---|
| CPU cores Druse may use at once (C++ core, ligand preparation, minimisation) | `cores=N` | performance cores − 1 | Settings ▸ Performance |
| Default headless log level on stderr | `log-level=` | `warn` | Settings ▸ Console |
| Console history length | `history=N` (10 – 10000) | 500 | Settings ▸ Console |
| Inline autosuggestions in the terminal | `autosuggest=on\|off` | `on` | Settings ▸ Console |
| PDB download cache (entries kept 7 days) | `pdb-cache=<path>` | app cache folder | Settings ▸ General |
| Folder the Save Project panel starts in | `project-dir=<path>` | — | Settings ▸ General |

`settings` shows every value with its default; `settings reset [setting]` restores one or all. Docking defaults are not software settings: they are session state changed with `dock set` and shown with `dock config`. Ligand preparation defaults are changed with `ligand prep set` and shown with `ligand prep config`.

## Examples

Runnable scripts on the PDBbind complex 1add (adenosine deaminase with 1-deazaadenosine) are in [`Examples/`](../Examples). Each reads `Benchmark/data/refined-set/1add` relative to its own folder and writes its outputs to `/tmp` (edit the `set out` line to change that).

| Script | What it shows |
|---|---|
| [`protein_preparation.druse`](../Examples/protein_preparation.druse) | Inspecting the receptor, then each preparation stage on its own: altlocs, non-standard residues, waters, hydrogen removal, atom repair, FASPR packing, pKa-aware protonation, H-bond network, charges, minimisation, report, export |
| [`prepare_ligand_library.druse`](../Examples/prepare_ligand_library.druse) | Session preparation defaults, ligands from SMILES and SDF, affinities, protomer × tautomer × conformer enumeration, forms, conformers, pKa, descriptors, ADMET, library export |
| [`dock_one_ligand.druse`](../Examples/dock_one_ligand.druse) | Redocking the crystal ligand: one-shot preparation, box fitted around the ligand, docking, results, clusters, interactions, RMSD to the crystal pose, validity, export |
| [`batch_dock_and_export.druse`](../Examples/batch_dock_and_export.druse) | A small library prepared in dominant mode, `dock batch`, the ranking and its CSV export |
| [`lead_optimization.druse`](../Examples/lead_optimization.druse) | Docking, a lead-optimization reference, analog generation, analog docking, export and hand-off to the Ligand Database |

```
druse Examples/dock_one_ligand.druse
```

## Command reference

Generated from the command registry (`help <command>` prints the same text in the console). Types: `text`, `path`, `integer`, `number`, `on|off` (also `true|false`, `yes|no`, `1|0`), `one of a|b|c`. *Takes the rest of the line* means the argument swallows every remaining word.

### Session

#### `help`

List commands by category, or show every form, argument and option of one command.

**`help [topic…]`** — Without a topic: all commands; with one: usage, types, ranges, defaults and examples

| Argument / option | Description | Type, range, default |
|---|---|---|
| `<topic>` | Command, optionally followed by subcommand words (help ligand db) | text, optional, takes the rest of the line |

Examples:

```
help
help dock
help ligand db
help set
```

#### `apropos`

Search command, subcommand, argument and option summaries for a word.

**`apropos <word…>`** — List every command form whose name or descriptions mention the word

| Argument / option | Description | Type, range, default |
|---|---|---|
| `<word>` | Text to look for (case-insensitive; several words are searched as one phrase) | text, takes the rest of the line |

Examples:

```
apropos charge
apropos tautomer
apropos exhaustiveness
```

#### `complete`

List the completions the console offers for a partial command line.

**`complete <line>`** — One candidate per line: completed line, kind and description (tab-separated)

| Argument / option | Description | Type, range, default |
|---|---|---|
| `<line>` | Partial line, quoted to keep a trailing space ("dock ") | text |

Examples:

```
complete "dock "
complete "prep pro"
complete "load ~/Doc"
```

#### `set`

Define a $variable, or change a session setting such as onerror.

**`set <name> = <value>   |   set onerror=stop|continue   |   set script_dir=<path>`** — Assign a value (spaces around '=' are optional; quote values containing spaces)

| Argument / option | Description | Type, range, default |
|---|---|---|
| `<name>` | Variable name: letters, digits and '_', not starting with a digit | text |
| `<value>` | Value; $other variables are substituted first | text |
| `onerror=` | stop: a failed statement ends the script; continue: report it and run the next one | one of stop\|continue, default stop |
| `script_dir=` | Folder relative paths resolve against (scripts set it to their own folder) | path |

Examples:

```
set ph = 7.4
set receptor="~/targets/1hsg.pdb"
set onerror=continue
echo "pH $ph"
```

#### `unset`

Remove a variable (removing onerror restores the default, stop).

**`unset <name>`** — Forget the variable

| Argument / option | Description | Type, range, default |
|---|---|---|
| `<name>` | Variable name, without '$' | text |

Examples:

```
unset ph
unset onerror
```

#### `vars`

List variables and session settings with their values.

**`vars`** — Show every variable and its value

Examples:

```
set ph = 7.0
vars
```

#### `echo`

Print text, with $variables substituted.

**`echo [text…]`** — Print the words, separated by spaces

| Argument / option | Description | Type, range, default |
|---|---|---|
| `<text>` | Text to print | text, optional, takes the rest of the line |

Examples:

```
echo "Docking $ligand at pH $ph"
```

#### `log`

Set how much activity-log output is shown, list recent entries, or locate the log file.

**`log level [level]`** — Show or set the log threshold (GUI: entries shown in the Logs panel; headless: stderr, for the rest of the run)

| Argument / option | Description | Type, range, default |
|---|---|---|
| `<level>` | debug, info, warn, error or none; omit to show the current level | one of debug\|info\|warn\|error\|none, optional |

**`log show [count] [level=debug|info|warn|error|none] [category=system|render|pdb|prep|dock|smiles|selection|network|molecule]`** — Print the most recent activity-log entries

| Argument / option | Description | Type, range, default |
|---|---|---|
| `<count>` | Number of entries | integer 1–2000, default 20 |
| `level=` | Minimum severity (default: the current log level) | one of debug\|info\|warn\|error\|none |
| `category=` | Only entries of this category | one of system\|render\|pdb\|prep\|dock\|smiles\|selection\|network\|molecule |

**`log file`** — Print the path of this session's log file

Examples:

```
log level info
log level
log show 50 level=warn
log show category=dock
log file
```

#### `history`

Show or clear the console's command history.

**`history [count]`** — List the most recent command lines, oldest first

| Argument / option | Description | Type, range, default |
|---|---|---|
| `<count>` | Number of entries | integer 1–10000, default 20 |

**`history clear`** — Forget every remembered command line

Examples:

```
history
history 50
history clear
```

#### `wait`

Pause for a number of seconds.

**`wait <seconds>`** — Sleep (Stop / Ctrl-C cancels it)

| Argument / option | Description | Type, range, default |
|---|---|---|
| `<seconds>` | Duration in seconds | number 0–86400 |

Examples:

```
wait 2.5
```

#### `run`

Run a .druse script (relative to the current script's folder).

**`run <file>`** — Run every statement of the file; honours onerror like the calling script

| Argument / option | Description | Type, range, default |
|---|---|---|
| `<file>` | Script file | path |

Examples:

```
run prepare.druse
run ~/scripts/screen_kinases.druse
```

#### `status`

Show what is loaded: protein, ligand, pocket, results, analogs and key settings.

**`status`** — Summarize the workspace

Examples:

```
status
```

#### `version`

Print the Druse version and build number.

**`version`** — Show the version of the running app

Examples:

```
version
```

#### `clear`

Clear the console transcript.

**`clear`** — Empty the console view (the activity log and its file are kept)

Examples:

```
clear
```

### Structures

#### `load`

Load a structure or ligand file (pdb, cif, sdf, mol, mol2, smi, csv).

**`load <file> [chains=text] [waters=keep|remove]`** — PDB/mmCIF replace the protein; SDF/MOL/MOL2 set the ligand; SMI/CSV import into the Ligand Database

| Argument / option | Description | Type, range, default |
|---|---|---|
| `<file>` | File to load | path |
| `chains=` | Keep only these protein chains, comma-separated (e.g. A or A,B) | text |
| `waters=` | Keep crystallographic waters (default) or delete them on load | one of keep\|remove |

Examples:

```
load 1hsg.pdb
load complex.cif chains=A waters=remove
load ~/ligands/aspirin.sdf
load library.smi
```

#### `fetch`

Download a structure from the RCSB PDB (cached locally).

**`fetch <id> [chains=text] [waters=keep|remove]`** — Load a PDB entry by its 4-character ID; its first HET ligand becomes the active ligand

| Argument / option | Description | Type, range, default |
|---|---|---|
| `<id>` | PDB ID, e.g. 1HSG | text |
| `chains=` | Keep only these protein chains, comma-separated (e.g. A or A,B) | text |
| `waters=` | Keep crystallographic waters (default) or delete them on load | one of keep\|remove |

Examples:

```
fetch 1HSG
fetch 4HHB chains=A,B waters=remove
```

#### `project`

Save or open a .druse project folder.

**`project save <dir>`** — Save protein, ligands, results and settings

| Argument / option | Description | Type, range, default |
|---|---|---|
| `<dir>` | Project folder (…/name.druse) | path |

**`project open <dir>`** — Replace the workspace with a saved project

| Argument / option | Description | Type, range, default |
|---|---|---|
| `<dir>` | Project folder | path |

Examples:

```
project save ~/Druse/hiv_protease.druse
project open ~/Druse/hiv_protease.druse
```

#### `chain`

List, remove, rename, merge or extract protein chains; print their sequence.

**`chain list`** — Chains with type, residue range, residue and atom counts

**`chain remove <id>`** — Delete a chain from the protein (and its extracted ligand, if any)

| Argument / option | Description | Type, range, default |
|---|---|---|
| `<id>` | Chain ID (case-sensitive) | text |

**`chain rename <id> <new>`** — Give a chain a new ID

| Argument / option | Description | Type, range, default |
|---|---|---|
| `<id>` | Chain ID (case-sensitive) | text |
| `<new>` | New chain ID (1–3 printable characters) | text |

**`chain merge <id> <into>`** — Move every residue of one chain into another

| Argument / option | Description | Type, range, default |
|---|---|---|
| `<id>` | Chain ID (case-sensitive) | text |
| `<into>` | Target chain ID | text |

**`chain extract <id>`** — Move a chain out of the protein and make it the active ligand (added to the Ligand Database)

| Argument / option | Description | Type, range, default |
|---|---|---|
| `<id>` | Chain ID (case-sensitive) | text |

**`chain sequence [id] [format=plain|fasta] [copy=on|off]`** — Print the one-letter amino-acid sequence of a chain (all chains when omitted)

| Argument / option | Description | Type, range, default |
|---|---|---|
| `<id>` | Chain ID (default: every protein chain) | text, optional |
| `format=` | plain (default) or FASTA records with a >name_chain header | one of plain\|fasta |
| `copy=` | Also copy it to the clipboard (GUI only, default off) | on\|off |

Examples:

```
chain list
chain remove B
chain rename B C
chain merge B A
chain extract L
chain sequence A
chain sequence format=fasta copy=on
```

#### `residue`

List residues, inspect one residue's atoms and protonation, or delete residues.

**`residue list [chain=text] [type=all|amino|het|water] [name=text]`** — Residues of the protein, filtered by chain, kind or name

| Argument / option | Description | Type, range, default |
|---|---|---|
| `chain=` | Only this chain | text |
| `type=` | all (default), amino = standard amino acids, het = ligands/ions/modified residues, water | one of all\|amino\|het\|water |
| `name=` | Only residues with this 3-letter name (e.g. HIS) | text |

**`residue list het [chain=text]`** — Hetero groups only: ligands, ions, cofactors, modified residues (no water)

| Argument / option | Description | Type, range, default |
|---|---|---|
| `chain=` | Only this chain | text |

**`residue info <residues…>`** — Atoms of the matched residues with charges, B-factors and the assigned protonation state

| Argument / option | Description | Type, range, default |
|---|---|---|
| `<residues>` | Residues as [chain:]number, [chain:]first-last, [chain:]NAME or [chain:]NAMEnumber (e.g. A:25, A:25-30, HIS57, B:HOH) | text, takes the rest of the line |

**`residue remove <residues…>`** — Delete residues from the protein (cannot be undone; clears the selection and subsets)

| Argument / option | Description | Type, range, default |
|---|---|---|
| `<residues>` | Residues as [chain:]number, [chain:]first-last, [chain:]NAME or [chain:]NAMEnumber (e.g. A:25, A:25-30, HIS57, B:HOH) | text, takes the rest of the line |

Examples:

```
residue list chain=A
residue list het
residue list type=water
residue list name=HIS
residue info A:HIS57
residue info A:25-27
residue remove A:SO4
```

#### `protein`

Summarise the loaded protein.

**`protein info`** — Atoms, residues, chains, waters, HET groups, mass, charge, secondary structure and preparation state

Examples:

```
protein info
```

### Selection

#### `select`

Select protein residues by number, chain, distance, secondary structure or pocket.

**`select residue <residues…> [add=on|off]`** — Select residues by specification

| Argument / option | Description | Type, range, default |
|---|---|---|
| `<residues>` | Residues as [chain:]number, [chain:]first-last, [chain:]NAME or [chain:]NAMEnumber (e.g. A:25, A:25-30, HIS57, B:HOH) | text, takes the rest of the line |
| `add=` | Add to the current selection instead of replacing it (default off) | on\|off |

**`select chain <id>`** — Select every residue of a chain (and ligand atoms carrying that chain ID)

| Argument / option | Description | Type, range, default |
|---|---|---|
| `<id>` | Chain ID (case-sensitive) | text |

**`select within <distance> <of> <target> [add=on|off] [pose=N]`** — Select residues with any atom within a distance of the ligand, the pocket or the current selection

| Argument / option | Description | Type, range, default |
|---|---|---|
| `<distance>` | Cutoff in Å | number |
| `<of>` | The word 'of' | one of of |
| `<target>` | ligand = displayed ligand (or pose=N), pocket = docking pocket probes, selection = grow the selection | one of ligand\|pocket\|selection |
| `add=` | Add to the current selection instead of replacing it (default off) | on\|off |
| `pose=` | With 'of ligand': measure from docked pose N instead of the displayed ligand | integer |

**`select ss <type> [chain=text]`** — Select residues by secondary structure (assigned from backbone H-bonds when the file has none)

| Argument / option | Description | Type, range, default |
|---|---|---|
| `<type>` | Secondary-structure class; loop = coil | one of helix\|sheet\|turn\|loop |
| `chain=` | Only this chain | text |

**`select pocket [add=on|off]`** — Select the residues lining the selected docking pocket

| Argument / option | Description | Type, range, default |
|---|---|---|
| `add=` | Add to the current selection instead of replacing it (default off) | on\|off |

**`select all`** — Select every protein residue

**`select none`** — Clear the selection

**`select invert`** — Select every protein residue that is not selected, and deselect the rest

**`select extend [count=N]`** — Grow the selection along the chain by residues at each end

| Argument / option | Description | Type, range, default |
|---|---|---|
| `count=` | Number of residues to grow or shrink by at each end (default 1) | integer |

**`select shrink [count=N]`** — Remove residues from each end of the selected segments

| Argument / option | Description | Type, range, default |
|---|---|---|
| `count=` | Number of residues to grow or shrink by at each end (default 1) | integer |

Examples:

```
select residue A:25 A:27-30 HIS57
select residue B:HOH add=on
select chain A
select within 5 of ligand
select within 4.5 of ligand pose=1
select within 3 of selection
select ss helix chain=A
select pocket
select invert
select extend count=2
select none
```

#### `selection`

Inspect the selection, turn it into the docking pocket, or delete it.

**`selection info [residues=on|off]`** — Selected residues by chain, atom count, centre and extent

| Argument / option | Description | Type, range, default |
|---|---|---|
| `residues=` | Also list every residue with its name (default off) | on\|off |

**`selection pocket`** — Define the docking pocket (and grid box) from the selected residues

**`selection delete`** — Delete the selected residues from the protein (cannot be undone)

Examples:

```
selection info
selection info residues=on
select within 6 of ligand; selection pocket
selection delete
```

#### `subset`

Named residue sets: save the selection, recall it, show or hide it.

**`subset save <name>`** — Save the selected residues as a named subset

| Argument / option | Description | Type, range, default |
|---|---|---|
| `<name>` | Subset name (quote names with spaces) | text |

**`subset list`** — List saved subsets

**`subset select <name>`** — Make a subset the current selection

| Argument / option | Description | Type, range, default |
|---|---|---|
| `<name>` | Subset name (quote names with spaces) | text |

**`subset remove <name>`** — Delete a subset (the residues stay)

| Argument / option | Description | Type, range, default |
|---|---|---|
| `<name>` | Subset name (quote names with spaces) | text |

**`subset show <name>`** — Highlight a subset in the 3D view (GUI only; ignored in headless mode)

| Argument / option | Description | Type, range, default |
|---|---|---|
| `<name>` | Subset name (quote names with spaces) | text |

**`subset hide <name>`** — Stop highlighting a subset in the 3D view (GUI only; ignored in headless mode)

| Argument / option | Description | Type, range, default |
|---|---|---|
| `<name>` | Subset name (quote names with spaces) | text |

Examples:

```
select within 5 of ligand; subset save site
subset list
subset select site
subset hide site
subset remove site
```

### Protein preparation

#### `prep`

Prepare the protein: cleanup, repair, side-chain packing, protonation, H-bond network, charges, minimisation.

**`prep [option=value…]`** — Full pipeline of the Preparation step: short loops, altlocs, non-standard residues, caps, missing atoms + FASPR repack, restrained minimisation, protonation at pH, hydrogens, H-bond network, charges

| Argument / option | Description | Type, range, default |
|---|---|---|
| `ph=` | Protonation pH, 0–14 (persists; 7.4 at start) | number |
| `charges=` | Partial-charge method (persists), or none to skip charges | one of gasteiger\|eem\|qeq\|xtb\|none |
| `waters=` | keep all waters, remove all, or keep only waters near the selected pocket (pocket) | one of keep\|remove\|pocket |
| `radius=` | With waters=pocket: retention radius around the pocket centre in Å (persists; default 5) | number |
| `nonstandard=` | Remove ligands, cofactors and modified residues (structural metal ions and caps are always kept) | one of remove\|keep |
| `caps=` | Cap uncapped chain-break termini with ACE/NME (default on) | on\|off |
| `loops=` | Model missing loops of up to 15 residues before capping (default on) | on\|off |
| `hydrogens=` | all (default), polar only, or none = heavy-atom receptor (no protonation) | one of all\|polar\|none |

**`prep all [option=value…]`** — Same as 'prep'

| Argument / option | Description | Type, range, default |
|---|---|---|
| `ph=` | Protonation pH, 0–14 (persists; 7.4 at start) | number |
| `charges=` | Partial-charge method (persists), or none to skip charges | one of gasteiger\|eem\|qeq\|xtb\|none |
| `waters=` | keep all waters, remove all, or keep only waters near the selected pocket (pocket) | one of keep\|remove\|pocket |
| `radius=` | With waters=pocket: retention radius around the pocket centre in Å (persists; default 5) | number |
| `nonstandard=` | Remove ligands, cofactors and modified residues (structural metal ions and caps are always kept) | one of remove\|keep |
| `caps=` | Cap uncapped chain-break termini with ACE/NME (default on) | on\|off |
| `loops=` | Model missing loops of up to 15 residues before capping (default on) | on\|off |
| `hydrogens=` | all (default), polar only, or none = heavy-atom receptor (no protonation) | one of all\|polar\|none |

**`prep clean [option=value…]`** — Cleanup only: preferred alternate locations, non-standard residues, waters, chain-break caps

| Argument / option | Description | Type, range, default |
|---|---|---|
| `altlocs=` | Keep only the preferred alternate location of each atom (default on) | on\|off |
| `nonstandard=` | Remove ligands, cofactors and modified residues (structural metal ions and caps are always kept) | one of remove\|keep |
| `waters=` | keep all waters, remove all, or keep only waters near the selected pocket (pocket) | one of keep\|remove\|pocket |
| `radius=` | With waters=pocket: retention radius around the pocket centre in Å (persists; default 5) | number |
| `caps=` | Cap uncapped chain-break termini with ACE/NME (default off; needs altlocs=on) | on\|off |

**`prep altlocs`** — Keep only the preferred alternate location (highest occupancy, then lowest B)

**`prep nonstandard`** — Remove ligands, cofactors and modified residues (keeps waters, caps and structural metal ions)

**`prep waters [mode] [radius=x]`** — Remove all waters, or keep only those near the selected pocket

| Argument / option | Description | Type, range, default |
|---|---|---|
| `<mode>` | remove (default) or pocket | one of remove\|pocket, optional |
| `radius=` | With waters=pocket: retention radius around the pocket centre in Å (persists; default 5) | number |

**`prep caps`** — Cap uncapped chain-break termini with ACE/NME

**`prep missing [all=on|off]`** — Report missing residues (gaps), chain breaks and residues with missing or extra atoms

| Argument / option | Description | Type, range, default |
|---|---|---|
| `all=` | List every incomplete residue instead of the first 20 (default off) | on\|off |

**`prep repair`** — Build short missing loops, rebuild missing heavy atoms, re-protonate if hydrogens were present

**`prep repair all`** — Same as 'prep repair'

**`prep repair atoms`** — Rebuild missing heavy atoms of existing residues from templates

**`prep repair loops [max-gap=N] [refine=gpu|openmm|none]`** — Model missing loops (SEQRES identities when available, else polyalanine) and relax them

| Argument / option | Description | Type, range, default |
|---|---|---|
| `max-gap=` | Longest gap to build, in residues (default 15) | integer |
| `refine=` | Loop relaxation: gpu = restrained Metal minimisation (default), openmm, none = raw builder geometry | one of gpu\|openmm\|none |

**`prep pack [residues=incomplete|all|selection]`** — Repack side chains with FASPR (heavy-atom model; remove hydrogens first)

| Argument / option | Description | Type, range, default |
|---|---|---|
| `residues=` | incomplete = rebuild + repack residues missing side-chain atoms (default), all, or the selected residues | one of incomplete\|all\|selection |

**`prep protonate [ph=x]`** — Predict pKa-shifted protonation states and add polar hydrogens at pH (with H-bond network optimisation)

| Argument / option | Description | Type, range, default |
|---|---|---|
| `ph=` | Protonation pH, 0–14 (persists; 7.4 at start) | number |

**`prep states [type=all|charged|his|shifted]`** — Protonation states chosen by the last protonation (pKa shifts, His tautomers, charges)

| Argument / option | Description | Type, range, default |
|---|---|---|
| `type=` | all titratable residues (default), charged ones, histidines, or pKa shifted by more than 1 | one of all\|charged\|his\|shifted |

**`prep hydrogens [ph=x]`** — Same as 'prep hydrogens add'

| Argument / option | Description | Type, range, default |
|---|---|---|
| `ph=` | Protonation pH, 0–14 (persists; 7.4 at start) | number |

**`prep hydrogens add [ph=x]`** — Add all hydrogens (polar + non-polar) from residue templates at pH

| Argument / option | Description | Type, range, default |
|---|---|---|
| `ph=` | Protonation pH, 0–14 (persists; 7.4 at start) | number |

**`prep hydrogens remove`** — Remove every hydrogen from the protein (and the ligand)

**`prep hbonds`** — Re-optimise the H-bond network: OH/SH/NH3 rotors, Asn/Gln flips, His tautomers and flips

**`prep charges [method=gasteiger|eem|qeq|xtb]`** — Assign partial charges

| Argument / option | Description | Type, range, default |
|---|---|---|
| `method=` | Partial-charge method (persists): gasteiger = RDKit on the PDB text, eem/qeq = electronegativity equalisation, xtb = GFN2-xTB | one of gasteiger\|eem\|qeq\|xtb |

**`prep minimize [steps=N]`** — Restrained GPU minimisation (backbone strongly restrained, side chains moderately, H free)

| Argument / option | Description | Type, range, default |
|---|---|---|
| `steps=` | Maximum L-BFGS iterations (default 100) | integer |

**`prep report`** — What preparation did to the current protein, step by step, and its current state

Examples:

```
prep
prep ph=6.5 charges=eem
prep waters=pocket radius=4 hydrogens=polar
prep loops=off caps=off charges=none
prep clean waters=remove caps=on
prep missing
prep repair loops refine=openmm
prep repair atoms; prep pack
prep protonate ph=7.0; prep states type=his
prep hydrogens remove; prep pack residues=all; prep hydrogens add
prep hbonds
prep charges method=qeq
prep minimize steps=200
prep report
```

### Ligands

#### `ligand`

Build, import, prepare (protomers × tautomers × conformers), inspect and export ligands.

**`ligand smiles <smiles> [option=value…]`** — Add a ligand from SMILES, prepare it with the session pipeline and make it the docking ligand

| Argument / option | Description | Type, range, default |
|---|---|---|
| `<smiles>` | SMILES string (quote it) | text |
| `name=` | Entry name (default Ligand_<n>) | text |
| `prep=` | Prepare right away (off: add as an unprepared entry) | on\|off, default on |
| `use=` | Make the prepared ligand (its dominant form) the docking ligand | on\|off, default on |
| `mode=` | Pipeline: enumerate (every protomer/tautomer as its own entry), dominant (most populated form, in place) or basic (RDKit 3D without pH) | one of enumerate\|dominant\|basic, default enumerate |
| `ph=` | Target pH for protonation states and form populations | number 0–14, default 7.4 |
| `pka=` | pKa source: gnn (Metal GNN, MAE ~1.0) or rules (SMARTS table, ~230 groups) | one of gnn\|rules, default gnn |
| `pka-threshold=` | Sites with \|pKa − pH\| below this get both protonation states | number 0–10, default 2 |
| `max-tautomers=` | Maximum tautomers per protomer (mode=enumerate) | integer 1–100, default 10 |
| `max-protomers=` | Maximum protonation states of the parent (mode=enumerate) | integer 1–64, default 8 |
| `energy-window=` | Discard forms more than this many kcal/mol (MMFF94) above the best one | number 0–100, default 10 |
| `conformers=` | 3D conformers per chemical form (enumerate/dominant) | integer 1–500, default 10 |
| `min-population=` | Drop enumerated forms below this population in % (dominant form always kept; 0 enumerates every site) | number 0–100, default 1 |
| `temperature=` | Boltzmann temperature for form populations, in K | number 100–1000, default 298.15 |
| `hydrogens=` | Add hydrogens (mode=basic; the ensemble modes always add them) | on\|off, default on |
| `minimize=` | MMFF94-minimize the 3D structure (mode=basic; always on otherwise) | on\|off, default on |
| `charges=` | Assign Gasteiger partial charges (mode=basic; always on otherwise) | on\|off, default on |

**`ligand load <file> [option=value…]`** — Add ligands from a .smi, .csv/.tsv, .sdf/.mol, .mol2 or Druse .json file (not prepared unless prep=on)

| Argument / option | Description | Type, range, default |
|---|---|---|
| `<file>` | Ligand file | path |
| `format=` | File format (auto: from the extension; json is a saved Druse ligand database) | one of auto\|smi\|csv\|sdf\|mol2\|json, default auto |
| `smiles-col=` | CSV: SMILES column, by header name or 1-based number | text |
| `name-col=` | CSV: name/ID column, by header name or number | text |
| `ki-col=` | CSV: Ki column (nM) | text |
| `pki-col=` | CSV: pKi column (−log10 Ki in M) | text |
| `ic50-col=` | CSV: IC50 column (nM) | text |
| `delimiter=` | CSV: field separator (auto: tab if the first line has one, else comma) | one of auto\|comma\|tab\|semicolon, default auto |
| `header=` | CSV: whether the first row is a header | one of auto\|on\|off, default auto |
| `extra-cols=` | CSV: keep unmapped columns as entry properties | on\|off, default on |
| `prep=` | Prepare the new entries after loading (sdf/mol2 with prep=off keep their file 3D coordinates) | on\|off, default off |
| `mode=` | Pipeline: enumerate (every protomer/tautomer as its own entry), dominant (most populated form, in place) or basic (RDKit 3D without pH) | one of enumerate\|dominant\|basic, default enumerate |
| `ph=` | Target pH for protonation states and form populations | number 0–14, default 7.4 |
| `pka=` | pKa source: gnn (Metal GNN, MAE ~1.0) or rules (SMARTS table, ~230 groups) | one of gnn\|rules, default gnn |
| `pka-threshold=` | Sites with \|pKa − pH\| below this get both protonation states | number 0–10, default 2 |
| `max-tautomers=` | Maximum tautomers per protomer (mode=enumerate) | integer 1–100, default 10 |
| `max-protomers=` | Maximum protonation states of the parent (mode=enumerate) | integer 1–64, default 8 |
| `energy-window=` | Discard forms more than this many kcal/mol (MMFF94) above the best one | number 0–100, default 10 |
| `conformers=` | 3D conformers per chemical form (enumerate/dominant) | integer 1–500, default 10 |
| `min-population=` | Drop enumerated forms below this population in % (dominant form always kept; 0 enumerates every site) | number 0–100, default 1 |
| `temperature=` | Boltzmann temperature for form populations, in K | number 100–1000, default 298.15 |
| `hydrogens=` | Add hydrogens (mode=basic; the ensemble modes always add them) | on\|off, default on |
| `minimize=` | MMFF94-minimize the 3D structure (mode=basic; always on otherwise) | on\|off, default on |
| `charges=` | Assign Gasteiger partial charges (mode=basic; always on otherwise) | on\|off, default on |

**`ligand rename <ligand> <new-name>`** — Rename an entry (its enumerated forms follow)

| Argument / option | Description | Type, range, default |
|---|---|---|
| `<ligand>` | Entry name or 1-based # from 'ligand list' | text |
| `<new-name>` | New entry name | text |

**`ligand remove <ligands…>`** — Remove entries; removing a source molecule also removes its enumerated forms

| Argument / option | Description | Type, range, default |
|---|---|---|
| `<ligands>` | Names, #numbers, ranges (2-5), or all \| unprepared \| prepared \| failed | text, takes the rest of the line |

**`ligand clear [what]`** — Empty the Ligand Database, or remove only its forms, unprepared or failed entries

| Argument / option | Description | Type, range, default |
|---|---|---|
| `<what>` | What to remove | one of all\|forms\|unprepared\|failed, default all |

**`ligand affinity <ligand> [ki=x] [pki=x] [ic50=x] [clear=on|off]`** — Show or set experimental affinity (copied to the entry's enumerated forms)

| Argument / option | Description | Type, range, default |
|---|---|---|
| `<ligand>` | Entry name or 1-based # from 'ligand list' | text |
| `ki=` | Ki in nM | number 1e-06–1000000000000 |
| `pki=` | pKi (−log10 Ki in M) | number 0–20 |
| `ic50=` | IC50 in nM | number 1e-06–1000000000000 |
| `clear=` | Remove all affinity values first | on\|off |

**`ligand list [filter]`** — List the Ligand Database (forms indented under their source molecule; * = docking ligand)

| Argument / option | Description | Type, range, default |
|---|---|---|
| `<filter>` | Which entries to list (forms: enumerated forms with populations) | one of all\|prepared\|unprepared\|forms\|dockable\|failed, default all |

**`ligand show <ligand>`** — Everything about one entry: SMILES, status, 3D, descriptors, Lipinski/Veber, forms, conformers, affinity, errors

| Argument / option | Description | Type, range, default |
|---|---|---|
| `<ligand>` | Entry name or 1-based # from 'ligand list' | text |

**`ligand use <ligand> [form=text] [conformer=N]`** — Make an entry the docking ligand (a source molecule uses its most populated form unless form= is given)

| Argument / option | Description | Type, range, default |
|---|---|---|
| `<ligand>` | Entry name or 1-based # from 'ligand list' | text |
| `form=` | Form # from 'ligand forms', its label (Taut1) or entry name | text |
| `conformer=` | Conformer # from 'ligand conformers' (default: lowest energy) | integer 1–10000 |

**`ligand forms <ligand>`** — Protomers/tautomers of a molecule: kind, population at pH, relative energy, charge, conformers

| Argument / option | Description | Type, range, default |
|---|---|---|
| `<ligand>` | Entry name or 1-based # from 'ligand list' | text |

**`ligand conformers <ligand> [action] [count=N] [minimize=on|off]`** — List an entry's conformers with MMFF94 energies, or generate a new set

| Argument / option | Description | Type, range, default |
|---|---|---|
| `<ligand>` | Entry name or 1-based # from 'ligand list' | text |
| `<action>` | generate replaces the conformers (keeps the entry's protonation) | one of list\|generate, default list |
| `count=` | generate: number of conformers to embed | integer 1–500, default 10 |
| `minimize=` | generate: MMFF94-minimize each conformer | on\|off, default on |

**`ligand prep <ligands…> [option=value…]`** — Prepare entries with the session pipeline; options override the session defaults for this run

| Argument / option | Description | Type, range, default |
|---|---|---|
| `<ligands>` | Names, #numbers, ranges (2-5), or all \| unprepared \| prepared \| failed | text, takes the rest of the line |
| `mode=` | Pipeline: enumerate (every protomer/tautomer as its own entry), dominant (most populated form, in place) or basic (RDKit 3D without pH) | one of enumerate\|dominant\|basic, default enumerate |
| `ph=` | Target pH for protonation states and form populations | number 0–14, default 7.4 |
| `pka=` | pKa source: gnn (Metal GNN, MAE ~1.0) or rules (SMARTS table, ~230 groups) | one of gnn\|rules, default gnn |
| `pka-threshold=` | Sites with \|pKa − pH\| below this get both protonation states | number 0–10, default 2 |
| `max-tautomers=` | Maximum tautomers per protomer (mode=enumerate) | integer 1–100, default 10 |
| `max-protomers=` | Maximum protonation states of the parent (mode=enumerate) | integer 1–64, default 8 |
| `energy-window=` | Discard forms more than this many kcal/mol (MMFF94) above the best one | number 0–100, default 10 |
| `conformers=` | 3D conformers per chemical form (enumerate/dominant) | integer 1–500, default 10 |
| `min-population=` | Drop enumerated forms below this population in % (dominant form always kept; 0 enumerates every site) | number 0–100, default 1 |
| `temperature=` | Boltzmann temperature for form populations, in K | number 100–1000, default 298.15 |
| `hydrogens=` | Add hydrogens (mode=basic; the ensemble modes always add them) | on\|off, default on |
| `minimize=` | MMFF94-minimize the 3D structure (mode=basic; always on otherwise) | on\|off, default on |
| `charges=` | Assign Gasteiger partial charges (mode=basic; always on otherwise) | on\|off, default on |

**`ligand prep config`** — Show the session preparation defaults and any run in progress

**`ligand prep set [option=value…]`** — Change session preparation defaults (also used by the Ligand Database window)

| Argument / option | Description | Type, range, default |
|---|---|---|
| `mode=` | Pipeline: enumerate (every protomer/tautomer as its own entry), dominant (most populated form, in place) or basic (RDKit 3D without pH) | one of enumerate\|dominant\|basic, default enumerate |
| `ph=` | Target pH for protonation states and form populations | number 0–14, default 7.4 |
| `pka=` | pKa source: gnn (Metal GNN, MAE ~1.0) or rules (SMARTS table, ~230 groups) | one of gnn\|rules, default gnn |
| `pka-threshold=` | Sites with \|pKa − pH\| below this get both protonation states | number 0–10, default 2 |
| `max-tautomers=` | Maximum tautomers per protomer (mode=enumerate) | integer 1–100, default 10 |
| `max-protomers=` | Maximum protonation states of the parent (mode=enumerate) | integer 1–64, default 8 |
| `energy-window=` | Discard forms more than this many kcal/mol (MMFF94) above the best one | number 0–100, default 10 |
| `conformers=` | 3D conformers per chemical form (enumerate/dominant) | integer 1–500, default 10 |
| `min-population=` | Drop enumerated forms below this population in % (dominant form always kept; 0 enumerates every site) | number 0–100, default 1 |
| `temperature=` | Boltzmann temperature for form populations, in K | number 100–1000, default 298.15 |
| `hydrogens=` | Add hydrogens (mode=basic; the ensemble modes always add them) | on\|off, default on |
| `minimize=` | MMFF94-minimize the 3D structure (mode=basic; always on otherwise) | on\|off, default on |
| `charges=` | Assign Gasteiger partial charges (mode=basic; always on otherwise) | on\|off, default on |

**`ligand prep reset`** — Restore the start-up preparation defaults

**`ligand prep stop`** — Cancel the preparation running in the Ligand Database (window or console)

**`ligand descriptors <ligands…>`** — Table of RDKit descriptors (MW, logP, TPSA, HBD/HBA, rotatable bonds, rings, Fsp3, Ro5, Veber)

| Argument / option | Description | Type, range, default |
|---|---|---|
| `<ligands>` | Names, #numbers, ranges (2-5), or all \| unprepared \| prepared \| failed | text, takes the rest of the line |

**`ligand admet <ligands…>`** — ADMET predictions (logP, logD7.4, logS, hERG, CYP2D6/3A4, BBB, metabolic stability, drug-likeness)

| Argument / option | Description | Type, range, default |
|---|---|---|
| `<ligands>` | Names, #numbers, ranges (2-5), or all \| unprepared \| prepared \| failed | text, takes the rest of the line |

**`ligand pka <ligand> [method=gnn|rules|both] [ph=x]`** — Per-site pKa predictions and ionization at pH (GNN, SMARTS rules, or both side by side)

| Argument / option | Description | Type, range, default |
|---|---|---|
| `<ligand>` | Entry name or 1-based # from 'ligand list' | text |
| `method=` | Prediction source | one of gnn\|rules\|both, default both |
| `ph=` | pH for the ionized fraction (default: session pH) | number 0–14 |

**`ligand db import <file> [option=value…]`** — Add and prepare ligands from a file (as 'ligand load' with prep=on)

| Argument / option | Description | Type, range, default |
|---|---|---|
| `<file>` | Ligand file (.smi, .csv, .tsv, .sdf, .mol, .mol2, .json) | path |
| `format=` | File format (auto: from the extension; json is a saved Druse ligand database) | one of auto\|smi\|csv\|sdf\|mol2\|json, default auto |
| `smiles-col=` | CSV: SMILES column, by header name or 1-based number | text |
| `name-col=` | CSV: name/ID column, by header name or number | text |
| `ki-col=` | CSV: Ki column (nM) | text |
| `pki-col=` | CSV: pKi column (−log10 Ki in M) | text |
| `ic50-col=` | CSV: IC50 column (nM) | text |
| `delimiter=` | CSV: field separator (auto: tab if the first line has one, else comma) | one of auto\|comma\|tab\|semicolon, default auto |
| `header=` | CSV: whether the first row is a header | one of auto\|on\|off, default auto |
| `extra-cols=` | CSV: keep unmapped columns as entry properties | on\|off, default on |
| `prep=` | Prepare the new entries (off: same as 'ligand load') | on\|off, default on |
| `mode=` | Pipeline: enumerate (every protomer/tautomer as its own entry), dominant (most populated form, in place) or basic (RDKit 3D without pH) | one of enumerate\|dominant\|basic, default enumerate |
| `ph=` | Target pH for protonation states and form populations | number 0–14, default 7.4 |
| `pka=` | pKa source: gnn (Metal GNN, MAE ~1.0) or rules (SMARTS table, ~230 groups) | one of gnn\|rules, default gnn |
| `pka-threshold=` | Sites with \|pKa − pH\| below this get both protonation states | number 0–10, default 2 |
| `max-tautomers=` | Maximum tautomers per protomer (mode=enumerate) | integer 1–100, default 10 |
| `max-protomers=` | Maximum protonation states of the parent (mode=enumerate) | integer 1–64, default 8 |
| `energy-window=` | Discard forms more than this many kcal/mol (MMFF94) above the best one | number 0–100, default 10 |
| `conformers=` | 3D conformers per chemical form (enumerate/dominant) | integer 1–500, default 10 |
| `min-population=` | Drop enumerated forms below this population in % (dominant form always kept; 0 enumerates every site) | number 0–100, default 1 |
| `temperature=` | Boltzmann temperature for form populations, in K | number 100–1000, default 298.15 |
| `hydrogens=` | Add hydrogens (mode=basic; the ensemble modes always add them) | on\|off, default on |
| `minimize=` | MMFF94-minimize the 3D structure (mode=basic; always on otherwise) | on\|off, default on |
| `charges=` | Assign Gasteiger partial charges (mode=basic; always on otherwise) | on\|off, default on |

**`ligand db export <file> [format=auto|sdf|csv|smi|json] [subset=all|prepared|dockable|forms] [conformers=best|all]`** — Write entries as SDF (3D), CSV (all data), SMI or Druse JSON

| Argument / option | Description | Type, range, default |
|---|---|---|
| `<file>` | Output file; the format follows the extension | path |
| `format=` | Output format (auto: from the extension) | one of auto\|sdf\|csv\|smi\|json, default auto |
| `subset=` | Entries to write (dockable: what 'dock batch' docks) | one of all\|prepared\|dockable\|forms, default all |
| `conformers=` | SDF: lowest-energy conformer only, or one record per conformer | one of best\|all, default best |

**`ligand db stats`** — Counts, preparation state, conformers, affinity coverage and property ranges

**`ligand db save`** — Save the database to the default file (the window's Save Database)

**`ligand db load`** — Replace the database with the saved default file (the window's Load Saved Database)

Examples:

```
ligand smiles "CC(=O)Oc1ccccc1C(=O)O" name=aspirin
ligand smiles "Cn1cnc2c1c(=O)n(C)c(=O)n2C" name=caffeine prep=off
ligand load actives.csv smiles-col=canonical_smiles name-col=chembl_id pki-col=pchembl
ligand prep all ph=7.4 max-tautomers=4 conformers=3
ligand prep unprepared mode=dominant pka=rules
ligand prep set ph=6.5 energy-window=6; ligand prep config
ligand list forms
ligand show imatinib
ligand forms imatinib
ligand use imatinib form=2 conformer=3
ligand conformers caffeine generate count=20
ligand pka imatinib ph=5.5
ligand descriptors all
ligand admet all
ligand affinity imatinib ki=12.5
ligand db export ~/Druse/ligands.csv subset=dockable
ligand db stats
```

### Docking

#### `pocket`

Detect, list and select binding pockets, or define and fit the docking box.

**`pocket detect [method=auto|geometric|ml|ligand]`** — Find pockets and select the best (auto = ML + geometric when the ML model is available)

| Argument / option | Description | Type, range, default |
|---|---|---|
| `method=` | auto, geometric only, ML only, or around the current ligand | one of auto\|geometric\|ml\|ligand |

**`pocket list`** — List detected pockets (* = selected)

**`pocket select <n>`** — Select a detected pocket for docking

| Argument / option | Description | Type, range, default |
|---|---|---|
| `<n>` | 1-based index from 'pocket list' | integer |

**`pocket residues <residues…> [padding=x]`** — Define the pocket around residues (box = their atoms + padding)

| Argument / option | Description | Type, range, default |
|---|---|---|
| `<residues>` | Residues as [chain:][name]number, e.g. A:ASP189 TYR95 120 | text, takes the rest of the line |
| `padding=` | Margin around the residues in Å (default 4) | number |

**`pocket box [center=x,y,z] [size=x,y,z] [grid-spacing=x]`** — Set the docking box explicitly (size = full edge lengths); keeps the current center or size when omitted

| Argument / option | Description | Type, range, default |
|---|---|---|
| `center=` | Box center in Å | x,y,z vector |
| `size=` | Box edge lengths in Å (default: current box, else 20,20,20) | x,y,z vector |
| `grid-spacing=` | Affinity-grid spacing; coarsened automatically when the grid exceeds GPU memory (0.2–1 Å) | number |

**`pocket box show`** — Print the docking box, grid spacing and estimated grid size

**`pocket box fit [around=ligand|pocket] [padding=x] [grid-spacing=x]`** — Fit the docking box around the ligand or the selected pocket's residues

| Argument / option | Description | Type, range, default |
|---|---|---|
| `around=` | What to enclose (default: ligand when one is loaded) | one of ligand\|pocket |
| `padding=` | Margin on every side in Å (default 4) | number |
| `grid-spacing=` | Affinity-grid spacing; coarsened automatically when the grid exceeds GPU memory (0.2–1 Å) | number |

Examples:

```
pocket detect
pocket detect method=ligand
pocket list; pocket select 2
pocket residues A:ASP189 SER195 HIS57 padding=5
pocket box center=12.5,-3,40.2 size=22,22,22
pocket box grid-spacing=0.5
pocket box fit around=ligand padding=6
pocket box show
```

#### `dock`

Dock the active ligand or Ligand Database entries; view and change every docking setting.

**`dock [option=value…]`** — Dock the active ligand into the selected pocket; options persist for later runs

| Argument / option | Description | Type, range, default |
|---|---|---|
| `preset=` | Budget preset: auto = Adapts parameters to protein/ligand complexity at launch; fast = Quick exploration: 50 × 30 × 3; standard = Balanced: 150 × 80 × 5; thorough = Deep search: 300 × 200 × 10 | one of auto\|fast\|standard\|thorough |
| `search=` | Search algorithm | one of ga\|fragment\|remc\|diffusion\|auto |
| `scoring=` | Scoring function used during search and for ranking | one of vina\|drusina\|druseaf\|pignet2 |
| `charges=` | Partial-charge model for the receptor and ligand | one of gasteiger\|eem\|qeq\|xtb |
| `population=` | Poses per population (8–4096) | integer |
| `generations=` | Generations (Monte Carlo steps) per run (1–10000) | integer |
| `runs=` | Independent runs (restarts) (1–100) | integer |

**`dock batch [option=value…]`** — Dock Ligand Database entries one after another and rank them

| Argument / option | Description | Type, range, default |
|---|---|---|
| `ligands=` | all (default: entries with 3D coordinates), prepared, or comma-separated names | text |
| `preset=` | Budget preset: auto = Adapts parameters to protein/ligand complexity at launch; fast = Quick exploration: 50 × 30 × 3; standard = Balanced: 150 × 80 × 5; thorough = Deep search: 300 × 200 × 10 | one of auto\|fast\|standard\|thorough |
| `search=` | Search algorithm | one of ga\|fragment\|remc\|diffusion\|auto |
| `scoring=` | Scoring function used during search and for ranking | one of vina\|drusina\|druseaf\|pignet2 |
| `charges=` | Partial-charge model for the receptor and ligand | one of gasteiger\|eem\|qeq\|xtb |
| `population=` | Poses per population (8–4096) | integer |
| `generations=` | Generations (Monte Carlo steps) per run (1–10000) | integer |
| `runs=` | Independent runs (restarts) (1–100) | integer |

**`dock set [option=value…]`** — Change docking settings for the session (validated; see 'dock config')

| Argument / option | Description | Type, range, default |
|---|---|---|
| `preset=` | Budget preset: auto = Adapts parameters to protein/ligand complexity at launch; fast = Quick exploration: 50 × 30 × 3; standard = Balanced: 150 × 80 × 5; thorough = Deep search: 300 × 200 × 10 | one of auto\|fast\|standard\|thorough |
| `search=` | Search algorithm | one of ga\|fragment\|remc\|diffusion\|auto |
| `scoring=` | Scoring function used during search and for ranking | one of vina\|drusina\|druseaf\|pignet2 |
| `charges=` | Partial-charge model for the receptor and ligand | one of gasteiger\|eem\|qeq\|xtb |
| `auto-tune=` | Adapt population, generations, runs, grid and local search to the system at launch | on\|off |
| `population=` | Poses per population (8–4096) | integer |
| `generations=` | Generations (Monte Carlo steps) per run (1–10000) | integer |
| `runs=` | Independent runs (restarts) (1–100) | integer |
| `live-update=` | Refresh the live pose every N generations (higher = fewer GPU syncs) (1–10000) | integer |
| `grid-spacing=` | Affinity-grid spacing; coarsened automatically when the grid exceeds GPU memory (0.2–1 Å) | number |
| `mutation-rate=` | Per-gene mutation probability after the exploration phase (0–1) | number |
| `crossover-rate=` | Crossover probability between parent poses (0–1) | number |
| `translation-step=` | Maximum translation of a mutation after exploration (0.05–20 Å) | number |
| `rotation-step=` | Maximum rotation of a mutation after exploration (0.01–3.1416 rad) | number |
| `torsion-step=` | Maximum torsion change of a mutation (0.01–3.1416 rad) | number |
| `mc-temperature=` | Metropolis temperature of the Monte Carlo acceptance test (0.05–10 kcal/mol) | number |
| `ls-frequency=` | Run gradient local search every N generations after exploration (1–100) | integer |
| `ls-steps=` | Gradient-descent steps per local search (0–500) | integer |
| `analytical-gradients=` | Analytical gradients (~28× fewer evaluations) instead of finite differences | on\|off |
| `explore-ratio=` | Fraction of generations spent in the broad exploration phase (0–1) | number |
| `explore-translation=` | Maximum translation of a mutation during exploration (0.05–30 Å) | number |
| `explore-rotation=` | Maximum rotation of a mutation during exploration (0.01–3.1416 rad) | number |
| `explore-mutation=` | Mutation probability during exploration (0–1) | number |
| `explore-ls-frequency=` | Run local search every N generations during exploration (1–100) | integer |
| `ligand-flex=` | Sample ligand rotatable bonds (off = rigid input conformer) | on\|off |
| `torsion-preset=` | Torsion-sampling policy; balanced, redocking and exploratory also set the five torsion-* values | one of balanced\|redocking\|exploratory\|manual |
| `torsion-exact=` | Fraction of torsion genes kept at the input value (0–1) | number |
| `torsion-local=` | Fraction of torsion genes perturbed locally around the input value (0–1) | number |
| `torsion-width=` | Amplitude of local torsion perturbations (0–3.1416 rad) | number |
| `torsion-reset=` | Probability of resetting a torsion to a random angle (0–1) | number |
| `torsion-scale=` | Scale applied to torsion mutation steps (0–3) | number |
| `rerank-clusters=` | Top cluster representatives rescored against explicit receptor atoms (0 = off) (0–100) | integer |
| `rerank-variants=` | Seeded variants refined around each reranked representative (1–32) | integer |
| `rerank-steps=` | Local-search steps of the rerank refinement (0–200) | integer |
| `strain=` | Add an MMFF94 strain penalty after docking (not applied with scoring=vina) | on\|off |
| `strain-threshold=` | Strain tolerated before penalizing (docked − relaxed MMFF94 energy) (0–100 kcal/mol) | number |
| `strain-weight=` | Penalty = weight × (strain − threshold) (0–10) | number |
| `fragment-beam=` | Partial poses kept at each fragment growth step (1–1024) | integer |
| `fragment-anchors=` | Anchor placements sampled in the site (1–8192) | integer |
| `fragment-prune=` | Discard partial poses this far above the best one (0–100 kcal/mol) | number |
| `fragment-torsions=` | Torsion angles sampled per connecting bond during growth (1–360) | integer |
| `fragment-scaffold=` | SMARTS of the scaffold fragment growth must keep as anchor; 'auto' uses the largest rigid fragment | text |
| `remc-replicas=` | Temperature replicas (2–64) | integer |
| `remc-tmin=` | Lowest replica temperature (0.05–50 kcal/mol) | number |
| `remc-tmax=` | Highest replica temperature (0.05–50 kcal/mol) | number |
| `remc-swap=` | Attempt replica swaps every N steps (1–1000) | integer |
| `remc-steps=` | Monte Carlo steps per replica (1–100000) | integer |
| `diffusion-steps=` | Reverse-diffusion denoising steps (1–1000) | integer |
| `diffusion-schedule=` | Noise variance schedule | one of linear\|cosine\|quadratic |
| `diffusion-guidance=` | Scale of the DruseAF attention-gradient guidance (0–10) | number |
| `diffusion-refine=` | Vina gradient local-search steps after denoising (0–1000) | integer |
| `gfn2=` | Refine the top poses with GFN2-xTB after every docking run | on\|off |
| `gfn2-top=` | Top poses optimized (1–500) | integer |
| `gfn2-solvation=` | Implicit solvation (none = gas phase) | one of none\|alpb\|gbsa |
| `gfn2-level=` | Optimization convergence level | one of crude\|normal\|tight |
| `gfn2-steps=` | Maximum optimization steps (0 = automatic) (0–10000) | integer |
| `gfn2-blend=` | Weight of the relative GFN2 energy added to the ranking score (0–1) | number |
| `gfn2-update=` | Replace pose coordinates by the optimized geometry (within gfn2-max-rmsd) | on\|off |
| `gfn2-restraint=` | Harmonic restraint keeping atoms near the docked pose (0 = none) (0–1 Hartree/Å²) | number |
| `gfn2-max-rmsd=` | Reject optimized geometries that moved more than this (0–10 Å) | number |
| `flex-auto=` | Soft flexibility of up to 6 pocket-lining sidechains chosen at launch (replaces manual 'flex add' residues) | on\|off |

**`dock config [section] [changed=on|off]`** — Print every docking setting by section; * marks values changed from the default

| Argument / option | Description | Type, range, default |
|---|---|---|
| `<section>` | Only this section | one of method\|budget\|grid\|operators\|local\|exploration\|torsions\|rerank\|strain\|fragment\|remc\|diffusion\|gfn2\|receptor, optional |
| `changed=` | Only settings that differ from the default | on\|off |

**`dock config reset [section]`** — Restore docking settings (all, or one section) to their defaults

| Argument / option | Description | Type, range, default |
|---|---|---|
| `<section>` | Only this section | one of method\|budget\|grid\|operators\|local\|exploration\|torsions\|rerank\|strain\|fragment\|remc\|diffusion\|gfn2\|receptor, optional |

**`dock status`** — Show whether a docking run (e.g. started from the Docking step) is in progress

**`dock stop`** — Stop a docking or batch docking run started from the Docking step

Examples:

```
dock
dock preset=thorough scoring=drusina
dock search=fragment runs=5
dock set population=200 generations=150 runs=4 grid-spacing=0.375
dock set search=remc remc-replicas=12 remc-tmax=6
dock set torsion-preset=redocking ls-steps=40 strain=off
dock set gfn2=on gfn2-top=10 gfn2-solvation=alpb
dock config; dock config fragment; dock config changed=on
dock config reset
dock batch ligands=prepared preset=fast
dock batch ligands=aspirin,ibuprofen
```

#### `results`

Ranked docking poses, per-pose score breakdown, clusters and batch ranking.

**`results [top=N] [sort=score|cluster|rmsd] [cluster=N]`** — Table of the best poses with score terms (ranks are those of 'results show' and 'pose show')

| Argument / option | Description | Type, range, default |
|---|---|---|
| `top=` | Number of rows (default 10) | integer |
| `sort=` | score (default), cluster = best pose of each cluster, rmsd = closest to the input pose | one of score\|cluster\|rmsd |
| `cluster=` | Only the poses of this cluster | integer |

**`results show <n>`** — Everything known about one pose: score terms, strain, OpenMM, GFN2-xTB, validity

| Argument / option | Description | Type, range, default |
|---|---|---|
| `<n>` | 1-based rank from 'results' | integer |

**`results clusters`** — Binding-mode clusters (2 Å RMSD leader clustering) with their best and mean scores

**`results batch [top=N]`** — Ranking of the last 'dock batch' run

| Argument / option | Description | Type, range, default |
|---|---|---|
| `top=` | Number of ligands (default 20) | integer |

Examples:

```
results
results top=25
results sort=cluster
results sort=rmsd top=5
results cluster=0
results show 1
results clusters
results batch top=50
```

#### `pose`

Show a docked pose in the 3D view.

**`pose show <n>`** — Display pose n with its interactions (GUI only; ignored in headless mode)

| Argument / option | Description | Type, range, default |
|---|---|---|
| `<n>` | 1-based rank from 'results' | integer |

Examples:

```
pose show 1
```

#### `flex`

Receptor flexibility: sidechains the docking search may rotate (induced fit).

**`flex add <residues…>`** — Make sidechains fully flexible (max 6; matched by residue number)

| Argument / option | Description | Type, range, default |
|---|---|---|
| `<residues>` | Residues as [chain:][name]number, e.g. TYR95 A:ASP189 120 | text, takes the rest of the line |

**`flex remove <residues…>`** — Make sidechains rigid again

| Argument / option | Description | Type, range, default |
|---|---|---|
| `<residues>` | Residues as [chain:][name]number, e.g. TYR95 A:ASP189 120 | text, takes the rest of the line |

**`flex auto [state]`** — Soft flexibility of the pocket-lining sidechains chosen at launch (replaces manual residues)

| Argument / option | Description | Type, range, default |
|---|---|---|
| `<state>` | on (default) or off | on\|off, optional |

**`flex list`** — Show the flexibility mode and the flexible residues

**`flex clear`** — Rigid receptor: remove every flexible residue and turn soft flexibility off

Examples:

```
flex add TYR95 ASP189
flex add A:120
flex remove TYR95
flex auto
flex auto off
flex list
flex clear
```

#### `constraint`

Pharmacophore constraints: interactions the docked pose must (hard) or should (soft) make.

**`constraint add <type> [option=value…]`** — Add a receptor-side constraint at a residue, an atom of it, or a point

| Argument / option | Description | Type, range, default |
|---|---|---|
| `<type>` | Interaction the ligand must make | one of donor\|acceptor\|salt-bridge\|pi-stack\|halogen\|metal\|hydrophobic |
| `residue=` | Target residue as [chain:][name]number, e.g. A:ASP189 | text |
| `atom=` | Atom name within the residue (e.g. OD1); implies scope=atom | text |
| `position=` | Target point in Å instead of a residue | x,y,z vector |
| `scope=` | residue = satisfied by any relevant atom of the residue (default without atom=) | one of atom\|residue |
| `radius=` | Distance within which the interaction counts, 1.5–10 Å (default 3.5) | number |
| `strength=` | soft = penalty that biases the search (default), hard = poses violating it are rejected | one of soft\|hard |
| `weight=` | Soft penalty in kcal/mol/Å², 0.1–100 (default 5) | number |

**`constraint list`** — List constraints with their state, target and strength

**`constraint enable <n>`** — Use a constraint in the next docking run

| Argument / option | Description | Type, range, default |
|---|---|---|
| `<n>` | 1-based index from 'constraint list' | integer |

**`constraint disable <n>`** — Keep a constraint but ignore it when docking

| Argument / option | Description | Type, range, default |
|---|---|---|
| `<n>` | 1-based index from 'constraint list' | integer |

**`constraint remove <n>`** — Delete a constraint

| Argument / option | Description | Type, range, default |
|---|---|---|
| `<n>` | 1-based index from 'constraint list' | integer |

**`constraint clear`** — Delete every constraint

Examples:

```
constraint add salt-bridge residue=ASP189 radius=4
constraint add acceptor residue=A:SER195 atom=OG strength=hard
constraint add hydrophobic position=12.1,4.5,-3.2 weight=10
constraint list
constraint disable 2
constraint remove 1
constraint clear
```

#### `refine`

Refine the current docking poses with OpenMM or GFN2-xTB.

**`refine openmm [top=N]`** — OpenMM pocket minimization of the top poses; stores the interaction energy (ranking unchanged)

| Argument / option | Description | Type, range, default |
|---|---|---|
| `top=` | Poses to refine (default 3) | integer |

**`refine gfn2 [option=value…]`** — GFN2-xTB optimization of the top poses with D4 dispersion and implicit solvation; re-ranks by the blended score

| Argument / option | Description | Type, range, default |
|---|---|---|
| `top=` | Poses to optimize (default: dock setting gfn2-top) | integer |
| `solvation=` | Implicit solvation (default: gfn2-solvation) | one of none\|alpb\|gbsa |
| `level=` | Convergence level (default: gfn2-level) | one of crude\|normal\|tight |
| `steps=` | Maximum optimization steps, 0 = automatic (default: gfn2-steps) | integer |
| `blend=` | Weight of the GFN2 energy in the ranking, 0–1 (default 0.3) | number |
| `restraint=` | Positional restraint in Hartree/Å², 0 = none (default: gfn2-restraint) | number |
| `max-rmsd=` | Keep the docked geometry when optimization moves it more, Å (default: gfn2-max-rmsd) | number |
| `update=` | Use the optimized coordinates (default: gfn2-update) | on\|off |

Examples:

```
refine openmm
refine openmm top=10
refine gfn2
refine gfn2 top=5 solvation=gbsa level=tight
refine gfn2 blend=0 update=off
```

#### `rescore`

Rescore the current poses with a neural model without moving them.

**`rescore <model>`** — Score every pose with DruseAF (pKd) or PIGNet2 (kcal/mol); that model becomes the active scoring

| Argument / option | Description | Type, range, default |
|---|---|---|
| `<model>` | druseaf = DruseAF v4 affinity, pignet2 = physics-informed GNN | one of druseaf\|pignet2 |

Examples:

```
rescore druseaf
rescore pignet2
```

#### `strain`

MMFF94 ligand strain of the current poses, with an optional score penalty.

**`strain [top=N] [threshold=x] [weight=x]`** — Compute docked − relaxed MMFF94 energy; poses above the threshold get weight × excess added to their score

| Argument / option | Description | Type, range, default |
|---|---|---|
| `top=` | Poses to evaluate (default 50) | integer |
| `threshold=` | Tolerated strain in kcal/mol (default: dock setting strain-threshold, 6) | number |
| `weight=` | Penalty weight, 0 = report only (default: strain-weight, 0.5) | number |

Examples:

```
strain
strain top=20 threshold=4
strain weight=0
```

#### `screen`

Virtual screening of a SMILES or SDF library against the selected pocket.

**`screen <library> [option=value…]`** — Prepare (RDKit 3D), prescreen, GA-dock a shortlist, refine and rank a library

| Argument / option | Description | Type, range, default |
|---|---|---|
| `<library>` | Library file (.smi or .sdf) | path |
| `max=` | Screen at most this many molecules (default 100000) | integer |
| `max-rotatable=` | Skip molecules with more rotatable bonds (default 15) | integer |
| `grid-spacing=` | Grid spacing in Å (default 0.5, coarser than 'dock') | number |
| `prescreen-poses=` | Rigid poses per ligand in the batched Metal prescreen (default 72, min 16) | integer |
| `shortlist=` | Fraction of prescreened ligands docked with the GA (default 0.35) | number |
| `shortlist-min=` | Always GA-dock at least this many ligands (default 128) | integer |
| `shortlist-max=` | Never GA-dock more than this many ligands (default 4096) | integer |
| `population=` | GA population per ligand (default 50, min 20) | integer |
| `generations=` | GA generations per ligand (default 30, min 10) | integer |
| `strain=` | MMFF94 strain penalty on hits (default off) | on\|off |
| `openmm-top=` | OpenMM-refine the top N hits, 0 = off (default 25) | integer |
| `gfn2-top=` | GFN2-xTB single-point energy of the top N hits, 0 = off (default 0) | integer |
| `admet=` | Predict ADMET properties of the top hits when the models are installed (default on) | on\|off |
| `top=` | Hits to print (default 20) | integer |

**`screen hits [top=N]`** — Print the hits of the last screening run

| Argument / option | Description | Type, range, default |
|---|---|---|
| `top=` | Hits to print (default 20) | integer |

**`screen stop`** — Stop a screening run in progress

Examples:

```
screen enamine_subset.smi top=50
screen library.sdf max=5000 shortlist=0.2 openmm-top=50
screen hits.smi population=80 generations=60 gfn2-top=10 admet=off
screen hits top=100
```

### Analysis

#### `interactions`

Protein–ligand interactions of a pose: type, residue, atoms and distance.

**`interactions [pose=N] [ligand=on|off] [type=hbond|hydrophobic|salt-bridge|pi-stack|pi-cation|halogen|metal|ch-pi|amide-pi|chalcogen] [all=on|off]`** — List the interactions of a pose (or the loaded ligand), grouped by type

| Argument / option | Description | Type, range, default |
|---|---|---|
| `pose=` | Docked pose to analyse, 1-based rank from 'results' (default 1) | integer |
| `ligand=` | Analyse the loaded ligand where it stands (e.g. the crystal ligand) instead of a docked pose | on\|off |
| `type=` | Only this interaction type | one of hbond\|hydrophobic\|salt-bridge\|pi-stack\|pi-cation\|halogen\|metal\|ch-pi\|amide-pi\|chalcogen |
| `all=` | Report every interaction type, not only those the active scoring function accounts for (default off) | on\|off |

Examples:

```
interactions
interactions pose=3
interactions type=hbond
interactions ligand=on all=on
```

#### `contacts`

Residues in contact with a pose: closest distance and number of atom pairs.

**`contacts [pose=N] [ligand=on|off] [cutoff=x]`** — Protein residues with a heavy atom within the cutoff of the ligand's heavy atoms

| Argument / option | Description | Type, range, default |
|---|---|---|
| `pose=` | Docked pose to analyse, 1-based rank from 'results' (default 1) | integer |
| `ligand=` | Analyse the loaded ligand where it stands (e.g. the crystal ligand) instead of a docked pose | on\|off |
| `cutoff=` | Contact distance in Å (default 4) | number |

Examples:

```
contacts
contacts pose=2 cutoff=4.5
contacts ligand=on cutoff=3.5
```

#### `rmsd`

Heavy-atom RMSD between poses, or of each pose against the input ligand.

**`rmsd <a> <b>`** — RMSD between two docked poses

| Argument / option | Description | Type, range, default |
|---|---|---|
| `<a>` | First pose (1-based rank) | integer |
| `<b>` | Second pose (1-based rank) | integer |

**`rmsd matrix [top=N]`** — Pairwise RMSD matrix of the best poses

| Argument / option | Description | Type, range, default |
|---|---|---|
| `top=` | Number of best-ranked poses (default 5) | integer |

**`rmsd reference [top=N]`** — RMSD of each pose against the input ligand coordinates (redocking RMSD for a crystal ligand)

| Argument / option | Description | Type, range, default |
|---|---|---|
| `top=` | Number of best-ranked poses (default 10) | integer |

Examples:

```
rmsd 1 2
rmsd matrix top=3
rmsd reference top=5
```

#### `fingerprint`

Residue interaction fingerprints of the best poses.

**`fingerprint [top=N]`** — Residue × pose table of interaction types, with how often each residue is contacted

| Argument / option | Description | Type, range, default |
|---|---|---|
| `top=` | Number of best-ranked poses (default 5) | integer |

Examples:

```
fingerprint
fingerprint top=10
```

#### `validity`

PoseBusters-style physical validity checks of poses.

**`validity [pose=N] [ligand=on|off] [top=N]`** — Bond lengths and angles, internal and protein clashes, connectivity and extent of one pose or the best poses

| Argument / option | Description | Type, range, default |
|---|---|---|
| `pose=` | Docked pose to analyse, 1-based rank from 'results' (default 1) | integer |
| `ligand=` | Analyse the loaded ligand where it stands (e.g. the crystal ligand) instead of a docked pose | on\|off |
| `top=` | Number of best-ranked poses (default 10) | integer |

Examples:

```
validity
validity pose=1
validity top=20
validity ligand=on
```

### Export

#### `export`

Write poses, results, structures, interactions, analogs and logs to files.

**`export poses <file> [top=N] [cluster-representatives=on|off] [protein=on|off]`** — Ranked docked poses: SDF with every score term, or multi-model PDB (default top 50)

| Argument / option | Description | Type, range, default |
|---|---|---|
| `<file>` | Output file (.sdf or .pdb) | path |
| `top=` | Write only the N best poses (after cluster filtering) | integer |
| `cluster-representatives=` | Write only the best pose of each cluster (default off) | on\|off |
| `protein=` | PDB only: write the protein once, poses as residues of chain X (default off) | on\|off |

**`export results <file> [top=N] [cluster-representatives=on|off]`** — Results table as CSV: score, every term incl. intra-and-other, cluster, validity, refinement

| Argument / option | Description | Type, range, default |
|---|---|---|
| `<file>` | Output file (.csv) | path |
| `top=` | Write only the N best poses (after cluster filtering) | integer |
| `cluster-representatives=` | Write only the best pose of each cluster (default off) | on\|off |

**`export pose <n> <file>`** — One docked ligand pose without the protein (.sdf with scores, or .pdb)

| Argument / option | Description | Type, range, default |
|---|---|---|
| `<n>` | 1-based pose rank from 'results' | integer |
| `<file>` | Output file (.sdf or .pdb) | path |

**`export complex <n> <file>`** — One docked pose together with the protein as a single PDB

| Argument / option | Description | Type, range, default |
|---|---|---|
| `<n>` | 1-based pose rank from 'results' | integer |
| `<file>` | Output file (.pdb) | path |

**`export protein <file> [hydrogens=on|off] [ligand=on|off]`** — The current (prepared) protein as PDB

| Argument / option | Description | Type, range, default |
|---|---|---|
| `<file>` | Output file (.pdb) | path |
| `hydrogens=` | Include hydrogen atoms (default on) | on\|off |
| `ligand=` | Append the active ligand at its current coordinates as HETATM (default off) | on\|off |

**`export ligand <file>`** — The active ligand at its current coordinates (.sdf/.mol with all atoms, or .pdb)

| Argument / option | Description | Type, range, default |
|---|---|---|
| `<file>` | Output file (.sdf, .mol or .pdb) | path |

**`export ligands <file> [format=auto|sdf|csv|smi|json] [subset=all|prepared|dockable|forms] [conformers=best|all]`** — The Ligand Database — same as 'ligand db export'

| Argument / option | Description | Type, range, default |
|---|---|---|
| `<file>` | Output file; the format follows the extension | path |
| `format=` | Output format (auto: from the extension) | one of auto\|sdf\|csv\|smi\|json, default auto |
| `subset=` | Entries to write (dockable: what 'dock batch' docks) | one of all\|prepared\|dockable\|forms, default all |
| `conformers=` | SDF: lowest-energy conformer only, or one record per conformer | one of best\|all, default best |

**`export interactions <file> [pose=N] [types=all|scored]`** — Protein–ligand interactions of one pose as CSV (type, distance, atoms, residue)

| Argument / option | Description | Type, range, default |
|---|---|---|
| `<file>` | Output file (.csv) | path |
| `pose=` | 1-based pose rank (default 1) | integer |
| `types=` | all detected contact types, or only those the scoring function accounts for (default all) | one of all\|scored |

**`export analogs <file> [status=all|passed|docked]`** — Lead-optimization analogs: CSV (properties, deltas, ADMET, score) or SDF (3D, docked pose when available)

| Argument / option | Description | Type, range, default |
|---|---|---|
| `<file>` | Output file (.csv or .sdf) | path |
| `status=` | all, passed (not filtered or failed) or docked analogs only (default all) | one of all\|passed\|docked |

**`export hits <file> [top=N]`** — Virtual-screening hits from 'screen': CSV table or SDF with best poses

| Argument / option | Description | Type, range, default |
|---|---|---|
| `<file>` | Output file (.csv or .sdf) | path |
| `top=` | SDF only: the N best hits (default 100) | integer |

**`export batch <file>`** — Batch-docking ranking from 'dock batch' as CSV (best pose terms per ligand)

| Argument / option | Description | Type, range, default |
|---|---|---|
| `<file>` | Output file (.csv) | path |

**`export session-log <file>`** — This session's command history and activity log as text

| Argument / option | Description | Type, range, default |
|---|---|---|
| `<file>` | Output file (.txt or .log) | path |

Examples:

```
export poses out/poses.sdf top=10
export poses out/representatives.pdb cluster-representatives=on
export results out/results.csv
export complex 1 out/best_complex.pdb
export pose 3 out/pose3.sdf
export protein out/prepared.pdb hydrogens=off
export ligands out/library.sdf subset=prepared
export interactions out/contacts.csv pose=1
export analogs out/analogs.csv status=docked
export session-log out/session.log
```

### Lead optimization

#### `leadopt`

Lead optimization: pick a reference, set generation and ADMET options, generate, dock and hand off analogs.

**`leadopt reference [pose=N] [ligand=text]`** — Set the optimization reference: a docked pose (default best; the active ligand when nothing is docked) or a ligand by name

| Argument / option | Description | Type, range, default |
|---|---|---|
| `pose=` | 1-based pose rank from 'results' (default 1) | integer |
| `ligand=` | Active ligand or Ligand Database entry name, used without a pose score | text |

**`leadopt set [option=value…]`** — Change generation parameters, property directions and ADMET filters (kept for later runs)

| Argument / option | Description | Type, range, default |
|---|---|---|
| `count=` | Maximum number of analogs to generate (1–500, default 50) | integer |
| `similarity=` | Minimum Tanimoto similarity to the reference (0–1, default 0.7) | number |
| `keep-scaffold=` | Edit R-groups only, never ring atoms (default on) | on\|off |
| `polarity=` | Polarity direction: -1 less polar … +1 more polar (default 0) | number |
| `rigidity=` | Rigidity direction: -1 more flexible … +1 more rigid (default 0) | number |
| `lipophilicity=` | Lipophilicity direction: -1 hydrophilic … +1 lipophilic (default 0) | number |
| `size=` | Size direction: -1 smaller … +1 larger (default 0) | number |
| `lipinski=` | Reject analogs violating Lipinski's rule of five (default off) | on\|off |
| `veber=` | Reject analogs violating Veber's rules (default off) | on\|off |
| `herg=` | Reject analogs with predicted hERG liability > 0.5 (default off) | on\|off |
| `cyp=` | Reject analogs with predicted CYP2D6 or CYP3A4 inhibition > 0.5 (default off) | on\|off |
| `max-logp=` | Reject analogs with logP above this value (default 7) | number |

**`leadopt config`** — Show the reference, generation parameters and ADMET filters

**`leadopt defaults`** — Restore every generation parameter and filter to its default

**`leadopt generate [option=value…]`** — Generate property-directed analogs (options as in 'leadopt set'); passing analogs go to the Ligand Database

| Argument / option | Description | Type, range, default |
|---|---|---|
| `count=` | Maximum number of analogs to generate (1–500, default 50) | integer |
| `similarity=` | Minimum Tanimoto similarity to the reference (0–1, default 0.7) | number |
| `keep-scaffold=` | Edit R-groups only, never ring atoms (default on) | on\|off |
| `polarity=` | Polarity direction: -1 less polar … +1 more polar (default 0) | number |
| `rigidity=` | Rigidity direction: -1 more flexible … +1 more rigid (default 0) | number |
| `lipophilicity=` | Lipophilicity direction: -1 hydrophilic … +1 lipophilic (default 0) | number |
| `size=` | Size direction: -1 smaller … +1 larger (default 0) | number |
| `lipinski=` | Reject analogs violating Lipinski's rule of five (default off) | on\|off |
| `veber=` | Reject analogs violating Veber's rules (default off) | on\|off |
| `herg=` | Reject analogs with predicted hERG liability > 0.5 (default off) | on\|off |
| `cyp=` | Reject analogs with predicted CYP2D6 or CYP3A4 inhibition > 0.5 (default off) | on\|off |
| `max-logp=` | Reject analogs with logP above this value (default 7) | number |

**`leadopt list [status=all|new|3d|docked|filtered|failed] [top=N]`** — List analogs with status, score and property deltas vs the reference

| Argument / option | Description | Type, range, default |
|---|---|---|
| `status=` | Only analogs with this status (default all) | one of all\|new\|3d\|docked\|filtered\|failed |
| `top=` | Show at most N rows | integer |

**`leadopt show <n>`** — Details of one analog: SMILES, descriptors and deltas, ADMET predictions, docking score

| Argument / option | Description | Type, range, default |
|---|---|---|
| `<n>` | 1-based row from 'leadopt list' | integer |

**`leadopt dock`** — Mini-dock the new analogs with the reference's scoring function and pocket (reduced budget)

**`leadopt stop`** — Stop a running analog generation or analog docking

**`leadopt send database [filtered=on|off]`** — Add the analogs to the Ligand Database (duplicates by SMILES are skipped)

| Argument / option | Description | Type, range, default |
|---|---|---|
| `filtered=` | Also add analogs rejected by the ADMET filters (default off) | on\|off |

**`leadopt send docking`** — Add the analogs to the Ligand Database and open the Docking step for a batch run

**`leadopt clear`** — Remove all generated analogs (the reference and settings are kept)

Examples:

```
leadopt reference pose=1
leadopt reference ligand=imatinib
leadopt set count=25 similarity=0.6 polarity=0.5 herg=on max-logp=5
leadopt generate count=30 keep-scaffold=off
leadopt list status=docked top=10
leadopt show 1
leadopt dock
leadopt send database
```

### View

#### `hide`

Hide atoms in the 3D view (GUI only).

**`hide selection`** — Hide the selected atoms and clear the selection (GUI only; ignored in headless mode)

**`hide chain <id>`** — Hide a whole chain (GUI only; ignored in headless mode)

| Argument / option | Description | Type, range, default |
|---|---|---|
| `<id>` | Chain ID (case-sensitive) | text |

Examples:

```
select chain B; hide selection
hide chain B
```

#### `show`

Show hidden atoms in the 3D view (GUI only).

**`show all`** — Show every hidden atom and chain (GUI only; ignored in headless mode)

**`show chain <id>`** — Show a hidden chain (GUI only; ignored in headless mode)

| Argument / option | Description | Type, range, default |
|---|---|---|
| `<id>` | Chain ID (case-sensitive) | text |

Examples:

```
show all
show chain B
```

#### `center`

Point the camera at the selection, the ligand or the protein (GUI only).

**`center selection`** — Centre the view on the selected atoms (GUI only; ignored in headless mode)

**`center ligand`** — Fit the view to the ligand (GUI only; ignored in headless mode)

**`center protein`** — Fit the view to the protein (GUI only; ignored in headless mode)

Examples:

```
select within 5 of ligand; center selection
center ligand
```

#### `view`

Change the 3D view: render modes, colours, surface, clipping, camera, snapshots.

**`view [option=value…]`** — Set several display options at once

| Argument / option | Description | Type, range, default |
|---|---|---|
| `mode=` | Render mode of the protein (and the ligand unless ligand-mode is set) | one of ballstick\|space\|wire\|ribbon |
| `ligand-mode=` | Render mode of the ligand; auto follows the protein (ball-and-stick under ribbon) | one of ballstick\|space\|wire\|auto |
| `hydrogens=` | Show hydrogen atoms | on\|off |
| `lighting=` | Uniform or directional lighting | one of uniform\|directional |
| `scheme=` | Colour scheme: element (CPK), ligand-focus (grey protein, gold ligand) or chain | one of element\|ligand-focus\|chain |
| `surface=` | Show the molecular surface | on\|off |
| `side-chains=` | Side chains drawn in ribbon mode | one of none\|interacting\|selected\|all |

**`view protein [option=value…]`** — Protein display: visibility, render mode, colouring, side chains

| Argument / option | Description | Type, range, default |
|---|---|---|
| `visible=` | Show the protein | on\|off |
| `mode=` | Render mode (also resets ligand-mode to auto) | one of ballstick\|space\|wire\|ribbon |
| `color=` | element, chain, or a colour (name, #rrggbb or r,g,b) for every chain | text |
| `scope=` | Colour carbons only or every atom (default carbons) | one of carbons\|all |
| `side-chains=` | Side chains drawn in ribbon mode | one of none\|interacting\|selected\|all |

**`view ligand [visible=on|off] [mode=ballstick|space|wire|auto] [carbon-color=text] [carbons-only=on|off]`** — Ligand display: visibility, render mode, carbon colour

| Argument / option | Description | Type, range, default |
|---|---|---|
| `visible=` | Show the ligand | on\|off |
| `mode=` | Render mode; auto follows the protein | one of ballstick\|space\|wire\|auto |
| `carbon-color=` | Ligand carbon colour (name, #rrggbb or r,g,b), or default | text |
| `carbons-only=` | Apply carbon-color to carbons only (on) or to every atom (off) | on\|off |

**`view chain <id> [visible=on|off] [color=text] [scope=carbons|all]`** — One chain: visibility and colour

| Argument / option | Description | Type, range, default |
|---|---|---|
| `<id>` | Chain identifier, e.g. A | text |
| `visible=` | Show the chain | on\|off |
| `color=` | cpk, chain (palette colour) or a colour (name, #rrggbb or r,g,b) | text |
| `scope=` | Colour carbons only or every atom | one of carbons\|all |

**`view surface [show] [color=uniform|esp|hydrophobicity|pharmacophore] [opacity=x] [probe=x] [spacing=x]`** — Molecular surface: show/hide, colouring, opacity, probe and grid

| Argument / option | Description | Type, range, default |
|---|---|---|
| `<show>` | on or off (default: leave as is) | on\|off, optional |
| `color=` | Surface colouring | one of uniform\|esp\|hydrophobicity\|pharmacophore |
| `opacity=` | Surface opacity (0–1, default 0.85) | number |
| `probe=` | Probe radius in Å (0.5–5, default 1.4 = water) | number |
| `spacing=` | Grid spacing in Å (0.2–2, default 0.5; finer is slower) | number |

**`view background <opacity>`** — Background gradient opacity (0 = plain white or black, 1 = full gradient)

| Argument / option | Description | Type, range, default |
|---|---|---|
| `<opacity>` | 0–1 | number |

**`view clip [enabled] [thickness=x] [offset=x]`** — Z-slab clipping around the view centre (or the focused pocket)

| Argument / option | Description | Type, range, default |
|---|---|---|
| `<enabled>` | on or off (default: leave as is) | on\|off, optional |
| `thickness=` | Slab thickness in Å (2–40) | number |
| `offset=` | Slab offset along the view axis in Å (-20–20) | number |

**`view grid [show] [width=x] [color=text]`** — Docking grid box wireframe: visibility, line width, colour (GUI only; ignored in headless mode)

| Argument / option | Description | Type, range, default |
|---|---|---|
| `<show>` | on or off (default on) | on\|off, optional |
| `width=` | Line width in screen pixels (0.5–10) | number |
| `color=` | Line colour (name, #rrggbb or r,g,b) | text |

**`view interactions <show>`** — Protein–ligand interaction lines of the shown pose (GUI only; ignored in headless mode)

| Argument / option | Description | Type, range, default |
|---|---|---|
| `<show>` | on or off | on\|off |

**`view fit [target]`** — Fit the camera to part of the scene (GUI only; ignored in headless mode)

| Argument / option | Description | Type, range, default |
|---|---|---|
| `<target>` | What to frame (default all) | one of all\|protein\|ligand\|pocket\|selection, optional |

**`view reset`** — Reset the camera orientation and refit the scene (GUI only; ignored in headless mode)

**`view rotate [x=x] [y=x] [z=x]`** — Rotate the camera around the view centre by angles in degrees (GUI only; ignored in headless mode)

| Argument / option | Description | Type, range, default |
|---|---|---|
| `x=` | Degrees around the screen's horizontal axis (tilt) | number |
| `y=` | Degrees around the vertical axis (turn) | number |
| `z=` | Degrees around the viewing axis (roll) | number |

**`view zoom <factor>`** — Zoom the camera by a factor (2 = twice as close, 0.5 = twice as far) (GUI only; ignored in headless mode)

| Argument / option | Description | Type, range, default |
|---|---|---|
| `<factor>` | Zoom factor (0.05–20) | number |

**`view snapshot <file> [width=N] [height=N] [fit=all|protein|ligand|pocket|selection] [theme=dark|light]`** — Render the 3D view offscreen to an image (.png, .jpg or .tiff); also works headless

| Argument / option | Description | Type, range, default |
|---|---|---|
| `<file>` | Output image file | path |
| `width=` | Width in pixels (default: the viewport, or 1600) | integer |
| `height=` | Height in pixels (default: the viewport, or 1200) | integer |
| `fit=` | Frame this part of the scene first (headless default all) | one of all\|protein\|ligand\|pocket\|selection |
| `theme=` | Background theme (default: the app's) | one of dark\|light |

**`view status`** — Show the current display settings

Examples:

```
view mode=ribbon ligand-mode=ballstick side-chains=interacting
view scheme=ligand-focus hydrogens=off
view ligand carbon-color=gold
view chain B visible=off
view surface on color=esp opacity=0.6
view clip on thickness=12
view fit pocket; view rotate y=90
view snapshot figures/pose1.png width=2400 height=1600
```

### Settings

#### `settings`

Show or change software settings (CPU cores, logging, console, folders).

**`settings`** — Show every setting, its default, the session flags and the machine

**`settings set [option=value…]`** — Change one or more settings; they are saved and shared with the GUI and the terminal

| Argument / option | Description | Type, range, default |
|---|---|---|
| `cores=` | CPU cores Druse may use at once; applied immediately | integer 1–8 |
| `log-level=` | Default activity-log verbosity on stderr for headless runs | one of debug\|info\|warn\|error\|none |
| `history=` | Console commands remembered | integer 10–10000 |
| `autosuggest=` | Inline completion suggestions in the app's console | on\|off |
| `pdb-cache=` | Folder for downloaded PDB entries (kept 7 days) | path |
| `project-dir=` | Folder the Save Project panel starts in | path |

**`settings reset [setting]`** — Restore one setting, or all of them, to the default

| Argument / option | Description | Type, range, default |
|---|---|---|
| `<setting>` | Setting to reset (all when left out) | one of cores\|log-level\|history\|autosuggest\|pdb-cache\|project-dir, optional |

Examples:

```
settings
settings set cores=4
settings set log-level=info history=1000
settings set pdb-cache=~/Data/pdb project-dir=~/Projects/druse
settings reset cores
settings reset
set onerror=continue      # session flag: keep running a script after a failing statement
```

#### `cli`

Install the 'druse' terminal command and its shell completions.

**`cli status`** — Show where 'druse' and the completion scripts are installed

**`cli install [location=user|system]`** — Install the 'druse' command (runs this Druse.app headless)

| Argument / option | Description | Type, range, default |
|---|---|---|
| `location=` | user: ~/.local/bin (no password); system: /usr/local/bin (administrator prompt, app only) | one of user\|system, default user |

**`cli uninstall [location=user|system]`** — Remove the 'druse' command Druse installed (other files are left alone)

| Argument / option | Description | Type, range, default |
|---|---|---|
| `location=` | user: ~/.local/bin (no password); system: /usr/local/bin (administrator prompt, app only) | one of user\|system, default user |

**`cli completions print <shell>`** — Print the completion script for a shell

| Argument / option | Description | Type, range, default |
|---|---|---|
| `<shell>` | fish, zsh or bash | one of fish\|zsh\|bash |

**`cli completions install [shell=fish|zsh|bash|all]`** — Write the completion script to the shell's standard per-user folder

| Argument / option | Description | Type, range, default |
|---|---|---|
| `shell=` | fish, zsh, bash or all | one of fish\|zsh\|bash\|all, default your login shell |

**`cli completions uninstall [shell=fish|zsh|bash|all]`** — Remove installed completion scripts

| Argument / option | Description | Type, range, default |
|---|---|---|
| `shell=` | fish, zsh, bash or all | one of fish\|zsh\|bash\|all, default your login shell |

Examples:

```
cli status
cli install
cli install location=system
cli completions install
cli completions install shell=all
cli completions print zsh
cli uninstall
```

