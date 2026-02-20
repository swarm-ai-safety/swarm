# /obsidian

Open a file or vault in Obsidian from the CLI, with auto-detected vault name.

## Usage

`/obsidian [file-path]`

Examples:
- `/obsidian` (opens the vault root)
- `/obsidian vault/dashboards/suggestions` (opens a specific note)
- `/obsidian vault/claims/claim-tax-welfare-tradeoff` (opens a claim card)

## Behavior

### 1. Find the Obsidian vault registry

```bash
cat ~/Library/Application\ Support/obsidian/obsidian.json
```

Parse the JSON to get the list of registered vaults. Each vault has a `path` and `open` status.

### 2. Match the vault

- If the current working directory (or a parent) matches a registered vault path, use that vault.
- If working in `swarm-artifacts/`, match the vault registered at `/Users/raelisavitt/swarm-artifacts` (or wherever it lives).
- If no match is found, report: "No Obsidian vault registered for this directory. Open Obsidian and add it as a vault first."

Extract the vault **directory name** from the path (e.g. `/Users/raelisavitt/swarm-artifacts` → `swarm-artifacts`). This is the vault name Obsidian uses in URLs.

### 3. Build the URL

If a file path argument was provided:
- Resolve it relative to the vault root
- Strip `.md` extension if present
- URL-encode the path (replace `/` with `%2F`, spaces with `%20`)
- Build: `obsidian://open?vault=<vault-name>&file=<encoded-path>`

If no file path:
- Build: `obsidian://open?vault=<vault-name>`

### 4. Open

```bash
open "<url>"
```

If the `open` command fails with "no application knows how to open", report: "Obsidian is not installed. Install from https://obsidian.md or `brew install --cask obsidian`."

### 5. Report

```
Opened in Obsidian:
  Vault: <vault-name>
  File:  <file-path> (or "vault root")
```

## Edge Cases

- If the user provides a full absolute path (e.g. `/Users/.../vault/claims/foo.md`), convert it to a vault-relative path.
- If the file doesn't exist, warn but still attempt to open (Obsidian will create a new note).
- If multiple vaults match, prefer the one marked `"open": true`.

## Constraints

- Never modify Obsidian configuration files.
- Never install Obsidian automatically — just report how to install if missing.
- The vault name in the URL must match exactly what Obsidian registered (usually the directory name).
