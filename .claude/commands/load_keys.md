# /load_keys

Load API keys from `~/.zshrc` (or `~/.bashrc`) into the current shell environment. Eliminates the need to manually grep and export keys each session.

## Usage

`/load_keys [provider]`

Examples:
- `/load_keys` (load all available keys)
- `/load_keys openrouter` (load only OpenRouter key)

## Behavior

1. **Detect shell config file**: Check for `~/.zshrc` first, then `~/.bashrc`, then `~/.bash_profile`.

2. **Extract keys**: For each known provider, grep the shell config for the key variable and export it. The known key variables are:

   | Variable | Provider |
   |---|---|
   | `OPENROUTER_API_KEY` | OpenRouter |
   | `ANTHROPIC_API_KEY` | Anthropic |
   | `OPENAI_API_KEY` | OpenAI |
   | `GROQ_API_KEY` | Groq |
   | `TOGETHER_API_KEY` | Together |
   | `DEEPSEEK_API_KEY` | DeepSeek |
   | `GOOGLE_API_KEY` | Google |

   Extraction command per key (handles `export VAR="val"`, `export VAR='val'`, `export VAR=val`):
   ```bash
   grep '^export KEY_NAME=' ~/.zshrc | tail -1 | sed 's/^export //' | cut -d= -f2- | tr -d '"' | tr -d "'"
   ```

3. **Export into current environment**: For each key found, run:
   ```bash
   export KEY_NAME="<extracted_value>"
   ```

4. **Report**: Print which keys were loaded and their character count (NOT the actual values):
   ```
   API Keys Loaded
     OPENROUTER_API_KEY:  set (73 chars)
     ANTHROPIC_API_KEY:   not found
     OPENAI_API_KEY:      not found
     GROQ_API_KEY:        not found
     TOGETHER_API_KEY:    not found
     DEEPSEEK_API_KEY:    not found
     GOOGLE_API_KEY:      not found
   ```

5. **If a `[provider]` argument is given**, only load that provider's key. Map provider names to variables:
   - `openrouter` → `OPENROUTER_API_KEY`
   - `anthropic` → `ANTHROPIC_API_KEY`
   - `openai` → `OPENAI_API_KEY`
   - `groq` → `GROQ_API_KEY`
   - `together` → `TOGETHER_API_KEY`
   - `deepseek` → `DEEPSEEK_API_KEY`
   - `google` → `GOOGLE_API_KEY`

## Constraints

- Never print the actual key values — only confirm presence and length.
- If the shell config file doesn't exist, say so and stop.
- If a key is already set in the environment AND matches what's in the config file, skip it and note "already set".
- If a key is already set but DIFFERS from the config file, load the config file version and note "updated from ~/.zshrc".
