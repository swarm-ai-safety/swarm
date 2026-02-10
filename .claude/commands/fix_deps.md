Scan the project for dependency issues and report findings.

## Steps

1. **Find all `package.json` files** (excluding `node_modules/`):
   - Check each for `"file:.."` or `"file:../.."`-style dependencies — these cause npm to recursively copy the parent into `node_modules`, creating symlink-loop-like bloat. Suggest using npm workspaces or `"link:.."` protocol instead.

2. **Check for deeply nested `node_modules/`** (more than 5 levels deep):
   - This is a strong indicator of a recursive copy loop from `file:` dependencies.
   - Report the deepest path found and suggest cleanup (`rm -rf node_modules && npm install` after fixing `package.json`).

3. **Check `requirements.txt` files** for pinned `git+https://` URLs:
   - These can break when repos go private, get renamed, or tags are deleted.
   - Suggest pinning to a release on PyPI or using a hash pin.

4. **Check `pyproject.toml`** for `file:` or `path:` dev dependencies that won't resolve in CI.

5. **Report a summary** with:
   - Total issues found (0 = clean)
   - Each issue with file path, line number, and suggested fix
   - If no issues found, confirm the dependency tree looks healthy.
