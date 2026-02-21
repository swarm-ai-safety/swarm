# /build_game

Type-check, build, and deploy-build the viz game in one step.

## Usage

`/build_game`

## Behavior

1. **Type-check**:
   - Run `cd viz && npx tsc --noEmit`
   - If there are type errors, report them and stop.

2. **Dev build**:
   - Run `npm run build` (from `viz/`)
   - If the build fails, report errors and stop.

3. **Deploy build**:
   - Run `npm run build:deploy` (static export with basePath to `docs/game-app/`)
   - If the build fails, report errors and stop.

4. **Report**:
   ```
   Game build complete
     Types:    clean
     Build:    success
     Deploy:   success (docs/game-app/)
     Bundle:   <first load JS size from build output>
   ```

## Constraints

- Always run all three steps in order. Do not skip the type-check.
- If any step fails, stop and report the error. Do not continue to the next step.
- Do not commit or push — use `/ship` for that.
