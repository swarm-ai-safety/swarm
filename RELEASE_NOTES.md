## Post-Release Additions

### New Features
1. **CrewAI Integration**:
   - Added CrewAI adapter allowing CrewAI crews to serve as the decision-making policies within SWARM agents.
   - Added YAML-level configurations for CrewAI agents, introduced agent profiles (market_team_v1 and others), and integrated logging enhancements.
   - Included an example integration scenario and 74+ rigorous tests to validate the feature.

2. **Agent API**:
   - Introduced endpoints for managing simulation runs, posting metric results, and voting on posts:
     - Example endpoints: `POST /api/runs`, `GET /api/runs/:id`, and `POST /api/posts`.
   - Integrated SQLite-backed persistence for runs and posts, added simulation comparison metrics, and downloadable artifacts.
   - Thorough testing implemented, with over 74 new tests covering API workflows and validations.

### Bug Fixes
- **CrewAI Security and Performance**:
   - Strengthened thread handling with proper timeout mechanisms during Crew integrations.
   - Applied validation strategies for metadata sanitation, enforced buffer limits for in-memory structures, and prevented threading leaks.

- **Agent API Security Enhancements**:
   - Implemented strong defenses against SSRF vulnerabilities, quota overrides, and race conditions on writes.
   - Further improved components like error sanitization and API token hashing optimization.

This builds on the v1.5.0 foundation by adding high-value integrations and safeguarding critical functionalities through robust testing and practices.

> Updated on 2026-02-20 22:31:39 by rsavitt