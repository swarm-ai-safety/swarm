### 1.6.0 - Released 2026-02-20

#### Added
- **Sprite-Based Movement Animations**:
  - Replaced procedural motion rendering with dynamic sprite animations, supporting idle poses and walking for all agent types (e.g., "honest," "adversarial").
  - Integrated `SpriteRegistry` for efficient sprite loading, background cleanup, and transparency adjustments.

- **Environmental Sprites**:
  - Added dynamic sprites for environment objects (e.g., tiles, towers) to enhance the isometric scene aesthetics.

#### Changed
- Animations for agent movements and interactions are now powered by sprites, synchronized with motion parameters such as `walkRate` and `bobAmplitude`.
- Updated rendering pipeline to dynamically scale and position agent sprites on the canvas.