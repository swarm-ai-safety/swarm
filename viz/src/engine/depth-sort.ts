import type { RenderEntity } from "./types";

/** Sort entities back-to-front using painter's algorithm.
 *  depth = gridX + gridY; lower values are further back and drawn first. */
export function depthSort(entities: RenderEntity[]): RenderEntity[] {
  return entities.slice().sort((a, b) => a.depth - b.depth);
}
