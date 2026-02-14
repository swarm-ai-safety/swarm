import React from "react";
import { Composition } from "remotion";
import { SwarmPromo } from "./SwarmPromo";

export const RemotionRoot: React.FC = () => {
  return (
    <Composition
      id="SwarmPromo"
      component={SwarmPromo}
      durationInFrames={1170}
      fps={30}
      width={1920}
      height={1080}
    />
  );
};
