import React from "react";
import { Composition } from "remotion";
import { SwarmPromo } from "./SwarmPromo";
import { HeroImage } from "./HeroImage";
import { CollusionCascade } from "./scenes/CollusionCascade";
import { GovernanceLag } from "./scenes/GovernanceLag";
import { DelegationGames } from "./scenes/DelegationGames";

export const RemotionRoot: React.FC = () => {
  return (
    <>
      <Composition
        id="SwarmPromo"
        component={SwarmPromo}
        durationInFrames={1170}
        fps={30}
        width={1920}
        height={1080}
      />
      <Composition
        id="HeroImage"
        component={HeroImage}
        durationInFrames={150}
        fps={30}
        width={1200}
        height={630}
      />
      <Composition
        id="CollusionCascade"
        component={CollusionCascade}
        durationInFrames={450}
        fps={30}
        width={1080}
        height={1080}
      />
      <Composition
        id="GovernanceLag"
        component={GovernanceLag}
        durationInFrames={540}
        fps={30}
        width={1080}
        height={1080}
      />
      <Composition
        id="DelegationGames"
        component={DelegationGames}
        durationInFrames={480}
        fps={30}
        width={1080}
        height={1080}
      />
    </>
  );
};
