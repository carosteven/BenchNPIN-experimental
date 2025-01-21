from gymnasium.envs.registration import register

register(
     id="ship-ice-v0",
     entry_point="benchnpin.environments.ship_ice_nav:ShipIceEnv",
     max_episode_steps=300,
)

register(
     id="object-pushing-v0",
     entry_point="benchnpin.environments.box_pushing:ObjectPushing",
     max_episode_steps=30000,
)

register(
     id="maze-NAMO-v0",
     entry_point="benchnpin.environments.maze_NAMO:MazeNAMO",
     max_episode_steps=300000,
)


register(
     id="area-clearing-v0",
     entry_point="benchnpin.environments.area_clearing:AreaClearingEnv",
     max_episode_steps=30000,
)