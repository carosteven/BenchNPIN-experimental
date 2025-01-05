from gymnasium.envs.registration import register

register(
     id="ship-ice-v0",
     entry_point="benchnamo.environments.ship_ice_nav:ShipIceEnv",
     max_episode_steps=300,
)

register(
     id="object-pushing-v0",
     entry_point="benchnamo.environments.box_pushing:ObjectPushing",
     max_episode_steps=30000,
)