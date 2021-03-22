env_info = dict(
        num_agent=10,
        env_size=35,
        radian=2,
        obs_size=[11]*10,
        num_action=8,
        intradistance_patch=None,
        apple_respawn=[0, 0.005, 0.02, 0.05],
        beam_dist=3,
        beam_reward=[-1, -50])

agent_info = dict(
        model='maddpg',
        batch_size=1024,
        capacity=1e6,
        update_freq=100,
        save_dir='./save',
        use_gpu=True,
        render=False,
        max_step=100,
        num_episode=10000,
        intinsic_reward=[45]*10)

model_info = dict(
        hidden_size=512,
        optimizer='adam',
        lr=1e-2,
        gamma=0.99,
        )

