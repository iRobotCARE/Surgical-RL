import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np

from rl.utils.general_utils import AttrDict, RecursiveAverageMeter


def _reset_env_return_obs(env, *args, **kwargs):
    """
    调用 env.reset(...)，兼容旧版 gym（只返回 obs）和 gymnasium（返回 (obs, info)）。
    返回 obs（第一个元素）。
    """
    result = env.reset(*args, **kwargs)
    if isinstance(result, tuple) or isinstance(result, list):
        return result[0]
    return result


def _ensure_obs_dict(obs):
    """
    如果传入的是 (obs, info) 之类的 tuple/list，则取第一个元素并返回。
    用于在缓存层兼容不同 env.reset()/env.step() 返回风格。
    """
    if isinstance(obs, (tuple, list)):
        obs = obs[0]
    return obs


def get_env_params(env, cfg):
    obs = _reset_env_return_obs(env)
    env_params = AttrDict(
        obs=np.asarray(obs['observation']).shape[0],
        achieved_goal=np.asarray(obs['achieved_goal']).shape[0],
        goal=np.asarray(obs['desired_goal']).shape[0],
        act=env.action_space.shape[0],
        act_rand_sampler=env.action_space.sample,
        max_timesteps=getattr(env, "_max_episode_steps",
                              getattr(env, "max_episode_steps",
                                      getattr(env, "spec", None).max_episode_steps if getattr(env, "spec", None) is not None else None)),
        max_action=env.action_space.high[0],
    )
    return env_params


def get_env_params_viskill(env, cfg):
    obs = _reset_env_return_obs(env)
    env_params = AttrDict(
        obs=np.asarray(obs['observation']).shape[0],
        achieved_goal=np.asarray(obs['achieved_goal']).shape[0],
        goal=np.asarray(obs['desired_goal']).shape[0],
        act=env.action_space.shape[0],
        act_rand_sampler=env.action_space.sample,
        max_timesteps=getattr(env, "max_episode_steps",
                              getattr(env, "_max_episode_steps",
                                      getattr(env, "spec", None).max_episode_steps if getattr(env, "spec", None) is not None else None)),
        max_action=env.action_space.high[0],
    )
    return env_params


class ReplayCache:
    def __init__(self, T):
        self.T = T
        self.reset()

    def reset(self):
        self.t = 0
        self.obs, self.ag, self.g, self.actions, self.dones = [], [], [], [], []

    def store_transition(self, obs, action, done):
        # 兼容 (obs, info) 返回
        obs = _ensure_obs_dict(obs)
        self.obs.append(obs['observation'])
        self.ag.append(obs['achieved_goal'])
        self.g.append(obs['desired_goal'])
        self.actions.append(action)
        self.dones.append(done)

    def store_obs(self, obs):
        # 兼容 (obs, info) 返回
        obs = _ensure_obs_dict(obs)
        self.obs.append(obs['observation'])
        self.ag.append(obs['achieved_goal'])

    def pop(self):
        assert len(self.obs) == self.T + 1 and len(self.actions) == self.T
        obs = np.expand_dims(np.array(self.obs.copy()), axis=0)
        ag = np.expand_dims(np.array(self.ag.copy()), axis=0)
        # print(self.ag)
        g = np.expand_dims(np.array(self.g.copy()), axis=0)
        actions = np.expand_dims(np.array(self.actions.copy()), axis=0)
        dones = np.expand_dims(np.array(self.dones.copy()), axis=1)
        dones = np.expand_dims(dones, axis=0)

        self.reset()
        episode = AttrDict(obs=obs, ag=ag, g=g, actions=actions, dones=dones)
        return episode


class ReplayCacheGT(ReplayCache):
    """
    ReplayCache variant that also stores ground-truth goal (gt_goal) per transition.
    pop() returns an AttrDict with an extra field 'gt' containing the gt goals
    with same leading batch dimension as other arrays (shape: (1, T, ...)).
    """
    def __init__(self, T):
        super().__init__(T)
        self.gt = []

    def reset(self):
        super().reset()
        self.gt = []

    def store_transition(self, obs, action, done, gt_goal=None):
        # reuse parent's storage for obs/ag/g/actions/dones (parent handles unwrapping)
        super().store_transition(obs, action, done)
        # store ground-truth goal (can be None)
        self.gt.append(gt_goal)

    def pop(self):
        # ensure lengths match
        assert len(self.obs) == self.T + 1 and len(self.actions) == self.T and len(self.gt) == self.T

        # Build arrays directly from current lists (do NOT call super().pop() because it resets lists)
        obs = np.expand_dims(np.array(self.obs.copy()), axis=0)
        ag = np.expand_dims(np.array(self.ag.copy()), axis=0)
        g = np.expand_dims(np.array(self.g.copy()), axis=0)
        actions = np.expand_dims(np.array(self.actions.copy()), axis=0)
        dones = np.expand_dims(np.array(self.dones.copy()), axis=1)
        dones = np.expand_dims(dones, axis=0)
        gt = np.expand_dims(np.array(self.gt.copy()), axis=0)

        # reset internal buffers
        self.reset()
        episode = AttrDict(obs=obs, ag=ag, g=g, actions=actions, dones=dones, gt_g=gt)
        return episode


def init_buffer(cfg, buffer, agent, normalize=True):
    '''Load demonstrations into buffer and initilaize normalizer'''
    demo_path = cfg.demo_path
    demo = np.load(demo_path, allow_pickle=True)
    demo_obs, demo_acs = demo['obs'], demo['acs']

    episode_cache = ReplayCache(buffer.T)
    for epsd in range(cfg.num_demo):
        episode_cache.store_obs(demo_obs[epsd][0])
        for i in range(buffer.T):
            # print(buffer.T)
            episode_cache.store_transition(
                obs=demo_obs[epsd][i+1],
                action=demo_acs[epsd][i],
                done=i == (buffer.T - 1),
            )
        episode = episode_cache.pop()
        buffer.store_episode(episode)
        if normalize:
            agent.update_normalizer(episode)


def init_demo_buffer(cfg, buffer, agent, subtask=None, update_normalizer=True):
    '''Load demonstrations into buffer and initilaize normalizer'''
    demo_path = os.path.join(os.getcwd(), 'SurRoL/surrol/data/demo')
    file_name = "data_"
    file_name += cfg.task
    file_name += "_" + 'random'
    if subtask is None:
        file_name += "_" + str(cfg.num_demo) + '_primitive_new' + cfg.subtask
    else:
        file_name += "_" + str(cfg.num_demo) + '_primitive_new' + subtask
    file_name += ".npz"

    demo_path = os.path.join(demo_path, file_name)
    demo = np.load(demo_path, allow_pickle=True)
    demo_obs, demo_acs, demo_gt = demo['observations'], demo['actions'], demo['gt_actions']

    episode_cache = ReplayCacheGT(buffer.T)
    for epsd in range(cfg.num_demo):
        episode_cache.store_obs(demo_obs[epsd][0])
        for i in range(buffer.T):
            episode_cache.store_transition(
                obs=demo_obs[epsd][i+1],
                action=demo_acs[epsd][i],
                done=i == (buffer.T - 1),
                gt_goal=demo_gt[epsd][i]
            )
        episode = episode_cache.pop()
        buffer.store_episode(episode)
        if update_normalizer:
            agent.update_normalizer(episode)


def init_sc_buffer(cfg, buffer, agent, env_params):
    '''Load demonstrations into buffer and initilaize normalizer'''
    for subtask in env_params.subtasks:
        demo_path = os.path.join(os.getcwd(), 'SurRoL/surrol/data/demo')
        file_name = "data_"
        file_name += cfg.task
        file_name += "_" + 'random'
        file_name += "_" + str(cfg.num_demo) + '_primitive_new' + subtask
        file_name += ".npz"

        demo_path = os.path.join(demo_path, file_name)
        demo = np.load(demo_path, allow_pickle=True)
        demo_obs, demo_acs, demo_gt = demo['observations'], demo['actions'], demo['gt_actions']

        for epsd in range(cfg.num_demo):
            obs = demo_obs[epsd][0]['observation']
            next_obs = demo_obs[epsd][-1]['observation']
            action = demo_obs[epsd][0]['desired_goal'][:-env_params.len_cond]
            # reward = sum([env_params.reward_funcs[subtask](demo_obs[epsd][i+1]['achieved_goal'], demo_obs[epsd][i+1]['desired_goal']) \
            #         for i in range(len(demo_acs[epsd]))])
            reward = env_params.reward_funcs[subtask](demo_obs[epsd][-1]['achieved_goal'], demo_obs[epsd][-1]['desired_goal'])
            # print(subtask, epsd, reward)
            done = subtask not in env_params.next_subtasks.keys()
            reward = done * reward
            gt_action = demo_gt[epsd][-1]
            buffer[subtask].add(obs, action, reward, next_obs, done, gt_action)
            if agent.sc_agent.normalize:
                # TODO: hide normalized
                agent.sc_agent.o_norm[subtask].update(obs)


class RolloutStorage:
    """Can hold multiple rollouts, can compute statistics over these rollouts."""
    def __init__(self):
        self.rollouts = []

    def append(self, rollout):
        """Adds rollout to storage."""
        self.rollouts.append(rollout)

    def rollout_stats(self):
        """Returns AttrDict of average statistics over the rollouts."""
        assert self.rollouts    # rollout storage should not be empty
        stats = RecursiveAverageMeter()
        for rollout in self.rollouts:
            stats.update(AttrDict(
                avg_reward=np.stack(rollout.reward).sum(),
                avg_success_rate=rollout.success[-1]
            ))
        return stats.avg

    def reset(self):
        del self.rollouts
        self.rollouts = []

    def get(self):
        return self.rollouts

    def __contains__(self, key):
        return self.rollouts and key in self.rollouts[0]
    