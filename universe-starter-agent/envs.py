import cv2
from gym.spaces.box import Box
import numpy as np
import gym
from gym import spaces
import logging
import universe
from universe import vectorized
from universe.wrappers import BlockingReset, GymCoreAction, EpisodeID, Unvectorize, Vectorize, Vision, Logger
from universe import spaces as vnc_spaces
from universe.spaces.vnc_event import keycode
import time
#from universe.wrappers.experimental import SoftmaxClickMouse
import random
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
universe.configure_logging()

def create_env(env_id, client_id, remotes, **kwargs):
    spec = gym.spec(env_id)

    if spec.tags.get('flashgames', False):
        return create_flash_env(env_id, client_id, remotes, **kwargs)
    elif spec.tags.get('atari', False) and spec.tags.get('vnc', False):
        return create_vncatari_env(env_id, client_id, remotes, **kwargs)
    elif spec.tags.get('wob', False):
        return create_miniwob_env(env_id, client_id, remotes, **kwargs)
    else:
        # Assume atari.
        assert "." not in env_id  # universe environments have dots in names.
        return create_atari_env(env_id)

def create_flash_env(env_id, client_id, remotes, **_):
    env = gym.make(env_id)
    env = Vision(env)
    env = Logger(env)
    env = BlockingReset(env)

    reg = universe.runtime_spec('flashgames').server_registry
    height = reg[env_id]["height"]
    width = reg[env_id]["width"]
    env = CropScreen(env, height, width, 84, 18)
    env = FlashRescale(env)

    keys = ['left', 'right', 'up', 'down', 'x']
    if env_id == 'flashgames.NeonRace-v0':
        # Better key space for this game.
        keys = ['left', 'right', 'up', 'left up', 'right up', 'down', 'up x']
    logger.info('create_flash_env(%s): keys=%s', env_id, keys)

    env = DiscreteToFixedKeysVNCActions(env, keys)
    env = EpisodeID(env)
    env = DiagnosticsInfo(env)
    env = Unvectorize(env)
    env.configure(fps=5.0, remotes=remotes, start_timeout=15 * 60, client_id=client_id,
                  vnc_driver='go', vnc_kwargs={
                    'encoding': 'tight', 'compress_level': 0,
                    'fine_quality_level': 50, 'subsample_level': 3})
    return env

def create_vncatari_env(env_id, client_id, remotes, **_):
    env = gym.make(env_id)
    env = Vision(env)
    env = Logger(env)
    env = BlockingReset(env)
    env = GymCoreAction(env)
    env = AtariRescale42x42(env)
    env = EpisodeID(env)
    env = DiagnosticsInfo(env)
    env = Unvectorize(env)

    logger.info('Connecting to remotes: %s', remotes)
    fps = env.metadata['video.frames_per_second']
    env.configure(remotes=remotes, start_timeout=15 * 60, fps=fps, client_id=client_id)
    return env

def create_atari_env(env_id):
    env = gym.make(env_id)
    env = Vectorize(env)
    env = AtariRescale42x42(env)
    env = DiagnosticsInfo(env)
    env = Unvectorize(env)
    return env

def create_miniwob_env(env_id, client_id, remotes, **_):
    env = gym.make(env_id)
    env = Vision(env)
    env = Logger(env)
    env = BlockingReset(env)

    env = CropScreen(env, 160, 160, 125, 10)
    if (env_id == 'wob.mini.ClickTest-v0') or (env_id == 'wob.mini.NumberCheckboxes-v0') or (env_id == 'wob.mini.ChaseCircle-v0') or (env_id == 'wob.mini.BisectAngle-v0') or (env_id == 'wob.mini.FindMidpoint-v0') or (env_id == 'wob.mini.CircleCenter-v0'):
        obs_height = 80
        obs_width = 80
    else:
        obs_height = 100
        obs_width = 100
    env = WobRescale(env, obs_height, obs_width)

    logger.info('create_miniwob_env(%s): ', env_id)

    if env_id == 'wob.mini.NumberCheckboxes-v0':
        env = SoftmaxClickMouse(env, active_region=(10 + 14, 75 + 57, 24 + 55, 132 + 102 + 11 + 22), noclick_regions=[(24 + 11 + 2, 42, 132 + 102 + 9, 28)], discrete_mouse_step=17)
    elif (env_id == 'wob.mini.BisectAngle-v0') or (env_id == 'wob.mini.FindMidpoint-v0') or (env_id == 'wob.mini.CircleCenter-v0'):
        env = SoftmaxClickAndSubmit(env, discrete_mouse_step=8)
    elif env_id == 'wob.mini.CopyPaste-v0':
        env = SoftmaxMouseKeyboardCopyPaste(env, discrete_mouse_step=20)
    elif (env_id == 'wob.mini.SimpleAlgebra-v0') or (env_id == 'wob.mini.SimpleArithmetic-v0') or (env_id == 'wob.mini.VisualAddition-v0'):
        env = SoftmaxKeyboardMath(env)
    else:
        env = SoftmaxClickMouse(env, discrete_mouse_step=8)

    env = EpisodeID(env)
    env = DiagnosticsInfo(env)
    env = Unvectorize(env)
    if (env_id == 'wob.mini.SimpleAlgebra-v0') or (env_id == 'wob.mini.SimpleArithmetic-v0') or (env_id == 'wob.mini.VisualAddition-v0'):
        env.configure(fps=1.0, remotes=remotes, start_timeout=15 * 60, client_id=client_id,
                  vnc_driver='go', vnc_kwargs={
                    'encoding': 'tight', 'compress_level': 0,
                    'fine_quality_level': 100, 'subsample_level': 0})
    else:
        env.configure(fps=5.0, remotes=remotes, start_timeout=15 * 60, client_id=client_id,
                      vnc_driver='go', vnc_kwargs={
                'encoding': 'tight', 'compress_level': 0,
                'fine_quality_level': 100, 'subsample_level': 0})
    return env

def DiagnosticsInfo(env, *args, **kwargs):
    return vectorized.VectorizeFilter(env, DiagnosticsInfoI, *args, **kwargs)

class DiagnosticsInfoI(vectorized.Filter):
    def __init__(self, log_interval=503):
        super(DiagnosticsInfoI, self).__init__()

        self._episode_time = time.time()
        self._last_time = time.time()
        self._local_t = 0
        self._log_interval = log_interval
        self._episode_reward = 0
        self._episode_length = 0
        self._all_rewards = []
        self._num_vnc_updates = 0
        self._last_episode_id = -1

    def _after_reset(self, observation):
        logger.info('Resetting environment')
        self._episode_reward = 0
        self._episode_length = 0
        self._all_rewards = []
        return observation

    def _after_step(self, observation, reward, done, info):
        to_log = {}
        if self._episode_length == 0:
            self._episode_time = time.time()

        self._local_t += 1
        if info.get("stats.vnc.updates.n") is not None:
            self._num_vnc_updates += info.get("stats.vnc.updates.n")

        if self._local_t % self._log_interval == 0:
            cur_time = time.time()
            elapsed = cur_time - self._last_time
            fps = self._log_interval / elapsed
            self._last_time = cur_time
            cur_episode_id = info.get('vectorized.episode_id', 0)
            to_log["diagnostics/fps"] = fps
            if self._last_episode_id == cur_episode_id:
                to_log["diagnostics/fps_within_episode"] = fps
            self._last_episode_id = cur_episode_id
            if info.get("stats.gauges.diagnostics.lag.action") is not None:
                to_log["diagnostics/action_lag_lb"] = info["stats.gauges.diagnostics.lag.action"][0]
                to_log["diagnostics/action_lag_ub"] = info["stats.gauges.diagnostics.lag.action"][1]
            if info.get("reward.count") is not None:
                to_log["diagnostics/reward_count"] = info["reward.count"]
            if info.get("stats.gauges.diagnostics.clock_skew") is not None:
                to_log["diagnostics/clock_skew_lb"] = info["stats.gauges.diagnostics.clock_skew"][0]
                to_log["diagnostics/clock_skew_ub"] = info["stats.gauges.diagnostics.clock_skew"][1]
            if info.get("stats.gauges.diagnostics.lag.observation") is not None:
                to_log["diagnostics/observation_lag_lb"] = info["stats.gauges.diagnostics.lag.observation"][0]
                to_log["diagnostics/observation_lag_ub"] = info["stats.gauges.diagnostics.lag.observation"][1]

            if info.get("stats.vnc.updates.n") is not None:
                to_log["diagnostics/vnc_updates_n"] = info["stats.vnc.updates.n"]
                to_log["diagnostics/vnc_updates_n_ps"] = self._num_vnc_updates / elapsed
                self._num_vnc_updates = 0
            if info.get("stats.vnc.updates.bytes") is not None:
                to_log["diagnostics/vnc_updates_bytes"] = info["stats.vnc.updates.bytes"]
            if info.get("stats.vnc.updates.pixels") is not None:
                to_log["diagnostics/vnc_updates_pixels"] = info["stats.vnc.updates.pixels"]
            if info.get("stats.vnc.updates.rectangles") is not None:
                to_log["diagnostics/vnc_updates_rectangles"] = info["stats.vnc.updates.rectangles"]
            if info.get("env_status.state_id") is not None:
                to_log["diagnostics/env_state_id"] = info["env_status.state_id"]
            # what of these statistics (see iuniverse/wrapper.logger.py) is part of our console output?

        if reward is not None:
            self._episode_reward += reward
            if observation is not None:
                self._episode_length += 1
            self._all_rewards.append(reward)

        if done:
            logger.info('Episode terminating: episode_reward=%s episode_length=%s', self._episode_reward, self._episode_length)
            total_time = time.time() - self._episode_time
            to_log["global/episode_reward"] = self._episode_reward
            to_log["global/episode_length"] = self._episode_length
            to_log["global/episode_time"] = total_time
            to_log["global/reward_per_time"] = self._episode_reward / total_time
            self._episode_reward = 0
            self._episode_length = 0
            self._all_rewards = []

        return observation, reward, done, to_log

def _process_frame42(frame):
    frame = frame[34:34+160, :160]
    # Resize by half, then down to 42x42 (essentially mipmapping). If
    # we resize directly we lose pixels that, when mapped to 42x42,
    # aren't close enough to the pixel boundary.
    frame = cv2.resize(frame, (80, 80))
    frame = cv2.resize(frame, (42, 42))
    frame = frame.mean(2)
    frame = frame.astype(np.float32)
    frame *= (1.0 / 255.0)
    frame = np.reshape(frame, [42, 42, 1])
    return frame

class AtariRescale42x42(vectorized.ObservationWrapper):
    def __init__(self, env=None):
        super(AtariRescale42x42, self).__init__(env)
        self.observation_space = Box(0.0, 1.0, [42, 42, 1])

    def _observation(self, observation_n):
        return [_process_frame42(observation) for observation in observation_n]

class FixedKeyState(object):
    def __init__(self, keys):
        self._keys = [keycode(key) for key in keys]
        self._down_keysyms = set()

    def apply_vnc_actions(self, vnc_actions):
        for event in vnc_actions:
            if isinstance(event, vnc_spaces.KeyEvent):
                if event.down:
                    self._down_keysyms.add(event.key)
                else:
                    self._down_keysyms.discard(event.key)

    def to_index(self):
        action_n = 0
        for key in self._down_keysyms:
            if key in self._keys:
                # If multiple keys are pressed, just use the first one
                action_n = self._keys.index(key) + 1
                break
        return action_n

class DiscreteToFixedKeysVNCActions(vectorized.ActionWrapper):
    """
    Define a fixed action space. Action 0 is all keys up. Each element of keys can be a single key or a space-separated list of keys

    For example,
       e=DiscreteToFixedKeysVNCActions(e, ['left', 'right'])
    will have 3 actions: [none, left, right]

    You can define a state with more than one key down by separating with spaces. For example,
       e=DiscreteToFixedKeysVNCActions(e, ['left', 'right', 'space', 'left space', 'right space'])
    will have 6 actions: [none, left, right, space, left space, right space]
    """
    def __init__(self, env, keys):
        super(DiscreteToFixedKeysVNCActions, self).__init__(env)

        self._keys = keys
        self._generate_actions()
        self.action_space = spaces.Discrete(len(self._actions))

    def _generate_actions(self):
        self._actions = []
        uniq_keys = set()
        for key in self._keys:
            for cur_key in key.split(' '):
                uniq_keys.add(cur_key)

        for key in [''] + self._keys:
            split_keys = key.split(' ')
            cur_action = []
            for cur_key in uniq_keys:
                cur_action.append(vnc_spaces.KeyEvent.by_name(cur_key, down=(cur_key in split_keys)))
            self._actions.append(cur_action)
        self.key_state = FixedKeyState(uniq_keys)

    def _action(self, action_n):
        # Each action might be a length-1 np.array. Cast to int to
        # avoid warnings.
        return [self._actions[int(action)] for action in action_n]

class CropScreen(vectorized.ObservationWrapper):
    """Crops out a [height]x[width] area starting from (top,left) """
    def __init__(self, env, height, width, top=0, left=0):
        super(CropScreen, self).__init__(env)
        self.height = height
        self.width = width
        self.top = top
        self.left = left
        self.observation_space = Box(0, 255, shape=(height, width, 3))

    def _observation(self, observation_n):
        return [ob[self.top:self.top+self.height, self.left:self.left+self.width, :] if ob is not None else None
                for ob in observation_n]

def _process_frame_flash(frame):
    frame = cv2.resize(frame, (200, 128))
    frame = frame.mean(2).astype(np.float32)
    frame *= (1.0 / 255.0)
    frame = np.reshape(frame, [128, 200, 1])
    return frame

class FlashRescale(vectorized.ObservationWrapper):
    def __init__(self, env=None):
        super(FlashRescale, self).__init__(env)
        self.observation_space = Box(0.0, 1.0, [128, 200, 1])

    def _observation(self, observation_n):
        return [_process_frame_flash(observation) for observation in observation_n]

def _process_frame_wob(frame, obs_height, obs_width):
    if (obs_height != 160) or (obs_width != 160):
        frame = cv2.resize(frame, (obs_height, obs_width))
    frame = frame.mean(2)
    frame = frame.astype(np.float32)
    frame *= (1.0 / 255.0)
    frame = np.reshape(frame, [obs_height, obs_width, 1])
    return frame

class WobRescale(vectorized.ObservationWrapper):
    def __init__(self, env=None, obs_height=160, obs_width=160):
        super(WobRescale, self).__init__(env)
        self.obs_height = obs_height
        self.obs_width = obs_width
        self.observation_space = Box(0.0, 1.0, [obs_height, obs_width, 1])

    def _observation(self, observation_n):
        return [_process_frame_wob(observation, self.obs_height, self.obs_width) for observation in observation_n]


def slither_vnc(space=False, left=False, right=False):
    return [vnc_spaces.KeyEvent.by_name('space', down=space),
            vnc_spaces.KeyEvent.by_name('left', down=left),
            vnc_spaces.KeyEvent.by_name('right', down=right)]

def racing_vnc(up=False, left=False, right=False):
    return [vnc_spaces.KeyEvent.by_name('up', down=up),
            vnc_spaces.KeyEvent.by_name('left', down=left),
            vnc_spaces.KeyEvent.by_name('right', down=right)]

def platform_vnc(up=False, left=False, right=False, space=False):
    return [vnc_spaces.KeyEvent.by_name('up', down=up),
            vnc_spaces.KeyEvent.by_name('left', down=left),
            vnc_spaces.KeyEvent.by_name('right', down=right),
            vnc_spaces.KeyEvent.by_name('space', down=space)]


class SoftmaxClickMouse(vectorized.ActionWrapper):
    """
    Creates a Discrete action space of mouse clicks.

    This wrapper divides the active region into cells and creates an action for
    each which clicks in the middle of the cell.
    """
    def __init__(self, env, active_region=(10, 75 + 50, 10 + 160, 75 + 210), discrete_mouse_step=10, noclick_regions=[]):
        super(SoftmaxClickMouse, self).__init__(env)
        logger.info('Using SoftmaxClickMouse with action_region={}, noclick_regions={}'.format(active_region, noclick_regions))
        xlow, ylow, xhigh, yhigh = active_region
        xs = range(xlow, xhigh, discrete_mouse_step)
        ys = range(ylow, yhigh, discrete_mouse_step)
        self.active_region = active_region
        self.discrete_mouse_step = discrete_mouse_step
        self.noclick_regions = noclick_regions
        self._points = []
        removed = 0
        for x in xs:
            for y in ys:
                xc = min(x+int(discrete_mouse_step/2), xhigh-1) # click to center of a cell
                yc = min(y+int(discrete_mouse_step/2), yhigh-1)
                if any(self.is_contained((xc, yc), r) for r in noclick_regions):
                    removed += 1
                    continue
                self._points.append((xc, yc))
        logger.info('SoftmaxClickMouse noclick regions removed {} of {} actions'.format(removed, removed + len(self._points)))
        self.action_space = gym.spaces.Discrete(len(self._points))

    def _action(self, action_n):
        #return [self._discrete_to_action(int(i)) for i in action_n]
        #return [self._random_action()]
        return [self._no_action()]


    def _discrete_to_action(self, i):
        xc, yc = self._points[i]
        return [
            vnc_spaces.PointerEvent(xc, yc, buttonmask=0), # release
            vnc_spaces.PointerEvent(xc, yc, buttonmask=1), # click
            vnc_spaces.PointerEvent(xc, yc, buttonmask=0), # release
        ]

    def _random_action(self):
        xc, yc = random.choice(self._points)
        return [
            vnc_spaces.PointerEvent(xc, yc, buttonmask=0), # release
            vnc_spaces.PointerEvent(xc, yc, buttonmask=1), # click
            vnc_spaces.PointerEvent(xc, yc, buttonmask=0), # release
        ]

    def _no_action(self):
        return

    def _reverse_action(self, action):
        xlow, ylow, xhigh, yhigh = self.active_region
        try:
            # find first valid mousedown, ignore everything else
            click_event = next(e for e in action if isinstance(e, vnc_spaces.PointerEvent) and e.buttonmask == 1)
            index = self._action_to_discrete(click_event)
            if index is None:
                return np.zeros(len(self._points))
            else:
                # return one-hot vector, expected by demo training code
                # FIXME(jgray): move one-hot translation to separate layer
                return np.eye(len(self._points))[index]
        except StopIteration:
            # no valid mousedowns
            return np.zeros(len(self._points))

    def _action_to_discrete(self, event):
        assert isinstance(event, vnc_spaces.PointerEvent)
        x, y = event.x, event.y
        step = self.discrete_mouse_step
        xlow, ylow, xhigh, yhigh = self.active_region
        xc = min((int((x - xlow) / step) * step) + xlow + step / 2, xhigh - 1)
        yc = min((int((y - ylow) / step) * step) + ylow + step / 2, yhigh - 1)
        try:
            return self._points.index((xc, yc))
        except ValueError:
            # ignore clicks outside of active region or in noclick regions
            return None

    @classmethod
    def is_contained(cls, point, coords):
        px, py = point
        x, width, y, height = coords
        return x <= px <= x + width and y <= py <= y + height

class SoftmaxClickAndSubmit(SoftmaxClickMouse):
    def __init__(self, env, active_region=(10, 75 + 50, 10 + 160, 75 + 210 - 35), discrete_mouse_step=10, noclick_regions=[]):
        super(SoftmaxClickAndSubmit, self).__init__(env, active_region, discrete_mouse_step, noclick_regions)
        logger.info('SoftmaxClickAndSubmit was used')

    def _discrete_to_action(self, i):
        xc, yc = self._points[i]
        return [
            vnc_spaces.PointerEvent(xc, yc, buttonmask=0), # release
            vnc_spaces.PointerEvent(xc, yc, buttonmask=1), # click
            vnc_spaces.PointerEvent(xc, yc, buttonmask=0), # release
            # Click on Submit Button
            vnc_spaces.PointerEvent(10 + 160 / 2, 75 + 50 + 160 - 30 / 2, buttonmask=0),  # release
            vnc_spaces.PointerEvent(10 + 160 / 2, 75 + 50 + 160 - 30 / 2, buttonmask=1),  # click
            vnc_spaces.PointerEvent(10 + 160 / 2, 75 + 50 + 160 - 30 / 2, buttonmask=0),  # release
        ]

class SoftmaxMouseKeyboardCopyPaste(SoftmaxClickMouse):
    def __init__(self, env, active_region=(10, 75 + 50, 10 + 160, 75 + 210), discrete_mouse_step=10, noclick_regions=[]):
        super(SoftmaxMouseKeyboardCopyPaste, self).__init__(env, active_region, discrete_mouse_step, noclick_regions)
        logger.info('SoftmaxMouseKeyboardCopyPaste was used')
        self._action_code = 0

    def _discrete_to_action(self, i):
        xc, yc = self._points[i]
        if self._action_code == 0:
            self._action_code = 1
            return [
                # click in text field
                vnc_spaces.PointerEvent(xc, yc, buttonmask=0),      # release
                vnc_spaces.PointerEvent(xc, yc, buttonmask=1),      # click
                vnc_spaces.PointerEvent(xc, yc, buttonmask=0),      # release
                vnc_spaces.KeyEvent.by_name('ctrl', down=True),     # press
                vnc_spaces.KeyEvent.by_name('a', down=True),        # press
                vnc_spaces.KeyEvent.by_name('ctrl', down=False),    # release
                vnc_spaces.KeyEvent.by_name('a', down=False),       # release
                vnc_spaces.KeyEvent.by_name('ctrl', down=True),     # press
                vnc_spaces.KeyEvent.by_name('c', down=True),        # press
                vnc_spaces.KeyEvent.by_name('ctrl', down=False),    # release
                vnc_spaces.KeyEvent.by_name('c', down=False)        # release
            ]
        elif self._action_code == 1:
            self._action_code = 2
            return [
                # click in empty field
                vnc_spaces.PointerEvent(xc, yc, buttonmask=0),      # release
                vnc_spaces.PointerEvent(xc, yc, buttonmask=1),      # click
                vnc_spaces.PointerEvent(xc, yc, buttonmask=0),      # release
                vnc_spaces.KeyEvent.by_name('ctrl', down=True),     # press
                vnc_spaces.KeyEvent.by_name('v', down=True),        # press
                vnc_spaces.KeyEvent.by_name('ctrl', down=False),    # release
                vnc_spaces.KeyEvent.by_name('v', down=False)        # release
            ]
        else:
            self._action_code = 0
            return [
                # click submit button
                vnc_spaces.PointerEvent(xc, yc, buttonmask=0),      # release
                vnc_spaces.PointerEvent(xc, yc, buttonmask=1),      # click
                vnc_spaces.PointerEvent(xc, yc, buttonmask=0),      # release
            ]

class SoftmaxKeyboardMath(vectorized.ActionWrapper):
    def __init__(self, env):
        super(SoftmaxKeyboardMath, self).__init__(env)
        logger.info('SoftmaxKeyboardMath was used')
        if env.spec.id == 'wob.mini.SimpleAlgebra-v0':
            self._keys = range(-99, 100)
        elif env.spec.id == 'wob.mini.SimpleArithmetic-v0':
            self._keys = range (-9, 100)
        elif env.spec.id == 'wob.mini.VisualAddition-v0':
            self._keys = range (0, 21)
        self.action_space = gym.spaces.Discrete(len(self._keys))

    def _action(self, action_n):
        return [self._discrete_to_action(int(i)) for i in action_n]

    def _discrete_to_action(self, i):
        key = self._keys[i]
        result = []
        # Click in text field
        if self.env.spec.id == 'wob.mini.SimpleAlgebra-v0':
            result.append(vnc_spaces.PointerEvent(10 + 110, 75 + 50 + 65, buttonmask=0))
            result.append(vnc_spaces.PointerEvent(10 + 110, 75 + 50 + 65, buttonmask=1))
            result.append(vnc_spaces.PointerEvent(10 + 110, 75 + 50 + 65, buttonmask=0))
        elif self.env.spec.id == 'wob.mini.SimpleArithmetic-v0':
            result.append(vnc_spaces.PointerEvent(10 + 140, 75 + 50 + 35, buttonmask=0))
            result.append(vnc_spaces.PointerEvent(10 + 140, 75 + 50 + 35, buttonmask=1))
            result.append(vnc_spaces.PointerEvent(10 + 140, 75 + 50 + 35, buttonmask=0))
        elif self.env.spec.id == 'wob.mini.VisualAddition-v0':
            result.append(vnc_spaces.PointerEvent(10 + 35, 75 + 50 + 125, buttonmask=0))
            result.append(vnc_spaces.PointerEvent(10 + 35, 75 + 50 + 125, buttonmask=1))
            result.append(vnc_spaces.PointerEvent(10 + 35, 75 + 50 + 125, buttonmask=0))
        # Enter number
        if key >= 0:
            if key < 10:
                result.append(vnc_spaces.KeyEvent.by_name(str(key), down=True))
                result.append(vnc_spaces.KeyEvent.by_name(str(key), down=False))
            else:
                result.append(vnc_spaces.KeyEvent.by_name(str(key)[:1], down=True))
                result.append(vnc_spaces.KeyEvent.by_name(str(key)[:1], down=False))
                result.append(vnc_spaces.KeyEvent.by_name(str(key)[1:], down=True))
                result.append(vnc_spaces.KeyEvent.by_name(str(key)[1:], down=False))
        else:
            if key > -10:
                result.append(vnc_spaces.KeyEvent.by_name('-', down=True))
                result.append(vnc_spaces.KeyEvent.by_name('-', down=False))
                result.append(vnc_spaces.KeyEvent.by_name(str(key)[1:], down=True))
                result.append(vnc_spaces.KeyEvent.by_name(str(key)[1:], down=False))
            else:
                result.append(vnc_spaces.KeyEvent.by_name('-', down=True))
                result.append(vnc_spaces.KeyEvent.by_name('-', down=False))
                result.append(vnc_spaces.KeyEvent.by_name(str(key)[1:2], down=True))
                result.append(vnc_spaces.KeyEvent.by_name(str(key)[1:2], down=False))
                result.append(vnc_spaces.KeyEvent.by_name(str(key)[2:], down=True))
                result.append(vnc_spaces.KeyEvent.by_name(str(key)[2:], down=False))
        # Click on submit button
        if self.env.spec.id == 'wob.mini.SimpleAlgebra-v0':
            result.append(vnc_spaces.PointerEvent(10 + 160 / 2, 75 + 50 + 115, buttonmask=0))
            result.append(vnc_spaces.PointerEvent(10 + 160 / 2, 75 + 50 + 115, buttonmask=1))
            result.append(vnc_spaces.PointerEvent(10 + 160 / 2, 75 + 50 + 115, buttonmask=0))
        elif self.env.spec.id == 'wob.mini.SimpleArithmetic-v0':
            result.append(vnc_spaces.PointerEvent(10 + 160 / 2, 75 + 50 + 83, buttonmask=0))
            result.append(vnc_spaces.PointerEvent(10 + 160 / 2, 75 + 50 + 83, buttonmask=1))
            result.append(vnc_spaces.PointerEvent(10 + 160 / 2, 75 + 50 + 83, buttonmask=0))
        elif self.env.spec.id == 'wob.mini.VisualAddition-v0':
            result.append(vnc_spaces.PointerEvent(10 + 110, 75 + 50 + 125, buttonmask=0))
            result.append(vnc_spaces.PointerEvent(10 + 110, 75 + 50 + 125, buttonmask=1))
            result.append(vnc_spaces.PointerEvent(10 + 110, 75 + 50 + 125, buttonmask=0))
        return result