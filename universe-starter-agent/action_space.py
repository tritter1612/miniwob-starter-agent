import logging

import gym
import numpy as np
import universe
from universe import spaces as vnc_spaces
from universe import vectorized
from universe.wrappers import Logger
from universe.wrappers.gym_core import gym_core_action_space

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
universe.configure_logging()

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


class SoftmaxClickTask(vectorized.ActionWrapper):
    """
    Creates a Discrete action space of mouse clicks.

    This wrapper divides the active region into cells and creates an action for
    each which clicks in the middle of the cell.
    """
    def __init__(self, env, active_region=(10, 75 + 50, 10 + 160, 75 + 210), discrete_mouse_step=10, noclick_regions=[], noAgent=False):
        super(SoftmaxClickTask, self).__init__(env)
        logger.info('Using SoftmaxClickTask with action_region={}, noclick_regions={}'.format(active_region, noclick_regions))
        xlow, ylow, xhigh, yhigh = active_region
        xs = range(xlow, xhigh, discrete_mouse_step)
        ys = range(ylow, yhigh, discrete_mouse_step)
        self.active_region = active_region
        self.discrete_mouse_step = discrete_mouse_step
        self.noclick_regions = noclick_regions
        self.noAgent = noAgent
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
        logger.info('SoftmaxClickTask noclick regions removed {} of {} actions'.format(removed, removed + len(self._points)))
        self.action_space = gym.spaces.Discrete(len(self._points))

    def _action(self, action_n):
        return [self._discrete_to_action(int(i)) for i in action_n]

    def _discrete_to_action(self, i):
        if self.noAgent:
            return
        else:
            xc, yc = self._points[i]
        return [
            vnc_spaces.PointerEvent(xc, yc, buttonmask=0), # release
            vnc_spaces.PointerEvent(xc, yc, buttonmask=1), # click
            vnc_spaces.PointerEvent(xc, yc, buttonmask=0) # release
        ]

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

class SoftmaxClickTaskDirectSubmit(SoftmaxClickTask):
    def __init__(self, env, active_region=(10, 75 + 50, 10 + 160, 75 + 210 - 35), discrete_mouse_step=10, noclick_regions=[], noAgent=False):
        super(SoftmaxClickTaskDirectSubmit, self).__init__(env, active_region, discrete_mouse_step, noclick_regions)
        logger.info('SoftmaxClickTaskDirectSubmit was used')

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

class SoftmaxDragTask(SoftmaxClickTask):
    def __init__(self, env, active_region=(10, 75 + 50, 10 + 160, 75 + 210 - 110), discrete_mouse_step=10, noclick_regions=[], noAgent=False):
        super(SoftmaxDragTask, self).__init__(env, active_region, discrete_mouse_step, noclick_regions, noAgent)
        logger.info('SoftmaxDragTask was used')
        self.action_space = gym.spaces.Discrete(len(self._points) * 3)

    def _discrete_to_action(self, i):
        if self.noAgent:
            return
        xc, yc = self._points[i % len(self._points)]
        if i < len(self._points):
            # Click on Object
            return [
                vnc_spaces.PointerEvent(xc, yc, buttonmask=0),
                vnc_spaces.PointerEvent(xc, yc, buttonmask=1)
            ]
        elif i < len(self._points * 2):
            # Drag Object to a place and release it
            return [
                vnc_spaces.PointerEvent(xc, yc, buttonmask=0)
            ]
        else:
            # Click Submit Button
            return [
                vnc_spaces.PointerEvent(xc, yc, buttonmask=0),
                vnc_spaces.PointerEvent(xc, yc, buttonmask=1),
                vnc_spaces.PointerEvent(xc, yc, buttonmask=0)
            ]

class SoftmaxDragTaskDirectSubmit(SoftmaxClickTask):
    def __init__(self, env, active_region=(10, 75 + 50, 10 + 160, 75 + 210 - 110), discrete_mouse_step=10, noclick_regions=[], noAgent=False):
        super(SoftmaxDragTaskDirectSubmit, self).__init__(env, active_region, discrete_mouse_step, noclick_regions, noAgent)
        logger.info('SoftmaxDragTaskDirectSubmit was used')
        self._is_clicked = False

    def _discrete_to_action(self, i):
        xc, yc = self._points[i]
        if self._is_clicked:
            self._is_clicked = False
            return [
                vnc_spaces.PointerEvent(xc, yc, buttonmask=0),  # release
                # Click Submit button
                vnc_spaces.PointerEvent(10 + 47, 75 + 50 + 160 - 38, buttonmask=0),  # release
                vnc_spaces.PointerEvent(10 + 47, 75 + 50 + 160 - 38, buttonmask=1),  # click
                vnc_spaces.PointerEvent(10 + 47, 75 + 50 + 160 - 38, buttonmask=0),  # release
            ]
        else:
            self._is_clicked = True
            return [
                vnc_spaces.PointerEvent(xc, yc, buttonmask=0),  # release
                vnc_spaces.PointerEvent(xc, yc, buttonmask=1)   # click
            ]

class SoftmaxCopyPasteTask(SoftmaxClickTask):
    def __init__(self, env, active_region=(10, 75 + 50, 10 + 160, 75 + 210), discrete_mouse_step=10, noclick_regions=[], noAgent=False):
        super(SoftmaxCopyPasteTask, self).__init__(env, active_region, discrete_mouse_step, noclick_regions, noAgent)
        logger.info('SoftmaxCopyPasteTask was used')
        self._keys = ['ctrl-a', 'ctrl-c', 'ctrl-v']
        self.action_space = gym.spaces.Discrete(len(self._points) + len(self._keys))

    def _discrete_to_action(self, i):
        merged_actions = self._points + self._keys
        if self.noAgent:
            return
        if type(merged_actions[i]) == tuple:
            xc, yc = merged_actions[i]
            # click in text field, in empty field or on submit button
            return [
                vnc_spaces.PointerEvent(xc, yc, buttonmask=0),      # release
                vnc_spaces.PointerEvent(xc, yc, buttonmask=1),      # click
                vnc_spaces.PointerEvent(xc, yc, buttonmask=0)       # release
            ]
        else:
            key = merged_actions[i]
            split1 = 'ctrl'
            if (key == 'ctrl-a'):
                split2 = 'a'
            elif (key == 'ctrl-c'):
                split2 = 'c'
            else:
                split2 = 'v'
            return [
                vnc_spaces.KeyEvent.by_name(split1, down=True),     # press
                vnc_spaces.KeyEvent.by_name(split2, down=True),     # press
                vnc_spaces.KeyEvent.by_name(split1, down=False),    # release
                vnc_spaces.KeyEvent.by_name(split2, down=False)     # release
            ]

class SoftmaxCopyPasteTaskWithOrder(SoftmaxClickTask): # TODO: Implement for random agent
    def __init__(self, env, active_region=(10, 75 + 50, 10 + 160, 75 + 210), discrete_mouse_step=10, noclick_regions=[], noAgent=False):
        super(SoftmaxCopyPasteTaskWithOrder, self).__init__(env, active_region, discrete_mouse_step, noclick_regions, noAgent)
        logger.info('SoftmaxCopyPasteTaskWithOrder was used')
        self._action_code = 0

    def _discrete_to_action(self, i):
        if self.noAgent:
            return
        else:
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

class SoftmaxMathTasks(SoftmaxClickTask):
    def __init__(self, env, active_region=(10, 75 + 50, 10 + 160, 75 + 210), discrete_mouse_step=10, noclick_regions=[], noAgent=False):
        super(SoftmaxMathTasks, self).__init__(env, active_region, discrete_mouse_step, noclick_regions, noAgent)
        logger.info('SoftmaxMathTasks was used')
        if env.spec.id == 'wob.mini.SimpleAlgebra-v0':
            self._keys = range(-99, 100)
        elif env.spec.id == 'wob.mini.SimpleArithmetic-v0':
            self._keys = range(-9, 100)
        elif env.spec.id == 'wob.mini.VisualAddition-v0':
            self._keys = range(0, 21)
        self.action_space = gym.spaces.Discrete(len(self._keys) + len(self._points))

    def _discrete_to_action(self, i):
        merged_actions = list(self._keys) + self._points
        if self.noAgent:
            return
        else:
            if type(merged_actions[i]) == tuple:
                xc, yc = merged_actions[i]
                return [
                    # Click in text field or on submit button
                    vnc_spaces.PointerEvent(xc, yc, buttonmask=0),
                    vnc_spaces.PointerEvent(xc, yc, buttonmask=1),
                    vnc_spaces.PointerEvent(xc, yc, buttonmask=0)
                ]
            else:
                key = merged_actions[i]
                # Enter number
                if key >= 0:
                    if key < 10:
                        return [
                            vnc_spaces.KeyEvent.by_name(str(key), down=True),
                            vnc_spaces.KeyEvent.by_name(str(key), down=False)
                        ]
                    else:
                        return [
                            vnc_spaces.KeyEvent.by_name(str(key)[:1], down=True),
                            vnc_spaces.KeyEvent.by_name(str(key)[:1], down=False),
                            vnc_spaces.KeyEvent.by_name(str(key)[1:], down=True),
                            vnc_spaces.KeyEvent.by_name(str(key)[1:], down=False)
                        ]
                else:
                    if key > -10:
                        return [
                            vnc_spaces.KeyEvent.by_name('-', down=True),
                            vnc_spaces.KeyEvent.by_name('-', down=False),
                            vnc_spaces.KeyEvent.by_name(str(key)[1:], down=True),
                            vnc_spaces.KeyEvent.by_name(str(key)[1:], down=False)
                        ]
                    else:
                        return [
                            vnc_spaces.KeyEvent.by_name('-', down=True),
                            vnc_spaces.KeyEvent.by_name('-', down=False),
                            vnc_spaces.KeyEvent.by_name(str(key)[1:2], down=True),
                            vnc_spaces.KeyEvent.by_name(str(key)[1:2], down=False),
                            vnc_spaces.KeyEvent.by_name(str(key)[2:], down=True),
                            vnc_spaces.KeyEvent.by_name(str(key)[2:], down=False)
                        ]

class SoftmaxMathTasksDirectSubmit(vectorized.ActionWrapper):
    def __init__(self, env, noAgent=False):
        super(SoftmaxMathTasksDirectSubmit, self).__init__(env)
        logger.info('SoftmaxMathTasksDirectSubmit was used')
        self.noAgent = noAgent
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
        if self.noAgent:
            return
        else:
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