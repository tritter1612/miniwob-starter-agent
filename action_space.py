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
            return []
        else:
            xc, yc = self._points[i]
        return [
            vnc_spaces.PointerEvent(xc, yc, buttonmask=0), # release
            vnc_spaces.PointerEvent(xc, yc, buttonmask=1), # click
            vnc_spaces.PointerEvent(xc, yc, buttonmask=0) # release
        ]

    @classmethod
    def is_contained(cls, point, coords):
        px, py = point
        x, width, y, height = coords
        return x <= px <= x + width and y <= py <= y + height

class SoftmaxClickTaskDirectSubmit(SoftmaxClickTask):
    """
    Creates a Discrete action space of mouse clicks.
    After one mouse click there is a click on the submit button following.
    This class was created to simulate a reward function assigns the reward immediately after one click.
    It works with the following tasks: BisectAngle, CircleCenter, FindMidpoint & RightAngle
    """
    def __init__(self, env, active_region=(10, 75 + 50, 10 + 160, 75 + 210 - 35), discrete_mouse_step=10, noclick_regions=[], noAgent=False):
        super(SoftmaxClickTaskDirectSubmit, self).__init__(env, active_region, discrete_mouse_step, noclick_regions, noAgent)
        logger.info('SoftmaxClickTaskDirectSubmit was used')

    def _discrete_to_action(self, i):
        if self.noAgent:
            return []
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
    """
    Creates a Discrete action space of mouse clicks and drag movements.
    There are 3 different kind of actions performable:
                1. a normal click without a release
                2. a release of the clicked mouse
                3. a usual click with release
    This class is normally used in combination with the DragBox or the HighlightText environment
    """
    def __init__(self, env, active_region=(10, 75 + 50, 10 + 160, 75 + 210), discrete_mouse_step=10, noclick_regions=[], noAgent=False):
        super(SoftmaxDragTask, self).__init__(env, active_region, discrete_mouse_step, noclick_regions, noAgent)
        logger.info('SoftmaxDragTask was used')
        self.action_space = gym.spaces.Discrete(len(self._points) * 3)

    def _discrete_to_action(self, i):
        if self.noAgent:
            return []
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
    """
    Creates a Discrete action space of mouse clicks and drag movements.
    There are 2 different kind of actions performable:
                1. a normal click without a release
                2. a release of the clicked mouse
    There is a boolean variable indicating whether the mouse is clicked or not.
    If the mouse is already clicked the only action performable is the release and vice versa.
    The click on the submit button after every mouse release is hardcoded to simulate the immediate reward assignment after a performed action.
    It works with the DragBox and the HighlightText environment.
    """
    def __init__(self, env, active_region=(10, 75 + 50, 10 + 160, 75 + 210 - 110), discrete_mouse_step=10, noclick_regions=[], noAgent=False):
        super(SoftmaxDragTaskDirectSubmit, self).__init__(env, active_region, discrete_mouse_step, noclick_regions, noAgent)
        logger.info('SoftmaxDragTaskDirectSubmit was used')
        self._is_clicked = False

    def _discrete_to_action(self, i):
        if self.noAgent:
            return []
        xc, yc = self._points[i]
        if self._is_clicked:
            self._is_clicked = False
            if self.env.spec.id == 'wob.mini.DragBox-v0':
                return [
                    vnc_spaces.PointerEvent(xc, yc, buttonmask=0),  # release
                    # Click Submit button
                    vnc_spaces.PointerEvent(10 + 47, 75 + 50 + 160 - 38, buttonmask=0),  # release
                    vnc_spaces.PointerEvent(10 + 47, 75 + 50 + 160 - 38, buttonmask=1),  # click
                    vnc_spaces.PointerEvent(10 + 47, 75 + 50 + 160 - 38, buttonmask=0)  # release
                ]
            else:
                return [
                    vnc_spaces.PointerEvent(xc, yc, buttonmask=0),  # release
                    # Click Submit button
                    vnc_spaces.PointerEvent(10 + 55, 75 + 50 + 67, buttonmask=0),  # release
                    vnc_spaces.PointerEvent(10 + 55, 75 + 50 + 67, buttonmask=1),  # click
                    vnc_spaces.PointerEvent(10 + 55, 75 + 50 + 67, buttonmask=0)  # release
                ]
        else:
            self._is_clicked = True
            return [
                vnc_spaces.PointerEvent(xc, yc, buttonmask=0),  # release
                vnc_spaces.PointerEvent(xc, yc, buttonmask=1)   # click
            ]

class SoftmaxCopyPasteTask(SoftmaxClickTask):
    """
    Creates a Discrete action space of mouse clicks and the keyboard input ctrl+a, ctrl+c, ctrl+v.
    There are 4 different kind of actions performable:
                1. a usual mouse click
                2. ctrl+a
                3. ctrl+c
                4. ctrl+v
    This class was created to perfectly fit the needs of the CopyPaste environment.
    """
    def __init__(self, env, active_region=(10, 75 + 50, 10 + 160, 75 + 210), discrete_mouse_step=10, noclick_regions=[], noAgent=False):
        super(SoftmaxCopyPasteTask, self).__init__(env, active_region, discrete_mouse_step, noclick_regions, noAgent)
        logger.info('SoftmaxCopyPasteTask was used')
        self._keys = ['ctrl-a', 'ctrl-c', 'ctrl-v']
        self.merged_actions = self._points + self._keys
        self.action_space = gym.spaces.Discrete(len(self.merged_actions))


    def _discrete_to_action(self, i):
        if self.noAgent:
            return []
        if type(self.merged_actions[i]) == tuple:
            xc, yc = self.merged_actions[i]
            # click in text field, in empty field or on submit button
            return [
                vnc_spaces.PointerEvent(xc, yc, buttonmask=0),      # release
                vnc_spaces.PointerEvent(xc, yc, buttonmask=1),      # click
                vnc_spaces.PointerEvent(xc, yc, buttonmask=0)       # release
            ]
        else:
            key = self.merged_actions[i]
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

class SoftmaxCopyPasteTaskWithOrder(SoftmaxClickTask):
    """
    Creates a Discrete action space of mouse clicks and the keyboard input ctrl+a, ctrl+c, ctrl+v.
    There are 3 different kind of actions performable:
                1. a mouse click + ctrl-a + ctrl-c (aimed for the text field)
                2. a mouse click + ctrl-v (aimed for the empty field)
                3. a mouse click (aimed for the submit button)
    This class always executes this actions in that exact order to increase the chance to succeed.
    """
    def __init__(self, env, active_region=(10, 75 + 50, 10 + 160, 75 + 210), discrete_mouse_step=10, noclick_regions=[], noAgent=False):
        super(SoftmaxCopyPasteTaskWithOrder, self).__init__(env, active_region, discrete_mouse_step, noclick_regions, noAgent)
        logger.info('SoftmaxCopyPasteTaskWithOrder was used')
        self._action_code = 0

    def _discrete_to_action(self, i):
        if self.noAgent:
            return []
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

class SoftmaxMathTask(SoftmaxClickTask):
    """
    Creates a discrete action space of mouse clicks and the keyboard input of a number.
    There are 2 possible actions performable:
            1. a mouse click
            2. a number typed by the keyboard
    The applicable numbers depend on the specified environment. They are placed between -99 & 99.
    """
    def __init__(self, env, active_region=(10, 75 + 50, 10 + 160, 75 + 210), discrete_mouse_step=10, noclick_regions=[], noAgent=False):
        super(SoftmaxMathTask, self).__init__(env, active_region, discrete_mouse_step, noclick_regions, noAgent)
        logger.info('SoftmaxMathTask was used')
        if env.spec.id == 'wob.mini.SimpleAlgebra-v0':
            self._keys = list(range(-99, 100))
        elif env.spec.id == 'wob.mini.SimpleArithmetic-v0':
            self._keys = list(range(-9, 100))
        elif env.spec.id == 'wob.mini.VisualAddition-v0':
            self._keys = list(range(0, 21))
        self.merged_actions = list(self._keys) + self._points
        self.action_space = gym.spaces.Discrete(len(self.merged_actions))

    def _discrete_to_action(self, i):
        if self.noAgent:
            return []
        else:
            if type(self.merged_actions[i]) == tuple:
                xc, yc = self.merged_actions[i]
                return [
                    # Click in text field or on submit button
                    vnc_spaces.PointerEvent(xc, yc, buttonmask=0),
                    vnc_spaces.PointerEvent(xc, yc, buttonmask=1),
                    vnc_spaces.PointerEvent(xc, yc, buttonmask=0)
                ]
            else:
                key = self.merged_actions[i]
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

class SoftmaxMathTaskDirectSubmit(vectorized.ActionWrapper):
    """
    Creates a discrete action space of mouse clicks and the keyboard inputs of numbers.
    The numbers are specified for the different environments and lie between -99 & 99.
    The click inside the text field and on the submit button are hardcoded to lay the focus on the selection of the correct number.
    Therefore this class simulates the use of a different reward function assigning rewards immediately after inserting a number.
    """
    def __init__(self, env, noAgent=False):
        super(SoftmaxMathTaskDirectSubmit, self).__init__(env)
        logger.info('SoftmaxMathTaskDirectSubmit was used')
        self.noAgent = noAgent
        if env.spec.id == 'wob.mini.SimpleAlgebra-v0':
            self._keys = list(range(-99, 100))
        elif env.spec.id == 'wob.mini.SimpleArithmetic-v0':
            self._keys = list(range(-9, 100))
        elif env.spec.id == 'wob.mini.VisualAddition-v0':
            self._keys = list(range(0, 21))
        self.action_space = gym.spaces.Discrete(len(self._keys))

    def _action(self, action_n):
        return [self._discrete_to_action(int(i)) for i in action_n]

    def _discrete_to_action(self, i):
        if self.noAgent:
            return []
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

class SoftmaxFullLettersAndMouse(SoftmaxClickTask):
    """
    Creates a discrete action space of mouse clicks and the keyboard inputs of letters.
    It is possible to make use of all letters of the alphabet in upper and lower case.
    Either the agent chooses to click somewhere (i.e. the text field or the submit button) or to write a letter.
    """
    def __init__(self, env, active_region=(10, 75 + 50, 10 + 160, 75 + 210), discrete_mouse_step=10, noclick_regions=[], noAgent=False):
        super(SoftmaxFullLettersAndMouse, self).__init__(env, active_region, discrete_mouse_step, noclick_regions, noAgent)
        logger.info('SoftmaxFullLettersAndMouse was used')
        numKeys = list(range(0, 10))
        letterKeys = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
                      'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        self._keys = numKeys + letterKeys
        self.merged_actions = self._keys + self._points
        self.action_space = gym.spaces.Discrete(len(self.merged_actions))

    def _discrete_to_action(self, i):
        if self.noAgent:
            return []
        else:
            if type(self.merged_actions[i]) == tuple:
                xc, yc = self.merged_actions[i]
                # Click in text field or on submit button
                return [
                    vnc_spaces.PointerEvent(xc, yc, buttonmask=0),
                    vnc_spaces.PointerEvent(xc, yc, buttonmask=1),
                    vnc_spaces.PointerEvent(xc, yc, buttonmask=0)
                ]
            elif type(self.merged_actions[i]) == int:
                num = self.merged_actions[i]
                # Enter number
                return [
                    vnc_spaces.KeyEvent.by_name(str(num), down=True),
                    vnc_spaces.KeyEvent.by_name(str(num), down=False)
                ]
            else:
                letter = self.merged_actions[i]
                # Enter letter
                return [
                    vnc_spaces.KeyEvent.by_name(str(letter), down=True),
                    vnc_spaces.KeyEvent.by_name(str(letter), down=False)
                ]