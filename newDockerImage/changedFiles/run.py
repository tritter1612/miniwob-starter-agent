#!/usr/bin/env python
import threading
import argparse
import manhole
import random
import time
import six
import sys
import os
import json
import traceback
if six.PY2:
    import Queue as queue
    import urlparse
else:
    import queue
    import urllib.parse as urlparse
import re

import universe
from universe.rewarder import remote
from universe import twisty, wrappers

import socket

from backend.server import MOCK_PORT, MockServer, ioloop
from config import global_registry, WEBDRIVER_DEVICES

from wob import ProxyController
from wob.rewarders import lexicalize_template
from wob.dom import DOMParser, safe_execute
import wob.s3 as s3

# -----------------------------------------------------------------------------
# Logging and setup
# -----------------------------------------------------------------------------
import logging
logger = logging.getLogger()
logger.info_blue = lambda msg, *args, **kwargs: logger.info('\033[94m%s\033[0m' % msg, *args, **kwargs)
logger.info_green = lambda msg, *args, **kwargs: logger.info('\033[32m%s\033[0m' % msg, *args, **kwargs)
# now lilac, as gray on black is not readable
logger.info_gray = lambda msg, *args, **kwargs: logger.info('\033[35m%s\033[0m' % msg, *args, **kwargs)
# new to immediately spot latest changes
logger.info_red = lambda msg, *args, **kwargs: logger.info('\033[31m%s\033[0m' % msg, *args, **kwargs)
logger.setLevel(logging.WARN)

twisty.start_once()
# selenium can be very verbose, calm it down
# (see http://stackoverflow.com/questions/23407142/how-do-i-reduce-the-verbosity-of-chromedriver-logs-when-running-it-under-seleniu)
from wob.chrome import WebDriver as Chrome
from selenium import webdriver
from selenium.webdriver.remote.remote_connection import LOGGER
LOGGER.setLevel(logging.WARNING)


# -----------------------------------------------------------------------------
# Env controller thread.
# -----------------------------------------------------------------------------
class EnvController(threading.Thread):
    daemon = True

    def __init__(self, env_status, agent_conn, error_buffer,
                 control_buffer, mode='DATA', fps=60):
        super(EnvController, self).__init__(name='EnvController')
        self.cv = threading.Condition()
        self.browser = None
        self.server = None
        self.init_browser = False
        self.init_server = False
        self.init_reset = False
        self.mode = mode
        self.fps = fps
        self.setting = {}

        self.env_status = env_status
        self.episode_config = {}
        self.agent_conn = agent_conn
        self.error_buffer = error_buffer
        self.control_buffer = control_buffer

        # variables for iterating over miniwob envs, when env_id is wob.MiniWob
        self.miniwob_envs = [k for k,v in global_registry.items() if type(v) == str and 'miniwob' in v]
        random.shuffle(self.miniwob_envs)
        self.miniwob_pointer = 0 # point to the next miniwob env to load
        print('[EnvController] found %d miniwob envs' % (len(self.miniwob_envs)), )

        self.load_env()

    def load_env(self):
        env_id = self.env_status.env_id

        if env_id == 'wob.MiniWorldOfBits-v0':
            # special case, cycle through envs
            assert len(self.miniwob_envs) > 0, 'There were 0 miniwob envs detected?! See the EnvController code'
            wob_env_id = self.miniwob_envs[self.miniwob_pointer]
            self.miniwob_pointer += 1
            if self.miniwob_pointer >= len(self.miniwob_envs):
                self.miniwob_pointer = 0 # wrap around
            registry_item = global_registry[wob_env_id]
        else:
            registry_item = global_registry[env_id]


        # parse url.
        # setting is a dict with the following fields.
        #   - reload: if set to True, will reload webpage whenever task resets.
        if type(registry_item) == str:
            # miniwob: javascript only mini enviroments.
            self.url = registry_item
            parsed_url = urlparse.urlparse(self.url)
            self.setting = {
                'scheme': 'miniwob',
                'path': parsed_url.path,
                'www': 'static',
                'server': None,
                'reload': False
            }
        elif type(registry_item) == dict and registry_item['type'] == 'mockwob':
            # mockwob: mini enviroments + a mock backend in tornado.
            self.parsed_url = registry_item
            self.url = 'http://localhost:' + str(MOCK_PORT)
            self.setting = {
                'scheme': 'mockwob',
                'path': '/',
                'www': registry_item['www'],
                'server': registry_item['server'],
                'reload': registry_item.get('reload', False)
            }
        elif type(registry_item) == dict and registry_item['type'] == 'realwob':
            # realwob: real environments with truly real websites.
            self.parsed_url = registry_item
            self.url = registry_item['www']
            self.setting = {
                'scheme': 'realwob',
                'path': '/',
            }
            self.setting.update(registry_item)
        else:
            raise ValueError('unknown registry entry type ' + str(registry_item))

    def run(self):
        try:
            self.do_run()
        except Exception as e:
            self.error_buffer.record(e)

    def do_run(self):
        try:
            # initialize the mock server if necessary.
            self.launch_server()

            # initialize the browser
            self.launch_browser()

            # reset env.
            self.reset()
            self.init_reset = True

        except Exception as e:
            self.error_buffer.record(e)

        # and loop
        try:
            while True:
                self.process_control_messages()
                self.agent_conn.check_status()
                time.sleep(1. / self.fps)

                with self.cv:
                    self.cv.wait(timeout=1)

                if self.env_status.env_state == 'resetting':
                    self.reset()
        except KeyboardInterrupt:
            logger.warn('keyboard interrupt in thread')
            try:
                if self.server:
                    self.server.close()
                if self.rewarder:
                    self.rewarder.close()
            except Exception as e:
                self.error_buffer.record(e)

    def launch_server(self):
        if self.setting['scheme'] == 'miniwob':
            self.server = None
            self.rewarder = None

        elif self.setting['scheme'] == 'realwob':
            logger.info('launching cache proxy server...')
            if self.setting.get('rewarder'):
                self.rewarder = self.setting['rewarder'](self.mode)
                logger.warn('initializing %s', self.setting['rewarder'](self.mode))
                rewarders = [self.rewarder]
            else:
                self.rewarder = None
                rewarders = []

            # set up cache server.
            self.server = ProxyController(mode=self.mode,
                                          cache_name=self.setting['db'],
                                          rewarders=rewarders)
            self.server.start()
            logger.info_green('Cache proxy server started - mode = %s', self.mode)

        self.init_server = True

    def launch_browser(self):
        if self.env_status.env_state != 'resetting':
            self.env_status.set_env_info('resetting')

        logger.info('Launching new Chrome process...')
        chrome_options = webdriver.ChromeOptions()

        chrome_options.add_argument('--disable-infobars')
        chrome_options.add_argument('--disable-application-cache')
        chrome_options.add_argument('--disable-web-security')
        chrome_options.add_argument('--ignore-certificate-errors')
        chrome_options.add_argument('--disable-notifications')

        if self.setting['scheme'] == 'realwob':
            # disable browser cache.
            chrome_options.add_argument('--proxy-server=0.0.0.0:8888')
            # chrome extensions: loading - block UI while loading, ublock -
            # block analytics.
            chrome_options.add_argument('--load-extension=chrome/extension/ublock,'
                                        'chrome/extension/loading,'
                                        'chrome/extension/check404,' # noop.
                                        'chrome/extension/background-tab')
            chrome_options.add_argument('--user-data-dir=/tmp/profile')
            if self.setting.get('device'):
                logger.info_green('Device set to be %s', self.setting['device'])
                chrome_options.add_experimental_option('mobileEmulation',
                                                       WEBDRIVER_DEVICES[self.setting['device']])
            chrome_options.add_experimental_option( "prefs", {'profile.default_content_setting_values.geolocation': 1})
            chrome_options.add_experimental_option( "prefs", {'': 1})


        # start browser.
        env = dict(os.environ)
        print('FAKETIME = ', self.setting.get('faketime', ''))
        env.update({'FAKETIME': self.setting.get('faketime', '')})
        self.browser = Chrome(chrome_options=chrome_options, env=env)
        self.browser.command_executor._conn.timeout = 5 # This is needed so that selenium doesn't hang forever if someone exited the chrome tab.
        self.browser.set_page_load_timeout(30)          # seconds timeout for pages.

        # tell rewarder about browser.
        if self.rewarder:
            self.rewarder.init_browser(self.browser)

        self.init_browser = True
        logger.info_green('Chrome browser launched')

    def process_control_messages(self):
        while True:
            try:
                type, payload = self.control_buffer.get(block=False)
            except queue.Empty:
                break
            else:
                if type == 'rpc':
                    context, message = payload
                    self.process_rpc(context, message)
                elif type == 'client_disconnect': pass
                else:
                    assert False, 'Unrecogized type: {}'.format(type)

    def process_rpc(self, context, message):
        if message['method'] == 'v0.env.reset':
            env_id = message['body']['env_id']
            episode_id = message['headers']['episode_id']
            episode_config = message['body'].get('episode_config')
            episode_config = episode_config if episode_config else {}
            logger.warn('episode config = %s', str(episode_config))

            if env_id not in global_registry and env_id != 'wob.MiniWorldOfBits-v0':
                self.agent_conn.send_reply_error(
                    message="No server-side registration for {}. (HINT: This is the runtime for World of Bits. Perhaps you tyop'd the ID or it's meant for a different runtime.)".format(env_id),
                    parent_message_id=message['headers']['message_id'],
                    parent_context=context,
                )
                return

            # TODO: validate env_id
            old_env_id = self.env_status.env_id
            env_info = self.env_status.set_env_info('resetting', env_id=env_id, bump_past=episode_id)

            if old_env_id != env_id or env_id == 'wob.MiniWorldOfBits-v0':
                old_setting = dict(self.setting)
                self.load_env()
                logger.info('Updated setting = %s', str(self.setting))
                # restart browser if device and scheme changes.
                if self.setting['scheme'] == 'realwob':
                    logger.info('Restarting browser because env_id changed')

                    while not self.init_server:
                        logger.info('ProcessRPC thread is waiting for the server instance...')
                        time.sleep(1.)

                    while not self.init_browser:
                        logger.info('ProcessRPC thread is waiting for the browser instance...')
                        time.sleep(1.)

                    if self.browser:
                        self.browser.quit()
                    if self.server:
                        self.server.shutdown()

                    self.init_browser = False
                    self.init_server = False
                    self.launch_server()
                    self.launch_browser()

            self.trigger_reset(episode_config)

            self.agent_conn.send_reply_env_reset(
                parent_message_id=message['headers']['message_id'],
                parent_context=context,
                episode_id=self.env_status.episode_id,
            )
        else:
            self.agent_conn.send_reply_error(
                'Unsupported RPC method: {}'.format(message['method']),
                parent_message_id=message['headers']['message_id'],
                parent_context=context
            )

    def trigger_reset(self, episode_config={}):
        # logger.info('Triggering a reset on EnvController {}'.format(str(episode_config)))
        with self.cv:
            self.episode_config = episode_config
            self.env_status.set_env_info('resetting')
            self.cv.notify()

    def reset(self):
        env_info = self.env_status.env_info()  # Be sure to only call this once, as it can change from under us.
        assert env_info['env_state'] == 'resetting', 'Env state should be resetting, but is instead: {}'.format(env_info['env_state'])
        self.agent_conn.send_env_describe_from_env_info(env_info)

        # change the url/setting/etc potentially based on env_status
        self.load_env()

        # restart backend server if exists.
        if self.server:
            logger.info('Restarting HTTP server')
            self.server.reset()

        # clear all cookies.
        while True:
            try:
                self.browser.delete_all_cookies()
                break
            except:
                logger.warn('Trying to delete cookie')
                time.sleep(1.)

        # point browser to target url.
        # first set to blank to make sure user do not generate redictions.
        if self.setting['scheme'] == 'realwob': self.browser.get('about:blank')
        while True:
            try:
                if self.browser.current_url != self.url or self.setting.get('reload'):
                    logging.info('Browser changing url to ' + self.url)
                    self.browser.get(self.url)
                    # self.browser.execute_script('window.location.replace("{}")'.format(self.url))
                break
            except socket.timeout:
                logger.warn('Browser reset timeout, stop.')
                break
            except Exception as e:
                logger.warn('Browser reset failed: %s', str(e))
                time.sleep(1.)
                #self.error_buffer.record(e)

        # Wait for browser to start loading url.
        while True:
            try:
                if self.browser.current_url != 'about:blank': break
            except Exception as e:
                pass

            time.sleep(1.)

        # Wait for browser to finish loading url.
        while True:
            dom_state = safe_execute(self.browser, 'return document.readyState;')
            logger.info_gray('Waiting for document to load: %s', dom_state)
            if dom_state == 'complete':
                break
            time.sleep(1.)

        # TODO: hack. pause for a few seconds to let RewarderThread know env
        # is being reset.
        time.sleep(.5)
        with self.cv:
            env_info = self.env_status.set_env_info('running')
            self.cv.notifyAll()

    def close(self):
        # close env server.
        if self.server:
            self.server.close()
            self.server = None
        # close rewarder
        if self.rewarder:
            self.rewarder.close()
            self.rewarder = None
        # close browser.
        if self.browser:
            self.browser.close()
            self.browser = None


# -----------------------------------------------------------------------------
# Rewarder thread.
# -----------------------------------------------------------------------------
class RewarderThread(threading.Thread):
    """
    The job the Rewarder thread is to periodically (e.g. at 60Hz) communicate
    {reward, done, info} over the agent_conn.
    """
    daemon = True

    def __init__(self, env_status, agent_conn, env_controller, error_buffer, fps=60):
        logger.info("RewarderThread initialized")

        super(RewarderThread, self).__init__(name='RewarderThread',)
        self.agent_conn = agent_conn
        self.env_controller = env_controller
        self.env_status = env_status
        self.error_buffer = error_buffer
        self.dom_parser = DOMParser()
        self.fps = fps # at what fps to run

    def run(self):
        try:
            self.do_run()
        except Exception as e:
            self.error_buffer.record(e)

    def _read_client_done(self, browser):
        done = safe_execute(browser, """ try { return WOB_DONE_GLOBAL; } catch(err) { return false; } """, default=False)
        return done

    def _read_server_done(self, server):
        if server:
            return server.WOB_DONE_GLOBAL
        else:
            return False

    def _read_client_reward(self, browser):
        ''' read reward off the client (javascript) and set the original value to zero.
        '''
        # TODO: this is not atomic right?
        ## async implementation
        #info = safe_execute(browser, """   var callback = arguments[arguments.length - 1];
        #                                   info = {'reward': WOB_REWARD_GLOBAL,
        #                                           'done': WOB_DONE_GLOBAL,
        #                                           'readyState': document.readyState};
        #                                   WOB_REWARD_GLOBAL = 0;
        #                                   callback(info);
        #                               """, default={}, async=True)
        info = safe_execute(browser, """info = {};
                                        try {
                                             info = {'reward': WOB_REWARD_GLOBAL,
                                                     'done': WOB_DONE_GLOBAL};
                                             WOB_REWARD_GLOBAL = 0;
                                        }catch(err) {}
                                        return info;
                                       """, default={}, async=False)
        if not info:
            logger.info_red('info object not set, returning done=False!!!')
            return (0., False)
        reward = info.get('reward', 0.)
        done = info.get('done')
        if done is None:
            logger.info_red('info object.done not set, returning done=False!!!')
            done = False
        if not done and reward != 0:
            logger.info_red('not done but reward %f ! SHOULD NOT HAPPEN!!!', reward)
        elif reward != 0:
            logger.info_green('done and reward %f', reward)
        return (reward, done)

    def _read_server_reward(self, server):
        ''' read reward off the server and set the original value to zero.
        '''
        if server is None: # serverless env. like MiniWoB
            return (0, False)

        server.WOB_LOCK.acquire()
        reward = server.WOB_REWARD_GLOBAL
        done = self._read_server_done(server)
        server.WOB_REWARD_GLOBAL = 0
        server.WOB_LOCK.release()
        return (reward, done)

    def reset(self):
        # logger.info('RewarderThread is resetting')
        # wait for server to come online
        while not self.env_controller.init_server:
            logger.info_gray('RewarderThread is waiting for the server instance...')
            time.sleep(1.)

        # wait for browser to come online
        while not self.env_controller.init_browser:
            logger.info_gray('RewarderThread is waiting for the browser instance...')
            time.sleep(1.)

        # wait for first reset to be done.
        while not self.env_controller.init_reset:
            logger.info_gray('RewarderThread is waiting for first reset to finish...')
            time.sleep(1.)

        # reset rewarder.
        while self.env_controller.rewarder:
            try:
                self.env_controller.rewarder.reset()
                break
            except Exception as e:
                logger.warn('RewarderThread reset failed. Trying again. %s', str(e))
                traceback.print_exc()
                time.sleep(1.)

        while self._read_client_done(self.env_controller.browser):
            logger.warn('RewarderThread is waiting for browser to reset')
            time.sleep(0.25)

        while self._read_server_done(self.env_controller.server):
            logger.warn('RewarderThread is waiting for proxy server to reset')
            time.sleep(0.25)

        # set up instructions.
        # for miniwob, the instruction is set on the client side.
        if self.env_controller.setting['scheme'] == 'realwob' and self.env_controller.rewarder:
            query_id = self.env_controller.episode_config.get('query_id')

            if self.env_controller.rewarder and query_id: self.env_controller.rewarder.set_instruction_by_id(query_id)

        env_info = self.env_controller.env_status.set_env_info('running')
        # logger.warn('Rewarder sent running')
        self.env_controller.agent_conn.send_env_describe_from_env_info(env_info)

        logger.warn('Instruction %s generated %s', self.instruction_id,
                    self.instruction)

    @property
    def instruction(self):
        if self.env_controller.setting['scheme'] == 'realwob':
            if not self.env_controller.rewarder:
                return None
            else:
                return self.env_controller.rewarder.instruction
        elif self.env_controller.setting['scheme'] == 'miniwob' and self.env_controller.browser:
            return safe_execute(self.env_controller.browser, "return document.getElementById('query').innerText;")
        else:
            raise ValueError('Unknown scheme type {}'.format(self.env_controller.setting['scheme']))

    @property
    def instruction_id(self):
        if self.env_controller.setting['scheme'] == 'realwob' and self.env_controller.rewarder:
            return self.env_controller.rewarder.instruction_id
        else:
            return ''

    def do_run(self):
        logger.info('RewarderThread has started')
        self.reset()

        # start the main loop
        t0 = time.time()
        n = 0
        trigger_reset = False
        prev_blocks = {}

        while True:
            # timing/sleeping computations to pursue fixed self.fps to best of our ability
            n += 1
            t = t0 + n * 1.0 / self.fps
            dt = t - time.time()
            if dt > 0:
                time.sleep(dt)
            elif self.env_controller.mode == 'ENV':
                logger.info_gray('RewarderThread falling behind %f', dt)

            # send env text.
            element_id = 'wrap' if self.env_controller.setting['scheme'] == 'miniwob' else None
            blocks = self.dom_parser.parse(self.env_controller.browser, element_id=element_id)
            if blocks and blocks != prev_blocks:
                if not prev_blocks: # first block recieved.
                    logger.info_blue('Query %s: %s', self.instruction_id, self.instruction)
                    meta = self.env_controller.setting.get('meta', {})
                    if 'question' in meta:
                        logger.info_blue('Template %s', str(meta['question']))
                        logger.info_blue('Query in natural language: %s',
                                         lexicalize_template(meta['question'], self.instruction))

                #print({
                #    'meta': self.env_controller.setting.get('meta', {}),
                #    'blocks': blocks,
                #    'instruction_id': self.instruction_id,
                #    'instruction': self.instruction
                #})

                prev_blocks = blocks
                self.agent_conn.send_env_text({
                    'meta': self.env_controller.setting.get('meta', {}),
                    'blocks': blocks,
                    'instruction_id': self.instruction_id,
                    'instruction': self.instruction
                }, episode_id=self.env_status.episode_id)

            # interact with the browser/server instance to read off current rewards and done flag
            (reward_client, done_client) = self._read_client_reward(self.env_controller.browser)
            (reward_server, done_server) = self._read_server_reward(self.env_controller.server)
            if reward_client is None: reward_client = 0
            if reward_server is None: reward_server = 0
            reward = reward_client + reward_server
            done = done_client or done_server # if one of them quit, the env ends.

            # send auxliary done text.
            if done:
                if self.env_controller.setting['scheme'] == 'miniwob':
                    self.agent_conn.send_env_text({'info': 'done! reward=%0.2f' % reward}, episode_id=self.env_status.episode_id)
                elif self.env_controller.setting['scheme'] == 'realwob' and self.env_controller.mode == 'ENV':
                    self.agent_conn.send_env_text({'info': 'done! reward=%0.2f' % reward}, episode_id=self.env_status.episode_id)
                elif self.env_controller.setting['scheme'] == 'realwob' and self.env_controller.mode == 'DATA':
                    self.agent_conn.send_env_text({'info': 'Done, good job!'}, episode_id=self.env_status.episode_id)

            # only log transmission if event is complete. some tasks might emit
            # continuous/multiple rewards in a single episode.
            if done and reward != 0:
                logger.info_green('Sending reward to agent: reward=%0.2f done=%s', reward, done)
                self.agent_conn.send_env_reward(reward, done, {}, episode_id=self.env_status.episode_id)
            elif reward != 0:
                self.agent_conn.send_env_reward(reward, done, {}, episode_id=self.env_status.episode_id)

            if done: self.env_controller.trigger_reset()

            # TODO: this assumes that rewarder ticks much faster than env reset.
            while self.env_controller.env_status.env_state == 'resetting':
                trigger_reset = True
                # logger.info_gray('RewarderThread waiting for env to be reset')
                time.sleep(0.1)

            if trigger_reset:
                self.reset()
                n = 0
                t0 = time.time()
                trigger_reset = False
                logger.info_gray('RewarderThread wait completed: env resetted')
                prev_blocks = {}

class IOThread(threading.Thread):
    daemon = True

    def __init__(self, env_controller, error_buffer):
        super(IOThread, self).__init__()
        self.env_controller = env_controller
        self.error_buffer = error_buffer

    def run(self):
        try:
            ioloop.IOLoop.current().start()
        except KeyboardInterrupt:
            logger.warn('keyboard interrupt in thread')
            try:
                self.env_controller.close()
            except Exception as e:
                self.error_buffer.record(e)
            ioloop.IOLoop.current().stop()

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():

    # command line option handling
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-e', '--env_id', default='wob.mini.ClickTest-v0', help='env id')
    parser.add_argument('-v', '--verbose', action='count', dest='verbosity', default=0, help='Set verbosity.')
    parser.add_argument('-m', '--mode', default='DATA', help='mode (DATA | ENV | DEMO)')
    parser.add_argument('-f', '--fps', default=5, type=int, help='Number of frames per second')
    parser.add_argument('-i', '--idle-timeout', type=float, help='How long to keep the environment around when it has no active connections')
    parser.add_argument('--rewarder-port', type=int, default=15900, help='Which port to start the agent_conn thread')
    args = parser.parse_args()
    print(args)

    # logging and setup
    if args.verbosity == 0:
        logger.setLevel(logging.INFO)
    elif args.verbosity >= 1:
        logger.setLevel(logging.DEBUG)
        logger.info("Starting world of bits run.py with: %s", sys.argv)

    error_buffer = universe.utils.ErrorBuffer()

    # Jot down the env_id so the uploader can find it later
    env_id_file_dir = os.path.join(os.sep, 'tmp', 'demo')
    env_id_file_path = os.path.join(env_id_file_dir, 'env_id.txt')
    if not os.path.exists(env_id_file_dir):
        logger.info("[world-of-bits] Creating directory %s", env_id_file_dir)
        os.makedirs(env_id_file_dir)

    try:
        with open(env_id_file_path,'w') as env_id_file:
            logger.info("[world-of-bits] Writing env id to file %s", env_id_file_path)
            env_id_file.write(args.env_id)
            env_id_file.write('\n')
    except PermissionError:
        logger.info("[world-of-bits] could not write env id to " + env_id_file_path + " due to a permission error. skipping.")
        pass

    # create connection to the agent
    env_status = universe.rewarder.EnvStatus()
    env_status.set_env_info(env_id=args.env_id, fps=args.fps)
    cv = threading.Condition()
    control_buffer = remote.ControlBuffer(cv)
    agent_conn = remote.AgentConn(env_status, cv, control_buffer, error_buffer=error_buffer, idle_timeout=args.idle_timeout)
    agent_conn.listen(port=args.rewarder_port)

    # start up the environment controller
    env_controller = EnvController(env_status, agent_conn, error_buffer,
                                   control_buffer, args.mode, fps=args.fps)
    env_controller.start()

    # start up the rewarder
    rewarder = RewarderThread(env_status, agent_conn, env_controller, error_buffer, fps=args.fps)
    rewarder.start()

    # run the iothread
    iothread = IOThread(env_controller, error_buffer)
    iothread.start()

    # Debugging tool
    manhole.install(locals={'rewarder': rewarder, 'env_controller': env_controller, 'agent_conn': agent_conn})

    while True:
        try:
            error_buffer.blocking_check(timeout=60)
        except remote.Exit as e:
            logger.info('%s', e)
            return 0

if __name__ == '__main__':
  sys.exit(main())
