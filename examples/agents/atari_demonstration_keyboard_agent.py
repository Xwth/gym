#!/usr/bin/env python
import argparse
import logging
import pyglet
import time
import sys

import gym

#
# Demonstration capture for Atari games
#
from gym.monitoring import demonstrations
from gym.monitoring.demonstrations import DemonstrationRecorder

logger = logging.getLogger()

class AtariDemonstration(object):
    def __init__(self, env_id, outfile):
        self.env = gym.make(env_id)
        self.outfile = outfile

        self.rollout_time = 10*60*30 # 10 minutes at 30 fps
        self.human_agent_action = 0
        self.human_wants_restart = False
        self.human_sets_pause = False
        self.human_speed_boost = 0

        self.env.render()
        self.env.viewer.window.on_key_press = self.key_press
        self.env.viewer.window.on_key_release = self.key_release

    def key_press(self, key, mod):
        logger.info("Pressed: %s", key)

        if key==pyglet.window.key.O:
            self.human_wants_restart = True
        if key==pyglet.window.key.SPACE:
            self.human_sets_pause = not self.human_sets_pause
        if key==pyglet.window.key.SEMICOLON:
            self.human_speed_boost = 20
        if key==pyglet.window.key.L:
            self.human_speed_boost = 10

        if key == pyglet.window.key.UP: # up arrow
            self.human_agent_action = 2
        if key == pyglet.window.key.DOWN: # down arrow
            self.human_agent_action = 3

    def key_release(self, key, mod):
        logger.info("Released: %s", key)
        self.human_agent_action = 0

    def rollout(self):
        self.human_wants_restart = False
        ob = self.env.reset()

        recorder = DemonstrationRecorder(self.outfile)

        for t in range(self.rollout_time):

            a = self.human_agent_action
            logger.info("Action: %s", a)

            # Record o[t] -> a[t] (that is, action is the *label* for the
            # observation)
            recorder.record_step(a, ob)
            ob, reward, done, info = self.env.step(a)

            self.env.render()

            if self.human_speed_boost > 0:
                # Boost because the user asked for it
                self.human_speed_boost -= 1
            else:
                # Slow down the game to make it easier for me to play
                time.sleep(0.16)

            if done: break
            if self.human_wants_restart: break
            while self.human_sets_pause:
                self.env.render()
                time.sleep(0.1)

        recorder.close()

    def run(self):
        logger.info("""INSTRUCTIONS:

- Press up/down to control the paddle
- Press the semicolon key (s on Dvorak) to speed up for the next 20 frames
- Press the l key (n on Dvorak) to speed up for the next 10 frames
- Press the o key (r on Dvorak) to reset the episode. The current episode will still have been recorded.
- Press the space key to pause the game, and space again to unpause.
""")

        while True:
            self.rollout()


def main():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-v', '--verbose', action='count', dest='verbosity', default=0, help='Set verbosity.')
    parser.add_argument('-e', '--env-id', default='Pong-v0', help='Which environment to run.')
    parser.add_argument('-o', '--outfile', default='/tmp/atari.demo', help='Where to write demo file.')
    args = parser.parse_args()

    if args.verbosity == 0:
        logger.setLevel(logging.INFO)
    elif args.verbosity >= 1:
        logger.setLevel(logging.DEBUG)

    atari = AtariDemonstration(args.env_id, args.outfile)
    atari.run()

    return 0

if __name__ == '__main__':
    sys.exit(main())
