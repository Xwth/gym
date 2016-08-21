import base64
import gzip
import json
import numpy as np
import pickle

import gym
from gym.utils import closer

recorder_closer = closer.Closer()

class DemonstrationRecorder(object):
    def __init__(self, env, file_name):
        self._id = recorder_closer.register(self)
        self.env = env
        self.i = 0

        self.file = gzip.GzipFile(file_name, 'w')
        self.file.write(json.dumps({'metadata': True, 'env_id': env.spec.id}))
        self.file.write("\n")

    def record_step(self, observation, reward, done, info, action):
        observation = self.env.observation_space.to_jsonable(observation)
        self.file.write(json.dumps({'i': 0, 'action': action, 'reward': reward, 'done': done, 'observation': observation, 'info': info}))
        self.file.write("\n")
        self.i += 1

    def close(self):
        self.file.close()
        recorder_closer.unregister(self._id)

    def __del__(self):
        self.close()

class DemonstrationReader(object):
    def __init__(self, file_name):
        self.file = gzip.GzipFile(file_name, 'rb')
        meta = self.file.readline()
        meta = json.loads(meta)
        self.env = gym.spec(meta['env_id']).make()

    def __iter__(self):
        return self

    def next(self):
        line = self.file.readline()
        if line == "":
            raise StopIteration()

        line = json.loads(line)
        observation = line['observation']
        reward = line['reward']
        done = line['done']
        info = line.get('info') # legacy
        action = line['action']

        observation = self.env.observation_space.from_jsonable(observation)
        return observation, reward, done, info, action


if __name__ == '__main__':
    import gym
    from gym.monitoring.demonstrations import DemonstrationReader

    env = gym.make('Pong-v0')
    recorder = DemonstrationRecorder(env, "/tmp/pong.demo.gz")
    ob = env.reset()
    reward = done = info = None
    action = 1
    for _ in range(10):
        recorder.record_step(ob, reward, done, info, action)
        ob, reward, done, info = env.step(action)
    recorder.close()

    reader = DemonstrationReader("/tmp/pong.demo.gz")
    for observation, reward, done, info, action in reader:
        print(action)
