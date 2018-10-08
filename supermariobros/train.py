import random
import typing

from gym_super_mario_bros import make
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
from numpy import argmax, float32, reshape, uint8
from numpy.random import rand
from skimage.color import rgb2gray
from skimage.transform import resize

Action = typing.Sequence[str]
#RGB = typing.Tuple[int, int, int]
#Screen = typing.Tuple[(RGB,) * 256]
#State = typing.Tuple[(Screen,) * 240]
RGB = typing.Tuple[int]
Screen = typing.Tuple[(RGB,) * 84]
State = typing.Tuple[(Screen,) * 84]

EPISODES = 1000

class Agent:
    def __init__(self, state_size: int, actions: typing.Sequence[Action]):
        self.actions = actions
        self.learning_rate = 0.01
        self.discount_factor = 0.9
        self.epsilon = 1.
        self.epsilon_min = 0.01
        self.epsilon_decay = .9999
        self.state_size = state_size
        self.model = self.build_model()

#        self.model.load_weights('./deep_sarsa.h5')

    def build_model(self) -> Sequential:
        model = Sequential()
        model.add(Dense(30, input_dim=self.state_size, activation='relu'))
        model.add(Dense(30, activation='relu'))
        model.add(Dense(len(self.actions), activation='linear'))
        model.summary()

        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def get_action(self, state: State) -> Action:
        if rand() <= self.epsilon:
            return random.randrange(len(self.actions))
        else:
            q_values = self.predict(state)
            return argmax(q_values[0])

    def train_model(self, state: State, action: Action, reward: int,
                    next_state: State, next_action: Action, done: bool):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        state = float32(state)
        next_state = float32(next_state)
        target = self.predict(state)[0]

        if done:
            target[action] = reward
        else:
            target[action] = (
                reward +
                self.discount_factor * self.predict(next_state)[0][next_action]
            )

        target = reshape(target, [1, len(self.actions)])
        self.model.fit(reshape(state, [1, self.state_size]), target, epochs=1, verbose=0)

    def predict(self, state: State) -> Action:
        return self.model.predict(reshape(state, [1, self.state_size]))

def downsample(observe, w = 84, h = 84):
    return uint8(resize(rgb2gray(observe), 
        (w, h), mode='constant') * 255)

if __name__ == '__main__':
    env = make('SuperMarioBros-v0')
    env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)

    agent = Agent(84*84, SIMPLE_MOVEMENT)
    scores, episodes = [], []

    for e in range(EPISODES):
        done = False
        state = env.reset()
        state = downsample(state)
        score = 0

        while not done:
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = downsample(next_state)            
            next_action = agent.get_action(next_state)


            agent.train_model(state, action, reward, next_state, next_action, done)
            state = next_state

            score += reward

            env.render()

            if done:
                scores.append(score)
                episodes.append(e)
                print(f"episode: {e}, score: {score}")

        if e % 100 == 0:
            agent.model.save_weights("./deep_sarsa.h5")

    env.close()