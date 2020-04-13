import numpy as np
import random
import tensorflow as tf
import gym
import matplotlib.pyplot as plt
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from ml.SuperMarioEnvironment import SuperMarioEnv

class NeuralNetwork:
      def __init__(self, D):
          eta = 0.1

          self.W = tf.Variable(tf.random_normal(shape=(D, 1)), name='w')
          self.X = tf.placeholder(tf.float32, shape=(None, D), name='X')
          self.Y = tf.placeholder(tf.float32, shape=(None,), name='Y')

          # make prediction and cost
          Y_hat = tf.reshape(tf.matmul(self.X, self.W), [-1])
          err = self.Y - Y_hat
          cost = tf.reduce_sum(tf.pow(err,2))

          # ops we want to call later
          self.train_op = tf.train.GradientDescentOptimizer(eta).minimize(cost)
          self.predict_op = Y_hat

          # start the session and initialize params
          init = tf.global_variables_initializer()
          self.session = tf.Session()
          self.session.run(init)

      def train(self, X, Y):
          self.session.run(self.train_op, feed_dict={self.X: X, self.Y: Y})

      def predict(self, X):
          return self.session.run(self.predict_op, feed_dict={self.X: X})

class FeatureTransformer:
    def __init__(self, env):
      #obs_examples = np.array([env.observation_space.sample() for x in range(20000)])
        obs_examples = np.random.random((20000, 4))
        print(obs_examples.shape)
        scaler = StandardScaler()
        scaler.fit(obs_examples)

      # Used to converte a state to a featurizes represenation.
      # We use RBF kernels with different variances to cover different parts of the space
        featurizer = FeatureUnion([
              ("cart_position", RBFSampler(gamma=0.02, n_components=500)),
              ("cart_velocity", RBFSampler(gamma=1.0, n_components=500)),
              ("pole_angle", RBFSampler(gamma=0.5, n_components=500)),
              ("pole_velocity", RBFSampler(gamma=0.1, n_components=500))
              ])
        feature_examples = featurizer.fit_transform(scaler.transform(obs_examples))
        print(feature_examples.shape)

        self.dimensions = feature_examples.shape[1]
        self.scaler = scaler
        self.featurizer = featurizer

    def transform(self, observations):
        scaled = self.scaler.transform(observations)
        return self.featurizer.transform(scaled)

class Agent:
    def __init__(self, env, feature_transformer):
      self.env = env
      self.agent = []
      self.feature_transformer = feature_transformer
      #for i in range(len(env.action_space)):
      for i in range(6):
        nn = NeuralNetwork(self.feature_transformer.dimensions)
        self.agent.append(nn)

    def predict(self, s):
      X = self.feature_transformer.transform([s])
      return np.array([m.predict(X)[0] for m in self.agent])

    def update(self, s, a, G):
      X = self.feature_transformer.transform([s])
      self.agent[a].train(X, [G])

    def sample_action(self, s, eps):
      if np.random.random() < eps:
        return random.randint(0,5)
      else:
        return np.argmax(self.predict(s))

class MLPlayer:  
    def __init__(self):
        self.env = SuperMarioEnv() # 5 Actions, left right up left+up right+up. 4 Observations, x, xvelocity, y, yvelocity.
        self.ft = FeatureTransformer(self.env)
        self.agent = Agent(self.env, self.ft)
  # def play_one(self, env, agent, eps, gamma):
  #   self.obs = self.env.reset()
  #   done = False
  #   totalreward = 0
  #   iters = 0
  #   while not done and iters < 2000:
  #     action = self.agent.sample_action(self.obs, eps)
  #     prev_obs = self.obs
  #     self.obs, reward, done, info = env.step(action)

  #     if done:
  #       reward = -400

  #     # update the model
  #     next = agent.predict(obs)
  #     assert(len(next.shape) == 1)
  #     G = reward + gamma*np.max(next)
  #     agent.update(prev_obs, action, G)

  #     if reward == 1:
  #       totalreward += reward
  #     iters += 1

  #   return totalreward

  # def getPrediction(self):
  #   self.agent.sample_action(obs, eps)
