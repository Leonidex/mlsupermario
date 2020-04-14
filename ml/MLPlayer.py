import numpy as np
import tensorflow.compat.v1 as tf
import gym
import matplotlib.pyplot as plt
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from ml.SuperMarioEnvironment import SuperMarioEnv
import sys

class NeuralNetwork:
    def __init__(self, D, techniqueIdentifier, globalStep, learningRate, param1, param2):
        self.techniqueIdentifier = techniqueIdentifier
        self.learningRate = learningRate
        self.param1 = param1
        self.param2 = param2

        tf.disable_v2_behavior()

        self.W = tf.Variable(tf.random.normal(shape=(D, 1)), name='w')
        self.X = tf.placeholder(tf.float32, shape=(None, D), name='X')
        self.Y = tf.placeholder(tf.float32, shape=(None,), name='Y')

        # make prediction and cost
        Y_hat = tf.reshape(tf.matmul(self.X, self.W), [-1])
        err = self.Y - Y_hat

        powVar = tf.placeholder(tf.float32, shape=None, name="powVar")
        cost = tf.reduce_sum(tf.pow(err,2))

        # ops we want to call later
        optimized = self.optimize()
        self.train_op = optimized.minimize(err)
        self.predict_op = Y_hat

        # start the session and initialize params
        init = tf.global_variables_initializer()
        self.session = tf.Session()
        self.session.run(init)

    def train(self, X, Y):
        self.session.run(self.train_op, feed_dict={self.X: X, self.Y: Y})

    def predict(self, X):
        return self.session.run(self.predict_op, feed_dict={self.X: X})

    def optimize(self):
        # print("TENSORS-----------------------")
        # tf.Print("{0}",[self.learningRate])
        result = tf.train.GradientDescentOptimizer(self.learningRate)
        return result

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
              ("x", RBFSampler(gamma=0.5, n_components=500)),
              ("x_velocity", RBFSampler(gamma=0.5, n_components=500)),
              ("y", RBFSampler(gamma=0.5, n_components=500)),
              ("y_velocity", RBFSampler(gamma=0.5, n_components=500))
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
    def __init__(self, env, feature_transformer, techniqueIdentifier, globalStep, learningRate, param1, param2):
        self.env = env
        self.nodes = []
        self.feature_transformer = feature_transformer
        for i in range(env.action_space.n):
            nn = NeuralNetwork(self.feature_transformer.dimensions, techniqueIdentifier, globalStep, learningRate, param1, param2)
            self.nodes.append(nn)

    def predict(self, s):
        X = self.feature_transformer.transform([s])
        return np.array([m.predict(X)[0] for m in self.nodes])

    def update(self, s, a, G):
      X = self.feature_transformer.transform([s])
      self.nodes[a].train(X, [G])

    def sample_action(self, s, eps):
      if np.random.random() < eps:
        return np.random.randint(0,self.env.action_space.n)
      else:
        return np.argmax(self.predict(s))

class MLPlayer:
    def __init__(self, techniqueIdentifier, learningRate, globalStep, param1, param2):
        self.env = SuperMarioEnv()  # 5 Actions, left right up left+up right+up. 4 Observations, x, xvelocity, y, yvelocity.
        self.ft = FeatureTransformer(self.env)
        self.agent = Agent(self.env, self.ft, techniqueIdentifier, globalStep, learningRate, param1, param2)