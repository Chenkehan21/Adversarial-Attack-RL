"""Abstract policy class and some concrete implementations."""

from gym.spaces import Box
import numpy as np
from stable_baselines.common.tf_layers import ortho_init
from stable_baselines.common.tf_util import seq_to_batch
from stable_baselines.common.distributions import DiagGaussianProbabilityDistribution
from stable_baselines.common.policies import ActorCriticPolicy, RecurrentActorCriticPolicy, register_policy
import tensorflow as tf


class RunningMeanStd(object):
    def __init__(self, scope="running", reuse=False, epsilon=1e-2, shape=()):
        with tf.variable_scope(scope, reuse=reuse):
            # We need these variables to be serialized/deserialized.
            # Stable Baselines reasonably assumes only trainable variables need to be serialized.
            # However, we do not want the optimizer to update these. In principle, we should
            # update these based on observation history. However, Bansal et al's open-source code
            # did not include support for this, and since they are unlikely to change much with
            # additional training I have not added support for this.
            # Hack: make them trainable, but use stop_gradients to stop them from being updated.
            self._sum = tf.stop_gradient(tf.get_variable(
                dtype=tf.float32,
                shape=shape,
                initializer=tf.constant_initializer(0.0),
                name="sum", trainable=True))
            self._sumsq = tf.stop_gradient(tf.get_variable(
                dtype=tf.float32,
                shape=shape,
                initializer=tf.constant_initializer(epsilon),
                name="sumsq", trainable=True))
            self._count = tf.stop_gradient(tf.get_variable(
                dtype=tf.float32,
                shape=(),
                initializer=tf.constant_initializer(epsilon),
                name="count", trainable=True))
            self.shape = shape

            self.mean = tf.to_float(self._sum / self._count)
            var_est = tf.to_float(self._sumsq / self._count) - tf.square(self.mean)
            self.std = tf.sqrt(tf.maximum(var_est, 1e-2))


def dense(x, size, name, weight_init=None, bias=True):
    w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=weight_init)
    ret = tf.matmul(x, w)
    if bias:
        b = tf.get_variable(name + "/b", [size], initializer=tf.zeros_initializer())
        return ret + b
    else:
        return ret


class GymCompetePolicy(ActorCriticPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, hiddens=None,
                 state_shape=None, scope="input", reuse=False, normalize=False):
        ActorCriticPolicy.__init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch,
                                   reuse=reuse, scale=False)
        self.hiddens = hiddens
        self.normalized = normalize
        self.weight_init = ortho_init(scale=0.01)
        self.observation_space = ob_space
        self.action_space = ac_space

        with self.sess.graph.as_default():
            with tf.variable_scope(scope, reuse=reuse):
                self.scope = tf.get_variable_scope().name

                assert isinstance(ob_space, Box)

                if self.normalized:
                    if self.normalized != 'ob':
                        self.ret_rms = RunningMeanStd(scope="retfilter")
                    self.ob_rms = RunningMeanStd(shape=ob_space.shape, scope="obsfilter")

                self.obz = self.processed_obs
                if self.normalized:
                    self.obz = tf.clip_by_value((self.processed_obs - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)

    def _setup_init(self):
        pdparam = tf.concat([self.policy, self.policy * 0.0 + self.logstd], axis=1)
        self._proba_distribution = DiagGaussianProbabilityDistribution(pdparam)
        super()._setup_init()

    def restore(self, params):
        with self.sess.graph.as_default():
            var_list = self.get_trainable_variables()
            shapes = list(map(lambda x: x.get_shape().as_list(), var_list))
            total_size = np.sum([int(np.prod(shape)) for shape in shapes])
            theta = tf.placeholder(tf.float32, [total_size])

            start = 0
            assigns = []
            for (shape, v) in zip(shapes, var_list):
                size = int(np.prod(shape))
                assigns.append(tf.assign(v, tf.reshape(theta[start:start + size], shape)))
                start += size

            op = tf.group(*assigns)
            self.sess.run(op, {theta: params})

    def get_trainable_variables(self):
        return self.sess.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)


class MlpPolicyValue(GymCompetePolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, hiddens=None,
                 scope="input", reuse=False, normalize=False):
        if hiddens is None:
            hiddens = [64, 64]
        super().__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, hiddens=hiddens,
                         scope=scope, reuse=reuse, normalize=normalize)
        self._initial_state = None
        with self.sess.graph.as_default():
            with tf.variable_scope(scope, reuse=reuse):
                def dense_net(prefix, shape):
                    last_out = self.obz
                    ff_outs = []
                    for i, hid_size in enumerate(hiddens):
                        h = dense(last_out, hid_size, f'{prefix}{i + 1}',
                                  weight_init=self.weight_init)
                        last_out = tf.nn.tanh(h)
                        ff_outs.append(last_out)
                    return dense(last_out, shape, f'{prefix}final',
                                 weight_init=self.weight_init), ff_outs

                self._value_fn, value_ff_acts = dense_net('vff', 1)
                if self.normalized and self.normalized != 'ob':
                    self._value_fn = self._value_fn * self.ret_rms.std + self.ret_rms.mean  # raw = not standardized

                self._policy, policy_ff_acts = dense_net('pol', ac_space.shape[0])
                self.ff_out = {'value': value_ff_acts, 'policy': policy_ff_acts}
                self.logstd = tf.get_variable(name="logstd", shape=[1, ac_space.shape[0]],
                                              initializer=tf.zeros_initializer())

                self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False, extra_op=None):
        action = self.deterministic_action if deterministic else self.action
        outputs = [action, self.value_flat, self.neglogp]
        if extra_op is not None:
            outputs.append(extra_op)
            a, v, neglogp, ex = self.sess.run(outputs, {self.obs_ph: obs})
            return a, v, self.initial_state, neglogp, ex
        else:
            a, v, neglogp = self.sess.run(outputs, {self.obs_ph: obs})
            return a, v, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        value = self.sess.run(self.value_flat, {self.obs_ph: obs})
        return value


class LSTMPolicy(GymCompetePolicy, RecurrentActorCriticPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, hiddens=None,
                 scope="input", reuse=False, normalize=False):
        if hiddens is None:
            hiddens = [128, 128]
        num_lstm = hiddens[-1]

        RecurrentActorCriticPolicy.__init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch,
                                           state_shape=(4, num_lstm), reuse=reuse)
        GymCompetePolicy.__init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch,
                                  hiddens=hiddens, scope=scope, reuse=reuse, normalize=normalize)

        with self.sess.graph.as_default():
            with tf.variable_scope(scope, reuse=reuse):
                self.state_out = []
                states = tf.transpose(self.states_ph, (1, 0, 2))

                def lstm(start, suffix):
                    # Feed forward
                    ff_out = self.obz
                    ff_list = []
                    for hidden in self.hiddens[:-1]:
                        ff_out = tf.contrib.layers.fully_connected(ff_out, hidden)
                        batch_ff_out = tf.reshape(ff_out, [self.n_env, n_steps, -1])
                        ff_list.append(batch_ff_out)
                    # Batch->Seq
                    input_seq = tf.reshape(ff_out, [self.n_env, n_steps, -1])
                    input_seq = tf.transpose(input_seq, (1, 0, 2))
                    masks = tf.reshape(self.dones_ph, [self.n_env, n_steps, 1])

                    # RNN
                    inputs_ta = tf.TensorArray(dtype=tf.float32, size=n_steps)
                    inputs_ta = inputs_ta.unstack(input_seq)

                    cell = tf.contrib.rnn.BasicLSTMCell(num_lstm, reuse=reuse)
                    initial_state = tf.contrib.rnn.LSTMStateTuple(states[start], states[start + 1])

                    def loop_fn(time, cell_output, cell_state, loop_state):
                        emit_output = cell_output

                        elements_finished = time >= n_steps
                        finished = tf.reduce_all(elements_finished)

                        # TODO: use masks
                        mask = tf.cond(finished,
                                       lambda: tf.zeros([self.n_env, 1], dtype=tf.float32),
                                       lambda: masks[:, time, :])
                        next_cell_state = cell_state or initial_state
                        next_cell_state = tf.contrib.rnn.LSTMStateTuple(next_cell_state.c * (1 - mask),
                                                                        next_cell_state.h * (1 - mask))

                        next_input = tf.cond(
                            finished,
                            lambda: tf.zeros([self.n_env, ff_out.shape[-1]],
                                             dtype=tf.float32),
                            lambda: inputs_ta.read(time))
                        next_loop_state = None
                        return (elements_finished, next_input, next_cell_state,
                                emit_output, next_loop_state)

                    outputs_ta, final_state, _ = tf.nn.raw_rnn(cell, loop_fn,
                                                               parallel_iterations=1,
                                                               scope=f'lstm{suffix}')
                    last_out = outputs_ta.stack()
                    last_out = seq_to_batch(last_out)
                    self.state_out.append(final_state)

                    return last_out, ff_list

                value_out, value_ff_acts = lstm(0, 'v')
                self._value_fn = tf.contrib.layers.fully_connected(value_out, 1, activation_fn=None)
                if self.normalized and self.normalized != 'ob':
                    self._value_fn = self.value_fn * self.ret_rms.std + self.ret_rms.mean  # raw = not standardized

                mean, policy_ff_acts = lstm(2, 'p')
                mean = tf.contrib.layers.fully_connected(mean, ac_space.shape[0],
                                                         activation_fn=None)
                logstd = tf.get_variable(name="logstd", shape=[1, ac_space.shape[0]],
                                         initializer=tf.zeros_initializer())
                self.ff_out = {'value': value_ff_acts, 'policy': policy_ff_acts}
                self._policy = tf.reshape(mean, [n_batch] + list(ac_space.shape))
                self.logstd = tf.reshape(logstd, ac_space.shape)

                zero_state = np.zeros((4, num_lstm), dtype=np.float32)
                self._initial_state = np.tile(zero_state, (self.n_env, 1, 1))

                for p in self.get_trainable_variables():
                    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.reduce_sum(tf.square(p)))

                self._setup_init()

    def _make_feed_dict(self, obs, state, mask):
        return {
            self.obs_ph: obs,
            self.states_ph: state,
            self.dones_ph: mask,
        }

    def step(self, obs, state=None, mask=None, deterministic=False, extra_op=None):
        action = self.deterministic_action if deterministic else self.action
        feed_dict = self._make_feed_dict(obs, state, mask)
        outputs = [action, self.value_flat, self.state_out, self.neglogp]
        if extra_op is not None:
            outputs.append(extra_op)
            a, v, s, neglogp, ex = self.sess.run(outputs, feed_dict)
        else:
            a, v, s, neglogp = self.sess.run(outputs, feed_dict)

        state = []
        for x in s:
            state.append(x.c)
            state.append(x.h)
        state = np.array(state)
        state = np.transpose(state, (1, 0, 2))

        if extra_op is not None:
            return a, v, state, neglogp, ex
        else:
            return a, v, state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, self._make_feed_dict(obs, state, mask))

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, self._make_feed_dict(obs, state, mask))


register_policy('BansalMlpPolicy', MlpPolicyValue)
register_policy('BansalLstmPolicy', LSTMPolicy)
