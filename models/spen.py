import tensorflow as tf

from base.base_model import BaseModel
from models.energy_net import EnergyNet
from models.inference_net import InferenceNet
from models.feature_net import FeatureNet
from utils.logger import get_logger

logger = get_logger(__name__)

BASE_FEAT_SCOPE = "feature_net"
BASE_ENERGY_NET_SCOPE = "energy_net"
BASE_INFNET_SCOPE = "inference_net"
BASE_COPY_INFNET_SCOPE = "copy_inference_net"

FEAT_SCOPE = "model/%s" % BASE_FEAT_SCOPE
ENERGY_NET_SCOPE = "model/%s" % BASE_ENERGY_NET_SCOPE
INFNET_SCOPE = "model/%s" % BASE_INFNET_SCOPE
COPY_INFNET_SCOPE = "model/%s" % BASE_COPY_INFNET_SCOPE

EPSILON = 1e-7


def random_uniform(limit):
    return tf.random_uniform_initializer(-limit, limit)


class SPEN(BaseModel):
    def __init__(self, config):
        super(SPEN, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        config = self.config
        batch_size = config.train.batch_size

        if config.data.embeddings is True:
            self.input_x = tf.placeholder(
                tf.int64, shape=[batch_size], name='input_x'
            )
            self.build_embeddings_graph()
        else:
            self.input_x = tf.placeholder(
                tf.float32, shape=[batch_size, config.data.finput_dim], name='input_x'
            )
            self.input_x_vector = tf.identity(self.input_x)

        self.labels_y = tf.placeholder(
            tf.float32, shape=[batch_size, config.type_vocab_size], name='labels_y'
        )

        self.build_feature_net()
        # Inference Network and Energy Network
        self.build_subnets()
        self.regularize()
        self.pretrain_feats()
        self.copy_infnet()
        if self.config.ssvm.enable is True:
            self.ssvm_losses()
        else:
            self.adversarial_losses()
        # Inference Network energy minimization & Inference Network pre-training
        self.infnet_losses()
        self.evaluate()

    def build_embeddings_graph(self):
        config = self.config
        vocab_size = config.entities_vocab_size
        e_size = config.feature_size
        # Logic for embeddings
        with tf.name_scope('embeddings'):
            self.embeddings_placeholder = tf.placeholder(
                tf.float32, [vocab_size, e_size], name="embeddings_placeholder"
            )
            self.embeddings = tf.get_variable(
                "embedding", [vocab_size, e_size],
                initializer=random_uniform(0.25),
                trainable=config.embeddings_tune
            )
            self.input_x_vector = tf.nn.embedding_lookup(self.embeddings, self.input_x)

            # Used in the static / non-static configurations
            self.load_embeddings = self.embeddings.assign(self.embeddings_placeholder)

    def build_feature_net(self):
        regularizer = tf.contrib.layers.l2_regularizer(1.0)

        with tf.variable_scope(BASE_FEAT_SCOPE, regularizer=regularizer):
            self.feature_network = FeatureNet(self.config, self.input_x_vector)
            self.feature_input = self.feature_network.layer2_out

    def build_subnets(self):
        config = self.config
        regularizer = tf.contrib.layers.l2_regularizer(1.0)

        # This is the cost-augmented inference network
        # This is used for GAN-style training
        # After training the energy network, we train the psi parameters
        with tf.variable_scope(BASE_INFNET_SCOPE, regularizer=regularizer):
            self.inference_net = InferenceNet(
                config, self.input_x_vector
            )
        # Energy network definitions
        with tf.variable_scope(BASE_ENERGY_NET_SCOPE, regularizer=regularizer):
            self.energy_net1 = EnergyNet(
                config, self.feature_input, self.labels_y
            )
            # DEBUG variable
            self.red_energy_gt_out = tf.reduce_mean(self.energy_net1.energy_out)
        with tf.variable_scope(BASE_ENERGY_NET_SCOPE, reuse=True, regularizer=regularizer):
            self.energy_net2 = EnergyNet(
                config, self.feature_input, self.inference_net.probs
            )
            # DEBUG variable
            self.red_energy_inf_out = tf.reduce_mean(self.energy_net2.energy_out)
        # Nodes for the W-GAN penalty
        eps = tf.random_uniform([config.train.batch_size, 1])
        self.modified_y = eps * self.labels_y + (1 - eps) * self.inference_net.probs
        with tf.variable_scope(BASE_ENERGY_NET_SCOPE, reuse=True, regularizer=regularizer):
            self.energy_net4 = EnergyNet(
                config, self.feature_input, self.modified_y
            )
        grads = tf.gradients(self.energy_net4.energy_out, self.modified_y)
        self.grad_penalty = tf.reduce_mean(tf.norm(grads, axis=1))

    def regularize(self):
        with tf.name_scope('regularize'):
            self.reg_losses_feats = tf.add_n(
                tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=FEAT_SCOPE)
            )
            self.reg_losses_phi = tf.add_n(
                tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=INFNET_SCOPE)
            )
            self.reg_losses_theta = tf.add_n(
                tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=ENERGY_NET_SCOPE)
            )
            # Entropy Regularization, Section 5 of Tu & Gimpel 2018
            self.reg_losses_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=self.inference_net.probs,
                logits=self.inference_net.layer3_out
            ))

    def pretrain_feats(self):
        with tf.name_scope('pretrain_feats'):
            # We take a negative sign here since we want "probabilities", not "energy"
            logits = -1 * self.energy_net1.negative_logits
            # Pre-training objective
            self.feats_ce_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=self.labels_y,
                logits=logits
            ))
            self.feats_opt = self.feats_ce_loss
            feat_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=FEAT_SCOPE)
            # The energy network local function is the final layer of the feed-forward network
            var_list = [self.energy_net1.linear_wt] + feat_vars
            self.feat_ce_opt = tf.train.AdamOptimizer(self.config.train.lr_feat).minimize(
                self.feats_opt, var_list=var_list
            )

    def copy_infnet(self):
        config = self.config
        regularizer = tf.contrib.layers.l2_regularizer(1.0)
        # These variables will store values of a preprocessed inference net
        # This allows us to keep a copy of the pre-trained network
        # Which can be used as a regularization term Section 5, Tu & Gimpel 2018
        with tf.variable_scope(BASE_COPY_INFNET_SCOPE, regularizer=regularizer):
            self.copy_inference_net = InferenceNet(
                config, self.input_x_vector
            )
        copy_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=COPY_INFNET_SCOPE)

        infnet_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=INFNET_SCOPE)

        pretrain_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=FEAT_SCOPE)
        # We take a negative sign here since we want "probabilities", not "energy"
        pretrain_vars.append(-1 * self.energy_net1.linear_wt)
        # Create assignment operators to initialize pretrained_inference_net
        with tf.name_scope('copy_infnet'):
            # This op will both copy pretrained network into infnet
            # as well as keep a copy against the pre-trained network
            self.copy_infnet_ops = [
                copy_var.assign(pretrain_var)
                for copy_var, pretrain_var in zip(copy_vars, pretrain_vars)
            ]
            # copying into main infnet
            self.copy_infnet_ops.extend([
                infnet_var.assign(pretrain_var)
                for infnet_var, pretrain_var in zip(infnet_vars, pretrain_vars)
            ])
            # The last loss term in Section 5, Tu & Gimpel 2018
            self.var_diff = [
                tf.reduce_sum(tf.square(copy_var - infnet_var))
                for copy_var, infnet_var in zip(copy_vars, infnet_vars)
            ]
            self.pretrain_bias = tf.add_n(self.var_diff)

    def ssvm_losses(self):
        config = self.config
        regularizer = tf.contrib.layers.l2_regularizer(1.0)
        batch_size = config.train.batch_size
        lamb_reg_theta = config.train.lamb_reg_theta

        # We are going to optimize over this variable
        with tf.variable_scope('ssvm_y'):
            self.ssvm_y_pred = tf.get_variable(
                'ssvm_y_pred', [batch_size, config.type_vocab_size],
                initializer=tf.random_uniform_initializer(0, 1)
            )
            # This is to enforce projected GD
            self.ssvm_y_pred_clip = tf.clip_by_value(
                self.ssvm_y_pred, EPSILON, 1 - EPSILON
            )
        # Reusing the Energy function module for this y_pred
        with tf.variable_scope(BASE_ENERGY_NET_SCOPE, reuse=True, regularizer=regularizer):
            self.energy_net3 = EnergyNet(
                config, self.feature_input, self.ssvm_y_pred_clip
            )
            self.red_energy_y_pred = tf.reduce_mean(self.energy_net3.energy_out)

        # Choosing the appropriate difference function
        if config.train.diff_type == 'sq_diff':
            self.ssvm_difference = tf.reduce_sum(
                tf.square(self.labels_y - self.ssvm_y_pred_clip), axis=1
            )
        elif config.train.diff_type == 'abs_diff':
            self.ssvm_difference = tf.reduce_sum(
                tf.abs(self.labels_y - self.ssvm_y_pred_clip), axis=1
            )
        elif config.train.diff_type == 'perceptron':
            self.ssvm_difference = tf.constant([0] * batch_size, dtype=tf.float32)
        elif config.train.diff_type == 'slack':
            self.ssvm_difference = tf.constant([1] * batch_size, dtype=tf.float32)

        # Defining the SSVM loss criterion
        ssvm_max_difference = tf.maximum(
            self.ssvm_difference - self.energy_net3.energy_out + self.energy_net1.energy_out,
            tf.constant([0] * batch_size, dtype=tf.float32)
        )
        self.ssvm_base_objective = tf.reduce_mean(ssvm_max_difference)
        # Defining the loss augmented inference loss
        self.cost_augmented_inference = tf.reduce_mean(
            -1 * self.ssvm_difference + self.energy_net3.energy_out
        )
        # This loss term is just used for inference by directly minimizing energy
        self.cost_inference = tf.reduce_mean(
            self.energy_net3.energy_out
        )
        # Defining the loss function for theta
        self.ssvm_cost_theta = \
            self.ssvm_base_objective + \
            lamb_reg_theta * self.reg_losses_theta
        # Defining all the optimizers
        # Defining the optimizer for y, inner loop of optimization
        self.ssvm_y_opt = tf.train.GradientDescentOptimizer(config.ssvm.lr_inference).minimize(
            self.cost_augmented_inference, var_list=[self.ssvm_y_pred]
        )
        # This optimizer just used during inference
        self.ssvm_infer_y_opt = tf.train.GradientDescentOptimizer(config.ssvm.lr_inference).minimize(
            self.cost_inference, var_list=[self.ssvm_y_pred]
        )
        # Defining the optimizer for theta variables during training
        theta_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=ENERGY_NET_SCOPE)
        self.ssvm_theta_opt = tf.train.AdamOptimizer(config.train.lr_theta).minimize(
            self.ssvm_cost_theta, global_step=self.global_step_tensor, var_list=theta_vars
        )

    def adversarial_losses(self):
        config = self.config
        batch_size = config.train.batch_size

        lamb_reg_phi = config.train.lamb_reg_phi
        lamb_reg_theta = config.train.lamb_reg_theta
        lamb_reg_entropy = config.train.lamb_reg_entropy
        lamb_pretrain_bias = config.train.lamb_pretrain_bias
        lamb_wgan_penalty = config.train.lamb_wgan_penalty

        with tf.name_scope('base_objective'):
            if config.train.diff_type == 'sq_diff':
                self.difference = tf.reduce_sum(
                    tf.square(self.labels_y - self.inference_net.probs), axis=1
                )
            elif config.train.diff_type == 'abs_diff':
                self.difference = tf.reduce_sum(
                    tf.abs(self.labels_y - self.inference_net.probs), axis=1
                )
            elif config.train.diff_type == 'perceptron':
                self.difference = tf.constant([0] * batch_size, dtype=tf.float32)
            elif config.train.diff_type == 'slack':
                self.difference = tf.constant([1] * batch_size, dtype=tf.float32)

            # Applying the hinge to the loss function
            max_difference = tf.maximum(
                self.difference - self.energy_net2.energy_out + self.energy_net1.energy_out,
                tf.constant([0] * batch_size, dtype=tf.float32)
            )
            self.base_objective_real = tf.reduce_mean(
                self.difference - self.energy_net2.energy_out + self.energy_net1.energy_out
            )
            self.base_objective = tf.reduce_mean(max_difference)

        with tf.name_scope('theta_cost'):
            self.cost_theta = \
                self.base_objective + \
                lamb_reg_theta * self.reg_losses_theta
            if config.train.wgan_mode is True:
                self.cost_theta += lamb_wgan_penalty * tf.square(self.grad_penalty - 1)

        with tf.name_scope('gain_phi'):
            # Negative sign for reg_losses_phi since we want to maximize phi objective
            self.gain_phi = \
                self.base_objective - \
                lamb_reg_phi * self.reg_losses_phi - \
                lamb_reg_entropy * self.reg_losses_entropy - \
                lamb_pretrain_bias * self.pretrain_bias
            self.cost_phi = -1 * self.gain_phi

        with tf.name_scope('phi_opt'):
            phi_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=INFNET_SCOPE)
            self.phi_opt = tf.train.AdamOptimizer(config.train.lr_phi).minimize(
                self.cost_phi, global_step=self.global_step_tensor, var_list=phi_vars
            )
        with tf.name_scope('theta_opt'):
            theta_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=ENERGY_NET_SCOPE)[1:]
            self.theta_opt = tf.train.AdamOptimizer(config.train.lr_theta).minimize(
                self.cost_theta, var_list=theta_vars
            )

    def infnet_losses(self):
        config = self.config

        infnet_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=INFNET_SCOPE)
        lamb_reg_entropy = config.train.lamb_reg_entropy
        lamb_reg_phi = config.train.lamb_reg_phi

        with tf.name_scope('psi_cost'):
            # Psi parameter optimization, Equation (5) in Tu & Gimpel, 2018
            self.energy_psi = \
                tf.reduce_sum(self.energy_net2.energy_out) + \
                lamb_reg_entropy * self.reg_losses_entropy + \
                lamb_reg_phi * self.reg_losses_phi

        with tf.name_scope('psi_opt'):
            self.psi_opt = tf.train.AdamOptimizer(self.config.train.lr_psi).minimize(
                self.energy_psi, global_step=self.global_step_tensor_inf, var_list=infnet_vars
            )

    def evaluate(self):
        with tf.name_scope('evaluate'):
            self.pretrain_probs = self.energy_net1.pretrain_probs
            self.infnet_probs = self.inference_net.probs

            if self.config.ssvm.enable is True:
                # SSVM based inference
                # Please optimize over ssvm_y_pred before using these nodes
                self.ssvm_probs = self.ssvm_y_pred_clip

    def init_saver(self):
        with tf.name_scope('saver'):
            # here you initalize the tensorflow saver that will be used in saving the checkpoints.
            self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
