import numpy as np
import time
import os
import sys
import tensorflow as tf

from .bucket_utils import *

np.random.seed(1337) # for reproducibility
tf.set_random_seed(1337)

pdeep_lstm_cell = tf.contrib.rnn.LSTMCell
# pdeep_lstm_cell = tf.contrib.rnn.LayerNormBasicLSTMCell

use_tensorboard = True

def variable_summaries(var, scope):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope(scope+"/summaries"):
      mean = tf.reduce_mean(var)
      tf.summary.scalar("mean", mean)
      with tf.name_scope("stddev"):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.summary.scalar("stddev", stddev)
      tf.summary.scalar("max", tf.reduce_max(var))
      tf.summary.scalar("min", tf.reduce_min(var))
      tf.summary.histogram("histogram", var)

class IonLSTM:
    # conf is ETD_config or HCD_Config in fragmentation_config.py
    def __init__(self, conf):
        self.batch_size = 1024
        self.layer_size = 256
        self.epochs = 100
        self.config = conf
        self.sess = None
        self.learning_rate = 0.001
        self.num_threads = 2
        
        self.save_param = {}
        
        self.instrument_ce_scope = "instrument_nce" # this scope may have to be tuned in transfer learning
        
        self.mod_scope = "mod_scope"
    
    # model
    def BuildModel(self, input_size, output_size, nlayers = 2):
        print("BuildModel ... ")
        tf.reset_default_graph()
        self._x = tf.placeholder("float", [None, None, input_size], name = "input_x")
        self._y = tf.placeholder("float",[None, None, output_size], name = "input_y")
        self._time_step = tf.placeholder(tf.int32, [None,], name = "input_time_step")
        self._charge = tf.placeholder("float", [None, None, 1], name = "input_charge")
        self._nce = tf.placeholder("float", [None, None, 1], name = "input_nce")
        self._instrument = tf.placeholder("float", [None, None, self.config.max_instrument_num], name = "input_instrument")
        self.rnn_kp = tf.placeholder_with_default(1.0, shape=(), name = "rnn_keep_prob")
        self.output_kp = tf.placeholder_with_default(1.0, shape=(), name = "output_keep_prob")
        self._loss = 0
        regularization = 0.01
        
        instrument_ce_scope = self.instrument_ce_scope
        
        def QuadraticFunc(val, scope):
            with tf.variable_scope(scope):
                a = tf.Variable(0.1, dtype = tf.float32, name = "square_coef", trainable = True)
                b = tf.Variable(0.01, dtype = tf.float32, name = "linear_coef", trainable = True)
                c = tf.Variable(0.001, dtype = tf.float32, name = "bias", trainable = True)
            
            tf.summary.scalar(scope+"/a", a)
            tf.summary.scalar(scope+"/b", b)
            tf.summary.scalar(scope+"/c", c)
            
            return a * val*val + b * val + c
        
        def AddToInput(x, ch):
            a = tf.Variable(1, name = "charge_weight", trainable = True, dtype = tf.float32)
            b = tf.Variable(1, name = "rnn_weight", trainable = True, dtype = tf.float32)
            ch = tf.reshape(ch, (-1, 1, 1))
            ch = tf.tile(ch, [1, 1, x.shape[2]])
            return a*x + b*tf.cast(ch, tf.float32)
            
        def LinearTrans(x, to_size, scope):
            with tf.variable_scope(scope):
                x = tf.transpose(x, perm = (1,0,2))
                scan_w = tf.Variable(tf.random_normal([self.config.max_instrument_num+1, to_size]), name = "scan_w")
                init = tf.matmul(x[0,:,:], tf.zeros([self.config.max_instrument_num+1, to_size]))
                def step(pre_stat, in_x):
                    return tf.matmul(in_x, scan_w)
                outputs = tf.scan(step, x, init)
                outputs = tf.transpose(outputs, perm = (1,0,2))
            variable_summaries(scan_w, scope = scope+"/scan_w")
            return outputs
        
        def ConcatToRNN(x, ch):
            return tf.concat((x, ch), axis = 2)
        
        ch = self._charge
        # ch = QuadraticFunc(ch, scope = "charge")
        ch = tf.identity(ch, name = "charge_value")
        
        x = self._x
        x = ConcatToRNN(x, ch)
        
        if self.config.enable_instrument_and_nce:
            ins = ConcatToRNN(self._instrument, self._nce)
            ins = LinearTrans(ins, 3, scope = instrument_ce_scope)
            ins = tf.identity(ins, name = "ins_nce_value")
            # x = ConcatToRNN(x, ins)
            
            variable_summaries(ins, scope = instrument_ce_scope)
            
        def ConcatFeatures(x):
            x = ConcatToRNN(x, ch)
            if self.config.enable_instrument_and_nce: x = ConcatToRNN(x, ins)
            return x
        
        def StackBiLSTM(x, nlayers):
            #MultiCell
            def MultiCell(x):
                fw_cells = []
                bw_cells = []
                for id in range(nlayers):
                    lstm_fw_cell = pdeep_lstm_cell(self.layer_size, activation = tf.nn.tanh)
                    lstm_bw_cell = pdeep_lstm_cell(self.layer_size, activation = tf.nn.tanh)
                    lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(lstm_fw_cell, input_keep_prob = 1, output_keep_prob = self.rnn_kp, state_keep_prob = self.rnn_kp, variational_recurrent = False, dtype = tf.float32)
                    lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(lstm_bw_cell, input_keep_prob = 1, output_keep_prob = self.rnn_kp, state_keep_prob = self.rnn_kp, variational_recurrent = False, dtype = tf.float32)
                    fw_cells.append(lstm_fw_cell)
                    bw_cells.append(lstm_bw_cell)
                lstm_fw_cell = tf.contrib.rnn.MultiRNNCell(fw_cells)
                lstm_bw_cell = tf.contrib.rnn.MultiRNNCell(bw_cells)
                x, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, x, sequence_length = self._time_step, time_major = False, dtype=tf.float32, scope = "BiLSTM")
                x = tf.concat(x, axis=2)
                #x = tf.nn.relu(x)
                x = tf.nn.dropout(x, keep_prob = self.output_kp)
                x = tf.identity(x, name = "BiLSTM_output")
                x = ConcatFeatures(x) # concatenate to the last layer for fine-tining
                return x
            
            #MultiLayer
            def MultiLayer(x):
                
                def BiLSTM(x, id):
                    lstm_fw_cell = pdeep_lstm_cell(self.layer_size, activation = tf.nn.tanh)
                    lstm_bw_cell = pdeep_lstm_cell(self.layer_size, activation = tf.nn.tanh)
                    lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(lstm_fw_cell, input_keep_prob = 1, output_keep_prob = self.rnn_kp, state_keep_prob = self.rnn_kp, variational_recurrent = False, dtype = tf.float32)
                    lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(lstm_bw_cell, input_keep_prob = 1, output_keep_prob = self.rnn_kp, state_keep_prob = self.rnn_kp, variational_recurrent = False, dtype = tf.float32)
                    x, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, x, sequence_length = self._time_step, time_major = False, dtype=tf.float32, scope = "BiLSTM_%d"%id)
                    x = tf.concat(x, axis=2)
                    #x = tf.nn.relu(x)
                    x = tf.nn.dropout(x, keep_prob = self.output_kp)
                    return x
                
                for id in range(nlayers):
                    x = BiLSTM(x, id)
                    x = tf.identity(x, name = "BiLSTM_output_%d"%id)
                
                    x = ConcatFeatures(x) # concatenate to the last layer for fine-tining
                return x

            def MultiLayer_with_Attention(x):
                attn_conv_win = 10
                def BiLSTM_with_Attention(x, id):
                    lstm_fw_cell = pdeep_lstm_cell(self.layer_size, activation = tf.nn.tanh)
                    lstm_bw_cell = pdeep_lstm_cell(self.layer_size, activation = tf.nn.tanh)
                    lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(lstm_fw_cell, input_keep_prob = 1, output_keep_prob = self.rnn_kp, state_keep_prob = self.rnn_kp, variational_recurrent = False, dtype = tf.float32)
                    lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(lstm_bw_cell, input_keep_prob = 1, output_keep_prob = self.rnn_kp, state_keep_prob = self.rnn_kp, variational_recurrent = False, dtype = tf.float32)
                    lstm_fw_cell = tf.contrib.rnn.AttentionCellWrapper(lstm_fw_cell, attn_length=attn_conv_win)
                    lstm_bw_cell = tf.contrib.rnn.AttentionCellWrapper(lstm_bw_cell, attn_length=attn_conv_win)
                    x, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, x, sequence_length = self._time_step, time_major = False, dtype=tf.float32, scope = "BiLSTM_%d"%id)
                    x = tf.concat(x, axis=2)
                    #x = tf.nn.relu(x)
                    x = tf.nn.dropout(x, keep_prob = self.output_kp)
                    return x
                
                for id in range(nlayers):
                    x = BiLSTM_with_Attention(x, id)
                    x = tf.identity(x, name = "BiLSTM_output_%d"%id)
                
                    x = ConcatFeatures(x) # concatenate to the last layer for fine-tining
                return x
            
            return MultiLayer(x)
        
        x = StackBiLSTM(x, nlayers)
        
        def TransferLayer(x):
            cell_fw = pdeep_lstm_cell(output_size)
            cell_bw = pdeep_lstm_cell(output_size)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, x, sequence_length = self._time_step, time_major = False, dtype=tf.float32, scope = "output_scope/transfer_rnn")
            outputs = tf.add(outputs[0], outputs[1])
            return outputs
        
        def OutputLayer(x):
            def OutputRNN(x):
                cell_fw = pdeep_lstm_cell(output_size)
                cell_bw = pdeep_lstm_cell(output_size)
                outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, x, sequence_length = self._time_step, time_major = False, dtype=tf.float32, scope = "output_scope/backbone_rnn") # this scope may have to be tuned in transfer learning
                outputs = tf.add(outputs[0], outputs[1])
                return outputs
            
            # real time distributed layer
            def OutputScan(x):
                x = tf.transpose(x, perm = (1,0,2))
                scan_w = tf.Variable(tf.random_normal([x.shape[-1].value, output_size]), name = "scan_w")
                scan_b = tf.Variable(tf.random_normal([output_size]), name = "scan_b")
                init = tf.matmul(x[0,:,:], tf.zeros([x.shape[-1].value, output_size]))
                def step(pre_stat, in_x):
                    return tf.nn.bias_add(tf.matmul(in_x, scan_w), scan_b)
                outputs = tf.scan(step, x, init)
                outputs = tf.transpose(outputs, perm = (1,0,2))
                return outputs
            
            return OutputRNN(x)
            
        #outputs = 0.5 * OutputLayer(x) + 0.5 * TransferLayer(x)
        outputs = OutputLayer(x)
        #outputs = tf.nn.relu(outputs)
        #outputs = tf.minimum(outputs, 1)
        with tf.name_scope("output_scope"):
            self._prediction = tf.identity(outputs, name = "output")
            
        self._loss += tf.reduce_mean(tf.abs(self._prediction - self._y))
        
        variable_summaries(self._prediction - self._y, scope = "pred-y")
        
        # output sparsity
        #self._loss += 0.01*tf.reduce_mean(tf.abs(self._prediction))
        #self._optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
        self._optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        #self._optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum = 0.9)
        self._mininize = self._optimizer.minimize(self._loss)
        
        self.merged_summary_op = tf.summary.merge_all()
        
        self.restart_session()
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.summary_writer = tf.summary.FileWriter("../tensorboard/train", self.sess.graph)
    
    def BuildTransferModel(self, model_file):
        print("Fine-tuning pDeep model ...")
        self.LoadModel(model_file)
        
        transfer_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "output_scope/backbone_rnn")
        transfer_vars += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "BiLSTM_0")
        # transfer_vars += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.instrument_ce_scope)
        
        self._loss = tf.reduce_mean(tf.abs(self._prediction - self._y))
        
        self._optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, name = "transfer_Adam")
        
        self._mininize = self._optimizer.minimize(self._loss, var_list = transfer_vars)
        
        self.merged_summary_op = tf.summary.merge_all()
        
        adam_vars = [self.GetVariableByName("beta1_power_1:0"), self.GetVariableByName("beta2_power_1:0")] + self.GetVariablesBySubstr("transfer_Adam")
        init_op = tf.variables_initializer(var_list = adam_vars, name = "transfer_init")
        self.sess.run(init_op)
        self.summary_writer = tf.summary.FileWriter("../tensorboard/transfer_train", self.sess.graph)
        
    # def BuildTransferModel_offline(self, model_file):
        # print("pDeep knowledge transfer model ...")
        # self.LoadModel(model_file)
        
        # with tf.variable_scope(self.mod_scope):
        
            # rnn_kp = tf.placeholder_with_default(1.0, shape=(), name = "rnn_keep_prob")
            # output_kp = tf.placeholder_with_default(1.0, shape=(), name = "output_keep_prob")
            # def StackBiLSTM(x, nlayers):
                # #MultiCell
                # def MultiCell(x):
                    # fw_cells = []
                    # bw_cells = []
                    # for id in range(nlayers):
                        # lstm_fw_cell = tf.contrib.rnn.LSTMCell(self.layer_size, activation = tf.nn.tanh)
                        # lstm_bw_cell = tf.contrib.rnn.LSTMCell(self.layer_size, activation = tf.nn.tanh)
                        # if rnn_kp < 1:
                            # lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(lstm_fw_cell, input_keep_prob = 1, output_keep_prob = rnn_kp, state_keep_prob = rnn_kp, variational_recurrent = False, dtype = tf.float32)
                            # lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(lstm_bw_cell, input_keep_prob = 1, output_keep_prob = rnn_kp, state_keep_prob = rnn_kp, variational_recurrent = False, dtype = tf.float32)
                        # fw_cells.append(lstm_fw_cell)
                        # bw_cells.append(lstm_bw_cell)
                    # lstm_fw_cell = tf.contrib.rnn.MultiRNNCell(fw_cells)
                    # lstm_bw_cell = tf.contrib.rnn.MultiRNNCell(bw_cells)
                    # x, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, x, sequence_length = self._time_step, time_major = False, dtype=tf.float32, scope = 'backbone/BiLSTM')
                    # x = tf.concat(x, axis=2)
                    # #x = tf.nn.relu(x)
                    # x = tf.nn.dropout(x, keep_prob = output_kp)
                    # x = x + self.GetTensorByName('backbone/BiLSTM_output:0')
                    # return x
                
                # #MultiLayer
                # def MultiLayer(x):
                    
                    # def BiLSTM(x, id):
                        # lstm_fw_cell = tf.contrib.rnn.LSTMCell(self.layer_size, activation = tf.nn.tanh)
                        # lstm_bw_cell = tf.contrib.rnn.LSTMCell(self.layer_size, activation = tf.nn.tanh)
                        # if rnn_kp < 1:
                            # lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(lstm_fw_cell, input_keep_prob = 1, output_keep_prob = rnn_kp, state_keep_prob = rnn_kp, variational_recurrent = False, dtype = tf.float32)
                            # lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(lstm_bw_cell, input_keep_prob = 1, output_keep_prob = rnn_kp, state_keep_prob = rnn_kp, variational_recurrent = False, dtype = tf.float32)
                        # x, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, x, sequence_length = self._time_step, time_major = False, dtype=tf.float32, scope = 'backbone/BiLSTM_%d'%id)
                        # x = tf.concat(x, axis=2)
                        # #x = tf.nn.relu(x)
                        # x = tf.nn.dropout(x, keep_prob = output_kp)
                        # return x
                    
                    # for id in range(nlayers):
                        # x = BiLSTM(x, id)
                        # x = x + self.GetTensorByName('backbone/BiLSTM_output_%d:0'%id)
                    # return x
                
                # return MultiLayer(x)
            
            # x = StackBiLSTM(x, nlayers)
            
            # def OutputLayer(x, title = 'modion'):
            
                # def OutputRNN(x):
                    # cell_fw = tf.contrib.rnn.LSTMCell(output_size)
                    # cell_bw = tf.contrib.rnn.LSTMCell(output_size)
                    # outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, x, sequence_length = self._time_step, time_major = False, dtype=tf.float32, scope = title + '/output_rnn')
                    # outputs = tf.add(outputs[0], outputs[1])
                    # return outputs
                
                # # real time distributed layer
                # def OutputScan(x):
                    # x = tf.transpose(x, perm = (1,0,2))
                    # scan_w = tf.Variable(tf.random_normal([x.shape[-1].value, output_size]), name = 'scan_w')
                    # scan_b = tf.Variable(tf.random_normal([output_size]), name = 'scan_b')
                    # init = tf.matmul(x[0,:,:], tf.zeros([x.shape[-1].value, output_size]))
                    # def step(pre_stat, in_x):
                        # return tf.nn.bias_add(tf.matmul(in_x, scan_w), scan_b)
                    # outputs = tf.scan(step, x, init)
                    # outputs = tf.transpose(outputs, perm = (1,0,2))
                    # return outputs
                
                # return OutputRNN(x)
            
            
            
            # outputs = OutputLayer(x)
    
    def restart_session(self):
        self.close_session()
        self.init_session()
        
    def init_session(self):
        if self.sess is None:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth=True
            config.intra_op_parallelism_threads = self.num_threads
            self.sess = tf.Session(config=config)
            
    def close_session(self):
        if self.sess is not None:
            self.sess.close()
            self.sess = None
    
    def SaveVariableName(self, save_as):
        variables_names = [v.name for v in tf.global_variables()]
        with open(save_as+".varname","w") as f:
            for var in variables_names: f.write(var + "\n")
            
    def SaveTensorName(self, save_as):
        tensor_names = [n.name for n in tf.get_default_graph().as_graph_def().node]
        with open(save_as+".tenname","w") as f:
            for name in tensor_names: f.write(name + "\n")
    
    def GetTensorByName(self, name):
        return tf.get_default_graph().get_tensor_by_name(name)
    
    def GetVariableByName(self, name):
        l = [v for v in tf.global_variables() if v.name == name]
        if len(l) == 0: return None
        else: return l[0]
        
    def GetVariablesBySubstr(self, substr):
        return [v for v in tf.global_variables() if substr in v.name]
        
    def GetVarialbeListByScope(self, scope):
        return get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = scope)
        
    def convert_value_for_tf(self, ch, time_step, outsize = 1):
        ch = np.reshape(ch, (-1, 1, outsize))
        ch = np.repeat(ch, time_step, axis=1)
        ch.astype(np.float32, copy = False)
        return ch
    
    def mod_feature(self, x, mod_x):
        return np.concatenate((x, mod_x), axis = 2)
    
    def TrainModel(self, buckets, save_as = None):
        if self.sess is None: 
            print("[Error] no session for training tensorflow!")
            sys.exit(-1)
        
        bbatch = Bucket_Batch(buckets, batch_size = self.batch_size, shuffle = True)
        
        # print([v.name for v in tf.global_variables()])
        
        mean_costs = []
        
        var_list = []
        var_list.append(self.GetVariableByName("output_scope/backbone_rnn/fw/lstm_cell/kernel:0")) #.../gru_cell/gates(candidates)/kernel:0
        var_list.append(self.GetVariableByName("output_scope/backbone_rnn/bw/lstm_cell/kernel:0")) #.../gru_cell/gates(candidates)/kernel:0
      
        for epoch in range(self.epochs):
            batch_cost = []
            batch_time_cost = []
            ith_batch = 0
            bbatch.reset_batch()
            while True:
                start_time = time.perf_counter()
                batch = bbatch.get_next_batch()
                if batch is None: break
                ith_batch += 1
                peplen = bbatch.get_data_from_batch(batch, "peplen")
                ch = np.float32(bbatch.get_data_from_batch(batch, "charge"))
                x = np.float32(bbatch.get_data_from_batch(batch, "x"))
                mod_x = np.float32(bbatch.get_data_from_batch(batch, "mod_x"))
                instrument = np.float32(bbatch.get_data_from_batch(batch, "instrument"))
                nce = np.float32(bbatch.get_data_from_batch(batch, "nce"))
                y = bbatch.get_data_from_batch(batch, "y")
                
                x = self.mod_feature(x, mod_x)
                ch = self.convert_value_for_tf(ch, peplen[0] - 1)
                nce = self.convert_value_for_tf(nce, peplen[0] - 1)
                instrument = self.convert_value_for_tf(instrument, peplen[0] - 1, instrument.shape[-1])
                
                feed_dict = {
                    self._x: x, 
                    self._charge : ch, 
                    self._time_step : peplen - 1,
                    self._nce : nce,
                    self._instrument : instrument,
                    self._y : y,
                    self.rnn_kp: 0.8,
                    self.output_kp: 0.8
                }
                
                self.sess.run(self._mininize, feed_dict = feed_dict)
                end_time = time.perf_counter()
                batch_time_cost.append(end_time - start_time)
                
                summary_str, cost = self.sess.run([self.merged_summary_op, self._loss], feed_dict = feed_dict)
                self.summary_writer.add_summary(summary_str, epoch * 10000 + ith_batch)
                batch_cost.append(cost)
                
                #m = self._optimizer.get_slot(self.GetVariableByName("backbone_rnn/fw/lstm_cell/kernel:0"), "v")
                #v = self._optimizer.get_slot(self.GetVariableByName("backbone_rnn/fw/lstm_cell/kernel:0"), "m")
                #grads = self.sess.run([m,v], feed_dict = feed_dict)
                #print(grads)
                
                var_list = []
                if len(var_list) > 0:
                    grads = self._optimizer.compute_gradients(self._loss, var_list = var_list)
                    grads = self.sess.run(grads, feed_dict = feed_dict)
                    print("\n")
                    for i in range(len(grads)):
                        var = var_list[i]
                        print("mean update of {} = {}".format(var.name, np.mean(np.abs(grads[i][0]))))
                
                print("Epoch = {:3d}, peplen = {:3d}, Batch={:5d}, size = {:4d}, cost = {:.5f}, time = {:.2f}s\r".format(epoch + 1, peplen[0], ith_batch, len(x), cost, end_time - start_time), end = "")
            mean_costs.append("Epoch = {:3d}, mean_cost = {:.5f}, time = {:.2f}s".format(epoch+1, np.mean(batch_cost), np.sum(batch_time_cost)))
            print("\n" + mean_costs[-1])
        
        print("")
        for l in mean_costs:
            print(l)
        self.summary_writer.close()
            
        if save_as is not None: 
            dir = os.path.dirname(save_as)
            if not os.path.exists(dir): os.makedirs(dir)
            saver = tf.train.Saver()
            save_path = saver.save(self.sess, save_as)
            print("Model save as %s"%save_path)
            self.SaveVariableName(save_as)
            self.SaveTensorName(save_as)
    # model
    def Predict(self, buckets):
        bbatch = Bucket_Batch(buckets, batch_size = self.batch_size, shuffle = False)
        output_buckets = {}
        while True:
            batch = bbatch.get_next_batch()
            if batch is None: break
            peplen = bbatch.get_data_from_batch(batch, "peplen")
            ch = np.float32(bbatch.get_data_from_batch(batch, "charge"))
            x = np.float32(bbatch.get_data_from_batch(batch, "x"))
            mod_x = np.float32(bbatch.get_data_from_batch(batch, "mod_x"))
            instrument = np.float32(bbatch.get_data_from_batch(batch, "instrument"))
            nce = np.float32(bbatch.get_data_from_batch(batch, "nce"))
            # y = bbatch.get_data_from_batch(batch, "y")
            
            x = self.mod_feature(x, mod_x)
            ch = self.convert_value_for_tf(ch, peplen[0] - 1)
            nce = self.convert_value_for_tf(nce, peplen[0] - 1)
            instrument = self.convert_value_for_tf(instrument, peplen[0] - 1, instrument.shape[-1])
            
            feed_dict = {
                self._x: x, 
                self._charge : ch, 
                self._time_step : peplen - 1,
                self._nce : nce,
                self._instrument : instrument
            }
            predictions = self.sess.run(self._prediction, feed_dict = feed_dict)
            predictions[predictions > 1] = 1
            predictions[predictions < 0] = 0
            _buckets = {peplen[0] : (predictions,)}
            output_buckets = merge_buckets(output_buckets, _buckets)
        return output_buckets
    
    # model
    def LoadModel(self, model_file):
        self.restart_session()
        saver = tf.train.import_meta_graph(model_file+".meta")
        saver.restore(self.sess, model_file)
        graph = tf.get_default_graph()
        self._x = graph.get_tensor_by_name("input_x:0")
        self._y = graph.get_tensor_by_name("input_y:0")
        self._time_step = graph.get_tensor_by_name("input_time_step:0")
        self._charge = graph.get_tensor_by_name("input_charge:0")
        self._instrument = graph.get_tensor_by_name("input_instrument:0")
        self._nce = graph.get_tensor_by_name("input_nce:0")
        self._prediction = graph.get_tensor_by_name("output_scope/output:0")
        self.rnn_kp = graph.get_tensor_by_name("rnn_keep_prob:0")
        self.output_kp = graph.get_tensor_by_name("output_keep_prob:0")
        

