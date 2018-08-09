import numpy as np
import time
import os
import tensorflow as tf

from .bucket_utils import *

np.random.seed(1337) # for reproducibility
tf.set_random_seed(1337)

class IonLSTM:
    # conf is ETD_config or HCD_Config in fragmentation_config.py
    def __init__(self, conf):
        self.batch_size = 1024
        self.layer_size = 256
        self.epochs = 100
        self.config = conf
        self.sess = None
        self.learning_rate = 0.001
        self.mod_scope = "mod_scope"
        self.backbone_idxs = self.config.GetIntenIdx('backbone')
        self.modloss_idxs = self.config.GetIntenIdx('ModLoss')

    def get_ion_inten(self, y, ion_idxs):
        return y[:,:,ion_idxs]
    
    # model
    def BuildModel(self, input_size, output_size, unmod_model, nlayers = 1):
        print('BuildModel ... ')
        self.__load_unmod_model__(unmod_model)
        self._mod_x = tf.placeholder('float', [None, None, input_size], name = 'input_mod_x')
        self._y = tf.placeholder('float',[None, None, output_size])
        # if len(self.modloss_idxs) > 0: self._modloss_y = tf.placeholder('float',[None, None, output_size])
        self._loss = 0
        
        def ConcatChToRNN(x, ch):
            return tf.concat((x, ch), axis = 2)
        x = self._mod_x
        x = ConcatChToRNN(x, self._charge_value)
        
        with tf.variable_scope(self.mod_scope):
            def StackBiLSTM(x, nlayers):
                rnn_kp = 0.8
                output_kp = 0.8
                
                #MultiCell
                def MultiCell(x):
                    fw_cells = []
                    bw_cells = []
                    for id in range(nlayers):
                        lstm_fw_cell = tf.contrib.rnn.LSTMCell(self.layer_size, activation = tf.nn.tanh)
                        lstm_bw_cell = tf.contrib.rnn.LSTMCell(self.layer_size, activation = tf.nn.tanh)
                        if rnn_kp < 1:
                            lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(lstm_fw_cell, input_keep_prob = 1, output_keep_prob = rnn_kp, state_keep_prob = rnn_kp, variational_recurrent = False, dtype = tf.float32)
                            lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(lstm_bw_cell, input_keep_prob = 1, output_keep_prob = rnn_kp, state_keep_prob = rnn_kp, variational_recurrent = False, dtype = tf.float32)
                        fw_cells.append(lstm_fw_cell)
                        bw_cells.append(lstm_bw_cell)
                    lstm_fw_cell = tf.contrib.rnn.MultiRNNCell(fw_cells)
                    lstm_bw_cell = tf.contrib.rnn.MultiRNNCell(bw_cells)
                    x, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, x, sequence_length = self._time_step, time_major = False, dtype=tf.float32, scope = 'backbone/BiLSTM')
                    x = tf.concat(x, axis=2)
                    #x = tf.nn.relu(x)
                    x = tf.nn.dropout(x, keep_prob = output_kp)
                    x = x + self.GetTensorByName('backbone/BiLSTM_output:0')
                    return x
                
                #MultiLayer
                def MultiLayer(x):
                    
                    def BiLSTM(x, id):
                        lstm_fw_cell = tf.contrib.rnn.LSTMCell(self.layer_size, activation = tf.nn.tanh)
                        lstm_bw_cell = tf.contrib.rnn.LSTMCell(self.layer_size, activation = tf.nn.tanh)
                        if rnn_kp < 1:
                            lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(lstm_fw_cell, input_keep_prob = 1, output_keep_prob = rnn_kp, state_keep_prob = rnn_kp, variational_recurrent = False, dtype = tf.float32)
                            lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(lstm_bw_cell, input_keep_prob = 1, output_keep_prob = rnn_kp, state_keep_prob = rnn_kp, variational_recurrent = False, dtype = tf.float32)
                        x, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, x, sequence_length = self._time_step, time_major = False, dtype=tf.float32, scope = 'backbone/BiLSTM_%d'%id)
                        x = tf.concat(x, axis=2)
                        #x = tf.nn.relu(x)
                        x = tf.nn.dropout(x, keep_prob = output_kp)
                        return x
                    
                    for id in range(nlayers):
                        x = BiLSTM(x, id)
                        x = x + self.GetTensorByName('backbone/BiLSTM_output_%d:0'%id)
                    return x
                
                return MultiLayer(x)
            
            x = StackBiLSTM(x, nlayers)
            
            def OutputLayer(x, title = 'modion'):
            
                def OutputRNN(x):
                    cell_fw = tf.contrib.rnn.LSTMCell(output_size)
                    cell_bw = tf.contrib.rnn.LSTMCell(output_size)
                    outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, x, sequence_length = self._time_step, time_major = False, dtype=tf.float32, scope = title + '/output_rnn')
                    outputs = tf.add(outputs[0], outputs[1])
                    return outputs
                
                # real time distributed layer
                def OutputScan(x):
                    x = tf.transpose(x, perm = (1,0,2))
                    scan_w = tf.Variable(tf.random_normal([x.shape[-1].value, output_size]), name = 'scan_w')
                    scan_b = tf.Variable(tf.random_normal([output_size]), name = 'scan_b')
                    init = tf.matmul(x[0,:,:], tf.zeros([x.shape[-1].value, output_size]))
                    def step(pre_stat, in_x):
                        return tf.nn.bias_add(tf.matmul(in_x, scan_w), scan_b)
                    outputs = tf.scan(step, x, init)
                    outputs = tf.transpose(outputs, perm = (1,0,2))
                    return outputs
                
                return OutputRNN(x)
            
            
            
            outputs = OutputLayer(x)
            #outputs = tf.nn.relu(outputs)
            #outputs = tf.minimum(outputs, 1)
            #outputs = outputs + self._unmod_prediction
            self._prediction = tf.identity(outputs, name = 'modion_output')
    
            self._loss += tf.reduce_mean(tf.abs(self._prediction - self._y))
            
            # if len(self.modloss_idxs) > 0:
                # modloss_outputs = OutputLayer(x, title = 'modloss')
                # modloss_outputs = modloss_outputs + self._unmod_prediction - self._prediction
                # self._modloss_prediction = tf.identity(modloss_outputs, name = 'modloss_output')
                
                # self._loss += tf.reduce_mean(tf.abs(self._modloss_prediction - self._modloss_y))
            
            # output sparsity
            #self._loss += 0.01*tf.reduce_mean(tf.abs(self._prediction))
            #self._optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
            self._optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            #self._optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum = 0.9)
            
            self._mininize = self._optimizer.minimize(self._loss, var_list = tf.global_variables(scope=self.mod_scope))
    
    def InitModel(self, optimizer = 'adam', init_weights_file = None):
        pass
        
    def restart_session(self):
        self.close_session()
        self.init_session()
        
    def init_session(self):
        if self.sess is None:
            self.sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=2))
            
    def close_session(self):
        if self.sess is not None:
            self.sess.close()
            self.sess = None
    
    def SaveVariableName(self, save_as):
        variables_names = [v.name for v in tf.global_variables()]
        with open(save_as+'.varname','w') as f:
            for var in variables_names: f.write(var + '\n')
            
    def SaveTensorName(self, save_as):
        tensor_names = [n.name for n in tf.get_default_graph().as_graph_def().node]
        with open(save_as+'.tenname','w') as f:
            for name in tensor_names: f.write(name + '\n')
            
    def GetVariableByName(self, name):
        l = [v for v in tf.global_variables() if v.name == name]
        if len(l) == 0: return None
        else: return l[0]
    
    def GetTensorByName(self, name):
        return tf.get_default_graph().get_tensor_by_name(name)
    
    def convert_charge_for_tf(self, ch, time_step):
        ch = np.reshape(ch, (-1, 1, 1))
        ch = np.repeat(ch, time_step, axis=1)
        ch.astype(np.float32, copy = False)
        return ch
    
    def mod_feature(self, x, mod_x):
        return np.concatenate((x, mod_x), axis = 2)
    
    def TrainModel(self, buckets, save_as = None):
        
        init_op = tf.variables_initializer(tf.global_variables(scope=self.mod_scope))
        self.init_session()
        self.sess.run(init_op)
        bbatch = Bucket_Batch(buckets, batch_size = self.batch_size, shuffle = True)
        
        # print([v.name for v in tf.global_variables()])
        
        mean_costs = []
        
        var_list = []
        #var_list.append(self.GetVariableByName('backbone/output_rnn/fw/lstm_cell/kernel:0')) #.../gru_cell/gates(candidates)/kernel:0
        #var_list.append(self.GetVariableByName('backbone/output_rnn/bw/lstm_cell/kernel:0')) #.../gru_cell/gates(candidates)/kernel:0
      
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
                peplen = bbatch.get_data_from_batch(batch, 'peplen')
                ch = np.float32(bbatch.get_data_from_batch(batch, 'charge'))
                x = np.float32(bbatch.get_data_from_batch(batch, 'x'))
                mod_x = np.float32(bbatch.get_data_from_batch(batch, 'mod_x'))
                mod_x = self.mod_feature(x, mod_x)
                
                y = bbatch.get_data_from_batch(batch, 'y')
                # backbone_y = self.get_ion_inten(y, self.backbone_idxs)
                
                ch = self.convert_charge_for_tf(ch, peplen[0] - 1)
                feed_dict = {self._x: x, self._y: y, self._charge : ch, self._time_step : peplen - 1, self._mod_x : mod_x}
                
                # if len(self.modloss_idxs) > 0:
                    # modloss_y = self.get_ion_inten(y, self.modloss_idxs)
                    # feed_dict[self._modloss_y] = modloss_y
                    
                self.sess.run(self._mininize, feed_dict = feed_dict)
                cost = self.sess.run(self._loss, feed_dict = feed_dict)
                end_time = time.perf_counter()
                batch_time_cost.append(end_time - start_time)
                batch_cost.append(cost)
                
                #m = self._optimizer.get_slot(self.GetVariableByName('output_rnn/fw/lstm_cell/kernel:0'), 'v')
                #v = self._optimizer.get_slot(self.GetVariableByName('output_rnn/fw/lstm_cell/kernel:0'), 'm')
                #grads = self.sess.run([m,v], feed_dict = feed_dict)
                #print(grads)
                
                var_list = []
                if len(var_list) > 0:
                    grads = self._optimizer.compute_gradients(self._loss, var_list = var_list)
                    grads = self.sess.run(grads, feed_dict = feed_dict)
                    print('\n')
                    for i in range(len(grads)):
                        var = var_list[i]
                        print('mean update of {} = {}'.format(var.name, np.mean(np.abs(grads[i][0]))))
                
                print('Epoch = {:3d}, peplen = {:3d}, Batch={:5d}, size = {:4d}, cost = {:.5f}, time = {:.2f}s\r'.format(epoch + 1, peplen[0], ith_batch, len(x), cost, end_time - start_time), end = "")
            mean_costs.append('Epoch = {:3d}, mean_cost = {:.5f}, time = {:.2f}s'.format(epoch+1, np.mean(batch_cost), np.sum(batch_time_cost)))
            print('\n' + mean_costs[-1])
        
        #print('')
        #for l in mean_costs:
        #    print(l)
            
        if save_as is not None: 
            dir = os.path.dirname(save_as)
            if not os.path.exists(dir): os.makedirs(dir)
            saver = tf.train.Saver()
            save_path = saver.save(self.sess, save_as)
            print('Model save as %s'%save_path)
            self.SaveVariableName(save_as)
            self.SaveTensorName(save_as)
    # model
    def Predict(self, buckets, mod = True):
        print('Predict ... ')
        bbatch = Bucket_Batch(buckets, batch_size = self.batch_size, shuffle = False)
        output_buckets = {}
        while True:
            batch = bbatch.get_next_batch()
            if batch is None: break
            peplen = bbatch.get_data_from_batch(batch, 'peplen')
            ch = np.float32(bbatch.get_data_from_batch(batch, 'charge'))
            x = np.float32(bbatch.get_data_from_batch(batch, 'x'))
            ch = self.convert_charge_for_tf(ch, peplen[0] - 1)
            if mod:
                mod_x = np.float32(bbatch.get_data_from_batch(batch, 'mod_x'))
                mod_x = self.mod_feature(x, mod_x)
                feed_dict = {self._x: x, self._charge : ch, self._time_step : peplen - 1, self._mod_x: mod_x}
                predictions = self.sess.run(self._prediction, feed_dict = feed_dict)
                # if len(self.modloss_idxs) > 0: 
                    # pred_list = [self._prediction, self._modloss_prediction]
                    # predictions = self.sess.run(pred_list, feed_dict = feed_dict)
                    # predictions = np.concatenate(predictions, axis = 2)
                # else:
                    # predictions = self.sess.run(self._prediction, feed_dict = feed_dict)
            else:
                feed_dict = {self._x: x, self._charge : ch, self._time_step : peplen - 1}
                predictions = self.sess.run(self._unmod_prediction, feed_dict = feed_dict)
            predictions[predictions > 1] = 1
            predictions[predictions < 0] = 0
            _buckets = {peplen[0] : (predictions,)}
            output_buckets = merge_buckets(output_buckets, _buckets)
        return output_buckets
    
    def __load_unmod_model__(self, unmod_model):
        self.init_session()
        saver = tf.train.import_meta_graph(unmod_model+'.meta')
        saver.restore(self.sess, unmod_model)
        graph = tf.get_default_graph()
        self._x = graph.get_tensor_by_name("input_x:0")
        self._time_step = graph.get_tensor_by_name("input_time_step:0")
        self._charge = graph.get_tensor_by_name("input_charge:0")
        self._unmod_prediction = graph.get_tensor_by_name("backbone_output:0")
        self._charge_value = graph.get_tensor_by_name("charge_value:0")
    
    # model
    def LoadModel(self, model_file):
        self.init_session()
        saver = tf.train.import_meta_graph(model_file+'.meta')
        saver.restore(self.sess, model_file)
        graph = tf.get_default_graph()
        self._x = graph.get_tensor_by_name("input_x:0")
        self._time_step = graph.get_tensor_by_name("input_time_step:0")
        self._charge = graph.get_tensor_by_name("input_charge:0")
        self._mod_x = graph.get_tensor_by_name("input_mod_x:0")
        self._unmod_prediction = graph.get_tensor_by_name("backbone_output:0")
        self._prediction = graph.get_tensor_by_name(self.mod_scope + "/modion_output:0")
        

