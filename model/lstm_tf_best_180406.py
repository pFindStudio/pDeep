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
    
    # model
    def BuildModel(self, input_size, output_size, nlayers = 2):
        print("This model is for testing and transfer-learning only !!!")
    
    def BuildTransferModel(self, model_file):
        print("Fine-tuning pDeep model ...")
        self.LoadModel(model_file)
        
        transfer_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "output_scope/backbone_rnn")
        transfer_vars += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.instrument_ce_scope)
        
        self._loss = tf.reduce_mean(tf.abs(self._prediction - self._y))
        
        self._optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, name = "transfer_Adam")
        
        self._mininize = self._optimizer.minimize(self._loss, var_list = transfer_vars)
        
        self.merged_summary_op = tf.summary.merge_all()
        
        adam_vars = [self.GetVariableByName("beta1_power_1:0"), self.GetVariableByName("beta2_power_1:0")] + self.GetVariablesBySubstr("transfer_Adam")
        init_op = tf.variables_initializer(var_list = adam_vars, name = "transfer_init")
        self.sess.run(init_op)
        self.summary_writer = tf.summary.FileWriter("../tensorboard/transfer_train", self.sess.graph)
    
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
        self._y = graph.get_tensor_by_name("Placeholder:0")
        self._time_step = graph.get_tensor_by_name("input_time_step:0")
        self._charge = graph.get_tensor_by_name("input_charge:0")
        self._instrument = graph.get_tensor_by_name("input_instrument:0")
        self._nce = graph.get_tensor_by_name("input_nce:0")
        self._prediction = graph.get_tensor_by_name("output_scope/output:0")
        self.rnn_kp = graph.get_tensor_by_name("rnn_keep_prob:0")
        self.output_kp = graph.get_tensor_by_name("output_keep_prob:0")
        

