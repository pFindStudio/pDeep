from scipy.stats.stats import pearsonr, spearmanr, kendalltau
import tensorflow as tf
import numpy as np

def cosine(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2))
    
def similarity(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    pcc = pearsonr(v1, v2)[0]
    cos = cosine(v1, v2)
    spc = spearmanr(v1, v2)[0]
    kdt = kendalltau(v1, v2)[0]
    return (pcc, cos, spc, kdt)
    
def CompareRNNPredict_buckets_tf(predict_buckets, real_buckets):
    sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=2))
    pccs = []
    spcs = []
    coses = []
    kdts = []
    SAs = []
    
    _x = tf.placeholder('float', [None, ])
    _y = tf.placeholder('float', [None, ])
    _len = tf.placeholder(tf.int32,())
        
    def spearman(x, y):
        y_rank = tf.nn.top_k(y, k = _len, sorted=True, name='y_rank').indices
        x_rank = tf.nn.top_k(x, k = _len, sorted=True, name='x_rank').indices
        rank_diff = y_rank - x_rank
        rank_diff_sq_sum = tf.reduce_sum(rank_diff * rank_diff)
        six = tf.constant(6)
        one = tf.constant(1.0)
        numerator = tf.cast(six * rank_diff_sq_sum, dtype = tf.float32)
        divider = tf.cast(_len*_len*_len - _len, dtype = tf.float32)
        return one - numerator / divider
        
    def cosine(x, y):
        norm_x = tf.nn.l2_normalize(x,0)
        norm_y = tf.nn.l2_normalize(y,0)
        return tf.reduce_sum(tf.multiply(norm_x, norm_y))
        
    def pearson(x, y):
        x = x - tf.reduce_mean(x)
        y = y - tf.reduce_mean(y)
        return cosine(x, y)
        
    def spectral_angle(cos_val):
        return 1 - 2*tf.acos(cos_val)/np.pi
        
    _pcc = pearson(_x, _y)
    _cos = cosine(_x, _y)
    _SA = spectral_angle(_cos)
    #_spc = spearman(_x, _y) # bad performance
    
    #init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    #sess.run(init_op)
    
    def similarity_tf(v1, v2):
        ret_pcc, ret_cos, ret_SA = sess.run([_pcc, _cos, _SA], feed_dict = {_x:v1, _y:v2, _len:len(v1)})
        ret_spc = spearmanr(v1, v2)[0]
        ret_kdt = kendalltau(v1, v2)[0]
        return ret_pcc, ret_cos, ret_spc, ret_kdt, ret_SA
    
    for key, value in predict_buckets.items():
        predict = value[-1]
        predict[predict < 1e-4] = 0
        real = real_buckets[key][-1]
        ypred_seq = np.reshape(predict, (predict.shape[0], predict.shape[1] * predict.shape[2]), order='C')
        ytest_seq = np.reshape(real, (real.shape[0], real.shape[1] * real.shape[2]), order='C')
        tmp_pcc = []
        tmp_spc = []
        for i in range(len(predict)):
            pcc, cos, spc, kdt, SA = similarity_tf(ypred_seq[i], ytest_seq[i])
            tmp_pcc.append(pcc)
            tmp_spc.append(spc)
            pccs.append(pcc)
            spcs.append(spc)
            coses.append(cos)
            kdts.append(kdt)
            SAs.append(SA)
        print('[I] peplen = {}, size = {}, Median: pcc = {:.3f}, spc = {:.3f}'.format(key, len(predict), np.median(tmp_pcc), np.median(tmp_spc)))
    
    sess.close()
    
    pccs, coses, spcs, kdts, SAs = np.array(pccs), np.array(coses), np.array(spcs), np.array(kdts), np.array(SAs)
    out_median = "[R] Median: pcc = {:.3f}, cos = {:.3f}, spc = {:.3f}, kdt = {:.3f}, SA = {:.3f}".format(np.median(pccs), np.median(coses), np.median(spcs), np.median(kdts), np.median(SAs))
    out_mean = "[R] Mean: pcc = {:.3f}, cos = {:.3f}, spc = {:.3f}, kdt = {:.3f}, SA = {:.3f}".format(np.mean(pccs), np.mean(coses), np.mean(spcs), np.mean(kdts), np.mean(SAs))
    
    print(out_median)
    print(out_mean)
    return (pccs, coses, spcs, kdts, SAs)

def CompareRNNPredict_buckets(predict_buckets, real_buckets):
    pccs = []
    spcs = []
    coses = []
    kdts = []
    for key, value in predict_buckets.items():
        predict = value[-1]
        predict[predict < 1e-4] = 0
        real = real_buckets[key][-1]
        ypred_seq = np.reshape(predict, (predict.shape[0], predict.shape[1] * predict.shape[2]), order='C')
        ytest_seq = np.reshape(real, (real.shape[0], real.shape[1] * real.shape[2]), order='C')
        tmp_pcc = []
        tmp_spc = []
        for i in range(len(predict)):
            pcc, cos, spc, kdt = similarity(ypred_seq[i], ytest_seq[i])
            tmp_pcc.append(pcc)
            tmp_spc.append(spc)
            pccs.append(pcc)
            spcs.append(spc)
            coses.append(cos)
            kdts.append(kdt)
        print('[I] peplen = {}, size = {}, Median: pcc = {:.3f}, spc = {:.3f}'.format(key, len(predict), np.median(tmp_pcc), np.median(tmp_spc)))
    pccs, coses, spcs, kdts = np.array(pccs), np.array(coses), np.array(spcs), np.array(kdts)
    out_median = "[R] Median: pcc = {:.3f}, cos = {:.3f}, spc = {:.3f}, kdt = {:.3f}".format(np.median(pccs), np.median(coses), np.median(spcs), np.median(kdts))
    out_mean = "[R] Mean: pcc = {:.3f}, cos = {:.3f}, spc = {:.3f}, kdt = {:.3f}".format(np.mean(pccs), np.mean(coses), np.mean(spcs), np.mean(kdts))
    
    print(out_median)
    print(out_mean)
    return (pccs, coses, spcs, kdts)
    
def CompareRNNPredict(predict, real, peplens):
    print("predict shape", predict.shape)
    ypred_seq = np.reshape(predict, (predict.shape[0], predict.shape[1] * predict.shape[2]), order='C')
    ytest_seq = np.reshape(real, (real.shape[0], real.shape[1] * real.shape[2]), order='C')
    pccs = []
    spcs = []
    coses = []
    kdts = []
    for i in range(len(predict)):
        # sim = pearsonr(np.reshape(predict[i,:(peplens[i]-1),:],-1), np.reshape(real[i,:(peplens[i]-1),:],-1))[0]
        pcc, cos, spc, kdt = similarity(ypred_seq[i][:(peplens[i]-1) * predict.shape[2]], ytest_seq[i][:(peplens[i]-1) * predict.shape[2]])
        pccs.append(pcc)
        spcs.append(spc)
        coses.append(cos)
        kdts.append(kdt)
    sims_nan = np.array(pccs)
    sims = sims_nan[np.isnan(sims_nan) == False]
    med = np.median(sims)
    mad = np.median(np.abs(sims - med))
    avg = np.mean(sims)
    std = np.std(sims)
    out_median = "    Median pcc = %.3f, MAD pcc = %.3f" %(med, mad)
    out_mean = "    Mean pcc = %.3f, STD pcc = %.3f" %(avg, std)
    print(out_median)
    print(out_mean)
    return (np.array(pccs), np.array(coses), np.array(spcs), np.array(kdts))
    