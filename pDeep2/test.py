import model.fragmentation_config as fconfig
import model.lstm_tf as lstm
import model.similarity_calc as sim_calc
import model.load_data as load_data
import model.evaluate as evaluate
import numpy as np
import tensorflow as tf
import time
import os

from model.bucket_utils import count_buckets

n = 1

ion_types = ['b{}', 'y{}']

model_folder = './tf-models/model-no-transfer'
model_name = 'mixed-180322-multi_ce_layer.ckpt'

unmod_config = fconfig.HCD_Config()
unmod_config.time_step = 100
unmod_config.SetIonTypes(ion_types)

mod_config = fconfig.HCD_CommonMod_Config()
mod_config.time_step = 100
mod_config.SetIonTypes(ion_types)
mod_config.min_var_mod_num = 1
mod_config.max_var_mod_num = 3

pho_config = fconfig.HCD_pho_Config()
pho_config.time_step = 100
pho_config.SetIonTypes(ion_types)
pho_config.min_var_mod_num = 1
pho_config.max_var_mod_num = 3

QEHO = "../datasets/Olsen-CellSys-2017/Hela-QE-28/plabel"
QE293O = "../datasets/Olsen-CellSys-2017/293T-QE-28/plabel"
PT25_ts = "../datasets/zengwenfeng-ProteomeTools/plabel/HCD25/test"
PT30_ts = "../datasets/zengwenfeng-ProteomeTools/plabel/HCD30/test"
PT35_ts = "../datasets/zengwenfeng-ProteomeTools/plabel/HCD35/test"
VlsMilkK = "../datasets/Kuster-Human-Nature-2014/milk-Velos-40/plabel"
VlsRtmK = "../datasets/Kuster-Human-Nature-2014/rectum-Velos-30/plabel"
EltStmK = "../datasets/Kuster-Human-Nature-2014/stomach-Elite-30/plabel"
VlsFGP = "../datasets/Pandey-Human-Nature-2014/Fetal_Gut_Gel_Velos_72_CE35/plabel"
VlsFBP = "../datasets/Pandey-Human-Nature-2014/Fetal_Brain_Gel_Velos_16_CE39/plabel"
VlsPhoSyn = "../datasets/Mann-PhoSyn-NBT-2011-Velos-40/plabel"
QEHchyO = "../datasets/Olsen-CellSys-2017/HelaChymo-QE-28/plabel"
QEHgluO = "../datasets/Olsen-CellSys-2017/HelaGluC-QE-28/plabel"
QEHlysO = "../datasets/Olsen-CellSys-2017/HelaLysC-QE-28/plabel"

pdeep = lstm.IonLSTM(unmod_config)

plot_folder = os.path.join(model_folder, 'log/plots/%s'%model_name)
try:
    os.makedirs(plot_folder)
except:
    pass

pdeep.LoadModel(model_file = os.path.join(model_folder, model_name))

with open(os.path.join(model_folder, 'log/test_%s.txt'%model_name),'w') as log_out:

    def test(folder, ce, ins, n, saveplot, phos = False):
        print('###################### Begin Unmod ######################', file = log_out)
        print("[D] " + folder, file = log_out)
        print("[T] Unmod PSMs:", file = log_out)
        buckets = load_data.load_folder_as_buckets(folder, unmod_config, nce = ce, instrument = ins, max_n_samples = n)
        print("[C] " + str(count_buckets(buckets)), file = log_out)
        output_buckets = pdeep.Predict(buckets)
        pcc, cos, spc, kdt, SA = sim_calc.CompareRNNPredict_buckets_tf(output_buckets, buckets)
        sim_names = ['PCC', 'COS', 'SPC', 'KDT', 'SA']
        print("[A] " + str(evaluate.cum_plot([pcc, cos, spc, kdt, SA], sim_names, saveplot = os.path.join(plot_folder, saveplot+'.eps'), print_file = log_out)), file = log_out)
        print('####################### End Unmod #######################', file = log_out)
        print("", file = log_out)
        
        print('####################### Begin Mod #######################', file = log_out)
        print("[D] " + folder, file = log_out)
        if phos: 
            print("[T] Phos PSMs:", file = log_out)
            config = pho_config
            mod = '-pho'
        else:
            print("[T] Mod PSMs:", file = log_out)
            config = mod_config
            mod = '-mod'
        buckets = load_data.load_folder_as_buckets(folder, config, nce = ce, instrument = ins, max_n_samples = n)
        print("[C] " + str(count_buckets(buckets)), file = log_out)
        output_buckets = pdeep.Predict(buckets)
        pcc, cos, spc, kdt, SA = sim_calc.CompareRNNPredict_buckets_tf(output_buckets, buckets)
        sim_names = ['PCC', 'COS', 'SPC', 'KDT', 'SA']
        print("[A] " + str(evaluate.cum_plot([pcc, cos, spc, kdt, SA], sim_names, saveplot = os.path.join(plot_folder, saveplot+mod+'.eps'), print_file = log_out)), file = log_out)
        print('######################## End Mod ########################', file = log_out)
        print("\n", file = log_out)

    start_time = time.perf_counter()

    ################### start one folder ##############################
    test_folder,ce,ins = QEHO,0.28,'QE'
    test(test_folder, ce, ins, n, "QE-H-O", phos = True)
    ################### end one folder ################################

    ################### start one folder ##############################
    test_folder,ce,ins = QE293O,0.28,'QE'
    test(test_folder, ce, ins, n, "QE-293-O", phos = True)
    ################### end one folder ################################


    ################### start one folder ##############################
    test_folder,ce,ins = PT25_ts,0.25,'Lumos'
    test(test_folder, ce, ins, n, "PT25", phos = False)
    ################### end one folder ################################

    ################### start one folder ##############################
    test_folder,ce,ins = PT30_ts,0.30,'Lumos'
    test(test_folder, ce, ins, n, "PT30", phos = False)
    ################### end one folder ################################

    ################### start one folder ##############################
    test_folder,ce,ins = PT35_ts,0.35,'Lumos'
    test(test_folder, ce, ins, n, "PT35", phos = False)
    ################### end one folder ################################


    ################### start one folder ##############################
    test_folder,ce,ins = VlsMilkK,0.40,'Velos'
    test(test_folder, ce, ins, n, "Velos-Milk-Ku", phos = False)
    ################### end one folder ################################


    ################### start one folder ##############################
    test_folder,ce,ins = VlsRtmK,0.30,'Velos'
    test(test_folder, ce, ins, n, "Velos-Rectum-Ku", phos = False)
    ################### end one folder ################################


    ################### start one folder ##############################
    test_folder,ce,ins = EltStmK,0.30,'Elite'
    test(test_folder, ce, ins, n, "Elite-Stm-Ku", phos = False)
    ################### end one folder ################################
    
    
    ################### start one folder ##############################
    test_folder,ce,ins = VlsFGP,0.35,'Velos'
    test(test_folder, ce, ins, n, "Velos-FtlGut-P", phos = False)
    ################### end one folder ################################
    
    
    ################### start one folder ##############################
    test_folder,ce,ins = VlsFBP,0.39,'Velos'
    test(test_folder, ce, ins, n, "Velos-FtlBrain-P", phos = False)
    ################### end one folder ################################
    
    
    ################### start one folder ##############################
    test_folder,ce,ins = VlsPhoSyn,0.40,'Velos'
    test(test_folder, ce, ins, n, "Velos-PhosSyn", phos = True)
    ################### end one folder ################################
    

    ################### start one folder ##############################
    test_folder,ce,ins = QEHchyO,0.28,'QE'
    test(test_folder, ce, ins, n, "QE-Hchy-O", phos = False)
    ################### end one folder ################################

    ################### start one folder ##############################
    test_folder,ce,ins = QEHgluO,0.28,'QE'
    test(test_folder, ce, ins, n, "QE-Hglu-O", phos = False)
    ################### end one folder ################################
    

    ################### start one folder ##############################
    test_folder,ce,ins = QEHlysO,0.28,'QE'
    test(test_folder, ce, ins, n, "QE-Hlys-O", phos = False)
    ################### end one folder ################################

    end_time = time.perf_counter()

    print("time = {:.3f}s".format(end_time - start_time))

pdeep.close_session()
