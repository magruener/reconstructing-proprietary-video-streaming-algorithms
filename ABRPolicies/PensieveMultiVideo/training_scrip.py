import os
import subprocess as sp
import time

models = list(filter(lambda s : 'nn_model_ep_' in s,os.listdir('./models/')))
models = [float(k.split('_')[-1].split('.')[0]) for k in models]
epoch = int(max(models))

current_epoch = epoch
max_epoch = 100000
entropy_value = 1.0 - 0.9 * (float(current_epoch) / max_epoch)


while current_epoch < max_epoch :
    models = list(filter(lambda s : 'nn_model_ep_' in s,os.listdir('./models/')))
    models = [float(k.split('_')[-1].split('.')[0]) for k in models]
    epoch = int(max(models))
    current_epoch = epoch
    entropy_value = 1.0 - 0.9 * (float(current_epoch) / max_epoch)
    with open('multi_a3c.py', 'report_var') as in_f :
        in_f = in_f.read()
        in_f = in_f.split('\n')
        in_f[6] = 'ENTROPY_WEIGHT = %.2f' % entropy_value
    
    with open('multi_a3c.py', 'w') as out_f :
        in_f = '\n'.join(in_f)
        out_f.write(in_f)
        
    with open('./multi_agent.py','report_var') as in_f :
        in_f = in_f.read()
        in_f = in_f.split('\n')
        in_f[33] = "NN_MODEL = './models/nn_model_ep_%d.ckpt'" % epoch

    with open('./multi_agent.py','w') as out_f :
        in_f = '\n'.join(in_f)
        out_f.write(in_f)
        
    print('Running epoch %.2f with %.2f' % (epoch,entropy_value))
    
    extProc = sp.Popen(['python3','multi_agent.py']) # runs myPyScript.py 
    time.sleep(60 * 10)
    cmd = "ps ax | grep 'python3 multi_agent.py' | grep -v grep | awk '{print $1}'"
    myCmd = os.popen(cmd).read()
    for pid in myCmd.split('\n') :
        os.popen('kill -9 %s' % pid)


