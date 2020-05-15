import argparse
import math
import matplotlib.pyplot as plt
import os
import numpy as np
import shutil
import pandas as pd
import seaborn as sns

sns.set()
sns.set_context("talk")

NUM_BINS = 100
path = '../Data/Video_Info/Pensieve_Info/PenieveVideo_video_info'

video_mappings = {}
video_mappings['300'] = '320x180x30_vmaf_score' 
video_mappings['750'] = '640x360x30_vmaf_score'
video_mappings['1200'] = '768x432x30_vmaf_score'
video_mappings['1850'] = '1024x576x30_vmaf_score'
video_mappings['2850'] = '1280x720x30_vmaf_score'
video_mappings['4300'] = '1280x720x60_vmaf_score'


metric_list = ["reward_vmaf", "reward_br", "rebuf", "br_avg", "vmaf_avg", "switching_vmaf", "switching_br"]
#MINERVA
rebuf_penalty = 25
switching_penalty = 2.5


segment_lenght = 4.0

def load_csv():
    video_info = pd.read_csv(path)
    return video_info

pensieve_video_csv = load_csv()

def get_qoe(abr, trace):
    logdir = os.path.join(args.result_dir, abr + "-" + trace, "result")
    logfile = os.path.join(logdir, abr + "_rewards_0.log")
     
    reward_vmaf = 0
    reward_bitrate = 0
    total_rebuffering = 0.0
    vmaf_avg = 0.0
    vmaf_switching_avg = 0.0
    bitrate_avg = 0.0
    bitrate_switching_avg = 0.0

    with open(logfile, "r") as fin:
        reward_lines = fin.readlines()
        
        if (len(reward_lines) != args.video_chunks):
            if len(reward_lines) < args.video_chunks:
                to_clean.append(logfile)
            print("{} has {} chunks instead of {}".format(logfile, len(reward_lines), args.video_chunks))
            print("Skip, please")
            return None, None, None, None, None, None, None

        for i, r_line in enumerate(reward_lines):
            data = r_line.split()
            if i == 0:
                br = int(data[1])
                br_previous = br
                vmaf_previous = pensieve_video_csv.loc[i, video_mappings[str(br)]]
            else: # skip first
                br = int(data[1])
                bitrate_avg += br
                bitrate_switching_avg += abs(br - br_previous)
                reward_bitrate += float(data[-1])

                total_rebuffering += float(data[3])
                
                vmaf_current = pensieve_video_csv.loc[i, video_mappings[str(br)]]
                vmaf_avg += vmaf_current
                vmaf_switching_avg += abs(vmaf_current - vmaf_previous)

                reward_vmaf +=  (float(vmaf_current) - 
                                rebuf_penalty*(float(data[3])) - 
                                switching_penalty*(abs(vmaf_current - vmaf_previous)))
                
                vmaf_previous = vmaf_current
                br_previous = br

        return  reward_vmaf,\
                reward_bitrate,\
                total_rebuffering,\
                bitrate_switching_avg/(segment_lenght*args.video_chunks),\
                vmaf_switching_avg/(segment_lenght*args.video_chunks),\
                vmaf_avg/(segment_lenght*args.video_chunks),\
                bitrate_avg/args.video_chunks

#
#def get_qoe(abr, trace):
#    logdir = os.path.join(args.result_dir, abr + "-" + trace, "result")
#    logfile = os.path.join(logdir, abr + "_rewards_0.log")
#     
#    reward = 0
#    
#
#    with open(logfile, "r") as fin:
#        reward_lines = fin.readlines()
#        
#        if (len(reward_lines) != args.video_chunks):
#            if len(reward_lines) < args.video_chunks:
#                to_clean.append(logfile)
#            print("{} has {} chunks instead of {}".format(logfile, len(reward_lines), args.video_chunks))
#            print("Skip, please")
#            return None
#
#        for i, r_line in enumerate(reward_lines):
#            if i > 0: # skip first
#                reward += float(r_line.split()[-1])
#    
#    return reward

def get_qoes(abrs_list, traces_list):
    
    global_results = {}
    for abr in abrs_list:
        global_results[abr] = []
        global_results[abr] = {}
        global_results[abr]['reward_vmaf'] = []
        global_results[abr]['reward_br'] = []
        global_results[abr]['rebuf'] = []
        global_results[abr]['switching_br'] = []
        global_results[abr]['switching_vmaf'] = []
        global_results[abr]['vmaf_avg'] = []
        global_results[abr]['br_avg'] = []

    for trace in traces_list:
        reward_vmaf, reward_br, rebuf, switching_br, switching_vmaf, vmaf_avg, br_avg = get_qoe(abr, trace)
        if reward_vmaf is not None:
            global_results[abr]['reward_vmaf'].append(reward_vmaf)
            global_results[abr]['reward_br'].append(reward_br)
            global_results[abr]['rebuf'].append(rebuf)
            global_results[abr]['switching_br'].append(switching_br)
            global_results[abr]['switching_vmaf'].append(switching_vmaf)
            global_results[abr]['vmaf_avg'].append(vmaf_avg)
            global_results[abr]['br_avg'].append(br_avg)

    return global_results

def get_qoes_partial(abrs_list, traces_list):
    
    total_experiments_expected = len(args.abrs) * len(args.traces)
        
    experiments_executed_so_far = 0
    partial_results = {}
    
    for abr in abrs_list:
        
        partial_results[abr] = {}
        partial_results[abr]['reward_vmaf'] = []
        partial_results[abr]['reward_br'] = []
        partial_results[abr]['rebuf'] = []
        partial_results[abr]['switching_br'] = []
        partial_results[abr]['switching_vmaf'] = []
        partial_results[abr]['vmaf_avg'] = []
        partial_results[abr]['br_avg'] = []


        for trace in traces_list:
            
            logdir = os.path.join(args.result_dir, abr + "-" + trace, "result")
            if os.path.exists(logdir):
                reward_vmaf, reward_br, rebuf, switching_br, switching_vmaf, vmaf_avg, br_avg = get_qoe(abr, trace)
                if reward_vmaf is not None:
                    partial_results[abr]['reward_vmaf'].append(reward_vmaf)
                    partial_results[abr]['reward_br'].append(reward_br)
                    partial_results[abr]['rebuf'].append(rebuf)
                    partial_results[abr]['switching_br'].append(switching_br)
                    partial_results[abr]['switching_vmaf'].append(switching_vmaf)
                    partial_results[abr]['vmaf_avg'].append(vmaf_avg)
                    partial_results[abr]['br_avg'].append(br_avg)

                    experiments_executed_so_far += 1
        if partial_results[abr] == []:
            del partial_results[abr]

    print("Experiment executed: {}/{}".format(experiments_executed_so_far, total_experiments_expected))
    return partial_results


def plot_cdf(results, reward_key):
    
    fig = plt.figure(figsize=(16.0, 10.0))
    ax = fig.add_subplot(111)
    
    def average_of_the_best():
        avg_best = -1000000000000
        abr_best = ''
        for scheme in results.keys():
            avg_tmp = np.mean(results[scheme][reward_key])
            if avg_best < avg_tmp:
                avg_best = avg_tmp
                abr_best = scheme
        
        print("Best provider in average is {} with {}".format(abr_best, avg_best))

        return abs(avg_best)

    schemes = []

    norm = average_of_the_best()

    markers = ['.', ',', 'o', 'v', '^', '>', '<', 's', 'x', 'D', 'd', '*', '_', '']

    for i, scheme in enumerate(results.keys()):
        values = [float(i)/norm for i in results[scheme][reward_key]]
        values, base = np.histogram(values, bins=len(values))
        cumulative = np.cumsum(values)
        cumulative = [float(i) / len(values) * 100 for i in cumulative]
        marker_index = i % len(markers)
        ax.plot(base[:-1], cumulative, linewidth=3, marker=markers[marker_index], markevery=2, markersize=15)
        schemes.append(scheme)

    ax.legend(schemes, loc=2)
    ax.set_xlim(-1.0, 1.8)
    plt.ylabel('CDF')
    plt.xlabel('total reward')
    fig.savefig(os.path.join(args.store_dir, 'cdf_{}.png'.format(reward_key)))


def plot_bar(results, metric):
    
    results_metric_avg = {}

    for scheme in results.keys():
        results_metric_avg[scheme] = np.mean(results[scheme][metric])

    fig = plt.figure(figsize=(16.0, 10.0))
    ax = fig.add_subplot(111)
 
    y_pos = np.arange(len(results_metric_avg.keys()))
    ax.bar(y_pos, results_metric_avg.values())
    ax.set_xticks(y_pos)
    ax.set_xticklabels(results_metric_avg.keys())
    fig.savefig(os.path.join(args.store_dir, 'bar_{}.png'.format(metric)))


def clean():
    timestamps = []
    for c in to_clean:
        timestamp_creation = os.path.getmtime(c)
        timestamps.append(timestamp_creation)
        print("File {} was created at {}".format(c, timestamp_creation))
    
    timestamps.sort()
    if not args.include_last and len(timestamps) >= 1:
        print("Skipping file created at {}: might be still running".format(timestamps[-1]))
        del timestamps[-1]

    
    removing = []

    for t in timestamps:
        for c in to_clean:
            if os.path.getmtime(c) == t:
                print("Removing {}".format(os.path.dirname(os.path.dirname(c))))
                removing.append(os.path.dirname(os.path.dirname(c)))
    for r in removing:
        shutil.rmtree(r)
def main():
    
    
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('result_dir', help='result directory', type=str)
    parser.add_argument('store_dir', help='result directory', type=str)
    parser.add_argument('video_chunks', help='result directory', type=int)
    parser.add_argument("--abrs", nargs="+",  help='ABR list')
    parser.add_argument("--traces", nargs="+",  help='Traces list')
    parser.add_argument('--partial', action="store_true", help="get the partial results")
    parser.add_argument('--allow_cleaning', action="store_true", help="if enabled, cleans the experiments that failed, a part of the most recent one (might still be running")
    parser.add_argument('--include_last', action="store_true", help="if enabled, also the last is getting cleaned")

    # args need to be global for simplicity
    global args
    args = parser.parse_args()
    
    global to_clean
    to_clean = []

    if not os.path.exists(args.store_dir):
        os.makedirs(args.store_dir)

    if args.partial:
        res = get_qoes_partial(args.abrs, args.traces)
    else:
        res = get_qoes(args.abrs, args.traces)
    
    for metric in metric_list:
            if "reward" in metric:
                plot_cdf(res, metric)
            plot_bar(res, metric)
    
    if args.allow_cleaning:
        print("Executing cleaning")
        clean()

if __name__ == "__main__":
    main()
