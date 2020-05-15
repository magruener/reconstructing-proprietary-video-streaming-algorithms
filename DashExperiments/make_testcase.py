import argparse
import os
from json import dump
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("main_dir_name", help="Directory in which all the testcases are stored", type=str)
    parser.add_argument("trace_name", help="Name of the trace", type=str)
    parser.add_argument("test_video", help="Video under test", type=str)
    parser.add_argument("repeat_n", help="Number of time the test has to be repeated", type=str)
    parser.add_argument("udp", help="Has UDP infrastructure enabled?", type=bool)
    parser.add_argument("name", help="Name of the experiment", type=str)
    parser.add_argument("abr", help=" BB | BOLA | RB| robustMPC | RL | dummy_tree", type=str)
    parser.add_argument("quic", help="Has QUIC enabled?", type=bool)
    parser.add_argument("transport", help="Which transport protocol", type=str)
    parser.add_argument("start", help="0", type=str)
    parser.add_argument("duration", help="200", type=str)
    parser.add_argument("server_module", help="path to the server module", type=str)

    args = parser.parse_args()
    filename = args.abr + '-'  + args.trace_name
    filepath = os.path.join(args.main_dir_name, filename + '.case')
    
    if not os.path.exists(args.main_dir_name):
        os.makedirs(args.main_dir_name, 0o777)
    
    structure_case = {}
    structure_case['test_id'] = filename
    structure_case['comment'] = ''
    structure_case['trace'] = args.trace_name
    structure_case['video'] = args.test_video
    structure_case['repeat_n'] = args.repeat_n
    structure_case['jobs'] = []
    job = {}
    job['udp'] = args.udp
    job['name'] = args.name
    job['abr'] = args.abr
    job['quic'] = args.quic
    job['transport'] = args.transport
    job['start'] = args.start
    job['duration'] = args.duration
    job['server_module'] = args.server_module

    structure_case['jobs'].append(job)

    with open(filepath, 'w') as fout:
        dump(structure_case, fout)
    
    
    
    





