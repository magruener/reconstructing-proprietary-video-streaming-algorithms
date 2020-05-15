#!/usr/bin/env python3


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
from http.server import BaseHTTPRequestHandler, HTTPServer
import socketserver
import base64
import urllib
import sys
import dill
sys.path.append('../')

from SimulationEnviroment.SimulatorEnviroment import StreamingEnviroment
from BehaviourCloning.MLABRPolicy import ABRPolicyClassifierSimple
import os
import json
import time
import random
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse
import numpy as np
import time
import itertools
import datetime
import logging
################## ROBUST MPC ###################

S_INFO = 5  # bit_rate, buffer_size, rebuffering_time, bandwidth_measurement, chunk_til_video_end
S_LEN = 8  # take how many frames in the past
MPC_FUTURE_CHUNK_COUNT = 5
VIDEO_BIT_RATE = [300,750,1200,1850,2850,4300]  # Kbps
BITRATE_REWARD = [1, 2, 3, 12, 15, 20]
BITRATE_REWARD_MAP = {0: 0, 300: 1, 750: 2, 1200: 3, 1850: 12, 2850: 15, 4300: 20}
M_IN_K = 1000.0
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
TOTAL_VIDEO_CHUNKS = 48
DEFAULT_QUALITY = 0  # default video quality without agent
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> this number of Mbps
SMOOTH_PENALTY = 1
TRAIN_SEQ_LEN = 100  # take as a train batch
MODEL_SAVE_INTERVAL = 100
RANDOM_SEED = 42
RAND_RANGE = 1000
SUMMARY_DIR = './results'
LOG_FILE = './results/log'


MILLISECONDS_IN_SEC = 1000
MAX_LOOKAHEAD = 3
MAX_LOOKBACK = 10
MAX_SWITCHES = 2

TREE_FOLDER = './src/video_server/trees/'

VIDEO_CSV_PATH = '../Data/Video_Info/Pensieve_Info/PenieveVideo_video_info'
TRACE_DUMMY = os.path.join(TREE_FOLDER, 'dummy_trace')

# in format of time_stamp bit_rate buffer_size rebuffer_time video_chunk_size future_download_time reward
NN_MODEL = None

CHUNK_COMBO_OPTIONS = []

# past errors in bandwidth
past_errors = []
past_bandwidth_ests = []


class OnlineStreaming(StreamingEnviroment):
    def __init__(self, video_information_csv_path, max_lookback: int,
                 max_lookahead: int, max_switch_allowed: int):
        logger.info("Initializing supeclass")
        super().__init__(TRACE_DUMMY, video_information_csv_path, None,  max_lookback, max_lookahead,
                         max_switch_allowed, 60*MILLISECONDS_IN_SEC)
        """
        Padding and stuff
        """
        self.timestamp_arr = [0] * self.max_lookback
        self.data_used_relative = [0] * self.max_lookback
        self.quality_arr = [0] * self.max_lookback
        self.downloadtime_arr = [0] * self.max_lookback
        self.sleep_time_arr = [0] * self.max_lookback
        self.buffer_size_arr = [0] * self.max_lookback
        self.rebuffer_time_arr = [0] * self.max_lookback
        self.video_chunk_remain_arr_relative = [0] * self.max_lookback
        self.video_chunk_size_arr = [0] * self.max_lookback
        self.rate_played_relative_arr = [0] * self.max_lookback
        self.segment_length_arr = [0] * self.max_lookback
        self.encoded_mbitrate_arr = [0] * self.max_lookback
        self.single_mbitrate_arr = [0] * self.max_lookback
        self.vmaf_arr = [0] * self.max_lookback

    def copy(self):
        return OnlineStreaming( self.video_information_csv_path, 
                                self.max_lookback,
                                self.max_lookahead, self.max_switch_allowed)

    def set_state(self, state):
        self.video_chunk_counter = state['video_chunk_counter']
        self.mahimahi_ptr = state['mahimahi_ptr']
        self.buffer_size_ms = state['buffer_size_ms']
        self.last_mahimahi_time = state['last_mahimahi_time']
        self.last_quality = state['last_quality']
        self.timestamp_s = state['timestamp_s']
        self.data_used_bytes = state['data_used_bytes_relative']
        self.timestamp_arr = self.timestamp_arr[:state['timestamp_s_arr_ptr']]
        self.data_used_relative = self.data_used_relative[:state['data_used_bytes_arr_ptr']]
        self.quality_arr = self.quality_arr[:state['quality_arr_ptr']]
        self.downloadtime_arr = self.downloadtime_arr[:state['downloadtime_arr_ptr']]
        self.sleep_time_arr = self.sleep_time_arr[:state['sleep_time_arr_ptr']]
        self.buffer_size_arr = self.buffer_size_arr[:state['buffer_size_arr_ptr']]
        self.rebuffer_time_arr = self.rebuffer_time_arr[:state['rebuffer_time_arr_ptr']]
        self.video_chunk_remain_arr_relative = self.video_chunk_remain_arr_relative[:state['video_chunk_ptr']]
        self.video_chunk_size_arr = self.video_chunk_size_arr[:state['video_chunk_size_ptr']]
        self.rate_played_relative_arr = self.rate_played_relative_arr[:state['rate_played_relative_ptr']]
        self.segment_length_arr = self.segment_length_arr[:state['segment_length_ptr']]
        self.encoded_mbitrate_arr = self.encoded_mbitrate_arr[:state['encoded_mbitrate_ptr']]
        self.single_mbitrate_arr = self.single_mbitrate_arr[:state['single_mbitrate_ptr']]
        self.vmaf_arr = self.vmaf_arr[:state['vmaf_ptr']]
        assert len(self.logging_file) >= state['logging_file_ptr'], 'We somehow lost logging data on the way'
        self.logging_file = self.logging_file[:state['logging_file_ptr']]

    def save_state(self):
        return {'mahimahi_ptr': self.mahimahi_ptr,
                'buffer_size_ms': self.buffer_size_ms,
                'last_mahimahi_time': self.last_mahimahi_time,
                'last_quality': self.last_quality,
                'video_chunk_counter': self.video_chunk_counter,
                'timestamp_s': self.timestamp_s,
                'data_used_bytes_relative': self.data_used_bytes,
                'video_chunk_size_ptr': len(self.video_chunk_size_arr),
                'timestamp_s_arr_ptr': len(self.timestamp_arr),
                'data_used_bytes_arr_ptr': len(self.data_used_relative),
                'quality_arr_ptr': len(self.quality_arr),
                'downloadtime_arr_ptr': len(self.downloadtime_arr),
                'sleep_time_arr_ptr': len(self.sleep_time_arr),
                'buffer_size_arr_ptr': len(self.buffer_size_arr),
                'rebuffer_time_arr_ptr': len(self.rebuffer_time_arr),
                'video_chunk_ptr': len(self.video_chunk_remain_arr_relative),
                'logging_file_ptr': len(self.logging_file),
                'rate_played_relative_ptr': len(self.rate_played_relative_arr),
                'segment_length_ptr': len(self.segment_length_arr),
                'encoded_mbitrate_ptr': len(self.encoded_mbitrate_arr),
                'single_mbitrate_ptr': len(self.single_mbitrate_arr),
                'vmaf_ptr': len(self.vmaf_arr)}

    def get_video_chunk(self, quality, download_time_ms, rebuffer_time_ms, buffer_size_ms, timestamp_s, activate_logging=True):

        assert quality >= 0

        video_chunk_size = self.byte_size_match.iloc[self.video_chunk_counter, quality]
        relative_encoded_bitrate = self.get_encoded_bitrate(quality) / self.get_encoded_bitrate(-1)
        segment_length_ms = self.video_information_csv.iloc[
                                self.video_chunk_counter].seg_len_s * 1000.
        encoded_mbitrate = self.get_encoded_bitrate(quality) * 1e-6
        current_mbitrate = self.bitrate_match.iloc[self.video_chunk_counter, quality] * 1e-6
        vmaf = self.vmaf_match.iloc[self.video_chunk_counter, quality]



        # add in the new chunk

        self.buffer_size_ms =  buffer_size_ms  # buffer size is in ms
        # sleep if buffer gets too large -> in a real environment it never get bigger than the actual one 
        sleep_time_ms = 0
        

        # the "last buffer size" return to the controller
        # Note: in old version of dash the lowest buffer is 0.
        # In the new version the buffer always have at least
        # one chunk of video

        self.video_chunk_counter += 1
        video_chunk_remain = self.n_video_chunk - self.video_chunk_counter

        end_of_video = False
        if self.video_chunk_counter >= self.n_video_chunk:
            end_of_video = True

        self.data_used_bytes += video_chunk_size
        self.timestamp_s = timestamp_s
        
        self.timestamp_arr.append(self.timestamp_s)
        self.data_used_relative.append(self.data_used_bytes / self.max_data_used)
        self.quality_arr.append(quality)
        self.downloadtime_arr.append(download_time_ms / MILLISECONDS_IN_SEC)
        self.sleep_time_arr.append(sleep_time_ms / MILLISECONDS_IN_SEC)
        self.buffer_size_arr.append(self.buffer_size_ms / MILLISECONDS_IN_SEC)
        self.rebuffer_time_arr.append(rebuffer_time_ms / MILLISECONDS_IN_SEC)
        self.video_chunk_size_arr.append(video_chunk_size)
        self.video_chunk_remain_arr_relative.append(video_chunk_remain / float(self.n_video_chunk))
        self.rate_played_relative_arr.append(relative_encoded_bitrate)
        self.segment_length_arr.append(segment_length_ms / 1000.)
        self.encoded_mbitrate_arr.append(encoded_mbitrate)
        self.single_mbitrate_arr.append(current_mbitrate)
        self.vmaf_arr.append(vmaf)

        observation = self.generate_observation_dictionary()



        self.last_quality = quality
        return observation,  end_of_video

    def generate_observation_dictionary(self):
        quality = self.quality_arr[-1]
        observation = []
        observation.append(self.timestamp_arr[-self.max_lookback:])
        observation.append(self.data_used_relative[-self.max_lookback:])
        observation.append(self.quality_arr[-self.max_lookback:])
        observation.append(self.downloadtime_arr[-self.max_lookback:])
        observation.append(self.sleep_time_arr[-self.max_lookback:])
        observation.append(self.buffer_size_arr[-self.max_lookback:])
        observation.append(self.rebuffer_time_arr[-self.max_lookback:])
        observation.append(self.video_chunk_size_arr[-self.max_lookback:])
        observation.append(self.video_chunk_remain_arr_relative[-self.max_lookback:])
        observation.append(self.rate_played_relative_arr[-self.max_lookback:])
        observation.append(self.segment_length_arr[-self.max_lookback:])
        observation.append(self.encoded_mbitrate_arr[-self.max_lookback:])
        observation.append(self.single_mbitrate_arr[-self.max_lookback:])
        observation.append(self.vmaf_arr[-self.max_lookback:])
        for switch in np.arange(quality - self.max_switch_allowed, quality + self.max_switch_allowed + 1):
            switch = np.clip(switch, a_min=0, a_max=self.max_quality_level)
            switch = int(switch)
            future_chunk_size_arr = []
            for lookahead in range(0, self.max_lookahead):
                if self.video_chunk_counter + lookahead < self.n_video_chunk:
                    future_chunk_size = self.byte_size_match.iloc[self.video_chunk_counter + lookahead, switch]
                else:
                    future_chunk_size = 0
                future_chunk_size_arr.append(future_chunk_size)
            observation.append(future_chunk_size_arr)
        for switch in np.arange(quality - self.max_switch_allowed, quality + self.max_switch_allowed + 1):
            switch = np.clip(switch, a_min=0, a_max=self.max_quality_level)
            switch = int(switch)
            future_chunk_size_arr = []
            for lookahead in range(0, self.max_lookahead):
                if self.video_chunk_counter + lookahead < self.n_video_chunk:
                    future_chunk_size = self.bitrate_match.iloc[self.video_chunk_counter + lookahead, switch]
                else:
                    future_chunk_size = 0
                future_chunk_size_arr.append(future_chunk_size)
            observation.append(future_chunk_size_arr)
        for switch in np.arange(quality - self.max_switch_allowed, quality + self.max_switch_allowed + 1):
            switch = np.clip(switch, a_min=0, a_max=self.max_quality_level)
            switch = int(switch)
            future_chunk_size_arr = []
            for lookahead in range(0, self.max_lookahead):
                if self.video_chunk_counter + lookahead < self.n_video_chunk:
                    future_chunk_size = self.vmaf_match.iloc[self.video_chunk_counter + lookahead, switch]
                else:
                    future_chunk_size = 0
                future_chunk_size_arr.append(future_chunk_size)
            observation.append(future_chunk_size_arr)

        observation.append(self)
        observation = {obs_key: obs_value for obs_key, obs_value in zip(self.get_obs_names(), observation)}
        return observation

    def get_past_dims(self):
        return len([v for v in self.get_obs_names() if 'future' not in v]) - 1

    def get_future_dims(self):
        return len([v for v in self.get_obs_names() if 'future' in v])

    def get_obs_names(self):
        column_switches = ['future_chunk_size_byte_switch_%d' % switch for switch in np.arange(
            -self.max_switch_allowed, self.max_switch_allowed + 1)]
        column_switches += ['future_chunk_bitrate_switch_%d' % switch for switch in np.arange(
            -self.max_switch_allowed, self.max_switch_allowed + 1)]
        column_switches += ['future_chunk_vmaf_switch_%d' % switch for switch in np.arange(
            -self.max_switch_allowed, self.max_switch_allowed + 1)]
        return ['timestamp_s', 'data_used_bytes_relative', 'current_level', 'download_time_s', 'sleep_time_s',
                'buffer_size_s', 'rebuffer_time_s', 'video_chunk_size_byte',
                'relative_chunk_remain', 'relative_rate_played', 'segment_length_s', 'encoded_mbitrate',
                'single_mbitrate',
                'vmaf'] + column_switches + [
                   'streaming_environment']
    


    def reset(self):
        self.video_chunk_counter = 0
        self.buffer_size_ms = 0
        self.mahimahi_ptr = 1
        self.last_quality = 0
        self.logging_file = []
        self.timestamp_s = 0
        self.data_used_bytes = 0

        """
        Padding and stuff
        """
        self.timestamp_arr = [0] * self.max_lookback
        self.data_used_relative = [0] * self.max_lookback
        self.quality_arr = [0] * self.max_lookback
        self.downloadtime_arr = [0] * self.max_lookback
        self.sleep_time_arr = [0] * self.max_lookback
        self.buffer_size_arr = [0] * self.max_lookback
        self.rebuffer_time_arr = [0] * self.max_lookback
        self.video_chunk_remain_arr_relative = [0] * self.max_lookback
        self.video_chunk_size_arr = [0] * self.max_lookback
        self.rate_played_relative_arr = [0] * self.max_lookback
        self.segment_length_arr = [0] * self.max_lookback
        self.encoded_mbitrate_arr = [0] * self.max_lookback
        self.single_mbitrate_arr = [0] * self.max_lookback
        self.vmaf_arr = [0] * self.max_lookback




def make_request_handler(input_dict):

    class Request_Handler(BaseHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            self.input_dict = input_dict
            self.log_file = input_dict['log_file']
            #self.saver = input_dict['saver']
            self.s_batch = input_dict['s_batch']
            #self.a_batch = input_dict['a_batch']
            #self.r_batch = input_dict['r_batch']
            BaseHTTPRequestHandler.__init__(self, *args, **kwargs)

        def do_POST(self):
            content_length = int(self.headers['Content-Length'])
            post_data = json.loads(self.rfile.read(content_length))
            
            if len(post_data) == 1: # message comes from the controller that suggests the optimal bitrate
                self.send_response(200)
                send_data = ''
                self.send_header('Content-Type', 'text/plain')
                self.send_header('Content-Length', len(send_data))
                self.send_header('Access-Control-Allow-Origin', "*")
                self.end_headers()
                self.wfile.write(send_data.encode('utf-8'))
                return

            if ( 'pastThroughput' in post_data ):
                # @Hongzi: this is just the summary of throughput/quality at the end of the load
                # so we don't want to use this information to send back a new quality
                
                print("Summary: {} ".format(post_data))
            else:
               
                if not 'streaming_env' in input_dict.keys():
                    logger.info("Creating streaming enviroment")
                    input_dict['streaming_env'] = OnlineStreaming(VIDEO_CSV_PATH, MAX_LOOKBACK, MAX_LOOKAHEAD, MAX_SWITCHES)
                if not 'classifier' in input_dict.keys():
                    with open(TREE_FILENAME, 'rb') as fin:
                        logger.info("Loading classifier")
                        clas = dill.load(fin)
                    logger.info("Creating ABR Policy Classifier")
                    input_dict['classifier'] = clas 
                if not 'time_start' in input_dict.keys() :
                    input_dict['time_start'] = datetime.datetime.now()

                streaming_env = input_dict['streaming_env']
                classifier = input_dict['classifier']
                time_start = input_dict['time_start']

                rebuffer_time = float(post_data['RebufferTime'] -self.input_dict['last_total_rebuf'])

                reward = VIDEO_BIT_RATE[post_data['lastquality']] / M_IN_K \
                        - REBUF_PENALTY * rebuffer_time / M_IN_K \
                        - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[post_data['lastquality']] -
                                                  self.input_dict['last_bit_rate']) / M_IN_K

                self.input_dict['last_bit_rate'] = VIDEO_BIT_RATE[post_data['lastquality']]
                self.input_dict['last_total_rebuf'] = post_data['RebufferTime']


                # compute bandwidth measurement
                video_chunk_fetch_time = post_data['lastChunkFinishTime'] - post_data['lastChunkStartTime']
                video_chunk_size = post_data['lastChunkSize']

                # compute number of video chunks left
                video_chunk_remain = TOTAL_VIDEO_CHUNKS - self.input_dict['video_chunk_coount']
                self.input_dict['video_chunk_coount'] += 1
                        
                
                time = datetime.datetime.now()
                time_str = datetime.time(time.hour, time.minute, time.second, time.microsecond)
                # log wall_time, bit_rate, buffer_size, rebuffer_time, video_chunk_size, future_download_time, reward
                self.log_file.write(str(time_str) + '\t' +
                                    str(VIDEO_BIT_RATE[post_data['lastquality']]) + '\t' +
                                    str(post_data['buffer']) + '\t' +
                                    str(rebuffer_time / M_IN_K) + '\t' +
                                    str(video_chunk_size) + '\t' +
                                    str(video_chunk_fetch_time) + '\t' +
                                    str(reward) + '\n')
                self.log_file.flush()

                time_relative = (time - time_start).total_seconds()

                logger.info("Retrieving Observation Dictionary")
                logger.info("Input parameters:  last quality (INDEX) {},\n\
                        video_chunk_fetch_time (MS) {},\n\
                        rebuffer_time {} (MS),\n\
                        buffer (MS) {},\n\
                        time elapsed since playback started (S) {}".format( post_data['lastquality'], 
                                                                                                    video_chunk_fetch_time, 
                                                                                                    rebuffer_time, 
                                                                                                    post_data['buffer']*MILLISECONDS_IN_SEC,
                                                                                                    time_relative) )

                observation_dictionary, end_of_video = streaming_env.get_video_chunk(post_data['lastquality'], 
                                                                                    video_chunk_fetch_time, 
                                                                                    rebuffer_time, 
                                                                                    post_data['buffer']*MILLISECONDS_IN_SEC,
                                                                                    time_relative)
                import pprint
                obs_dict_string = pprint.pformat(observation_dictionary)
                logger.debug(obs_dict_string)
                send_data = str(classifier.next_quality(observation_dictionary, reward))

                logger.info("Selected quality {}".format(send_data))
                end_of_video = False
                if ( post_data['lastRequest'] == TOTAL_VIDEO_CHUNKS ):
                    send_data = ""  # send_data = "REFRESH" we don't want the video to restart
                    end_of_video = True
                    self.input_dict['last_total_rebuf'] = 0
                    self.input_dict['last_bit_rate'] = DEFAULT_QUALITY
                    self.input_dict['video_chunk_coount'] = 0
                    self.log_file.write('\n')  # so that in the log we know where video ends
                    print("done_successful") # signal player finished a video
                    sys.stdout.flush()

                self.send_response(200)
                self.send_header('Content-Type', 'text/plain')
                self.send_header('Content-Length', len(send_data))
                self.send_header('Access-Control-Allow-Origin', "*")
                self.end_headers()
                self.wfile.write(send_data.encode('utf-8'))



        def do_GET(self):
            self.send_response(200)
            #self.send_header('Cache-Control', 'Cache-Control: no-cache, no-store, must-revalidate max-age=0')
            self.send_header('Cache-Control', 'max-age=3000')
            self.send_header('Content-Length', 20)
            self.end_headers()
            self.wfile.write("console.log('here');")

        def log_message(self, format, *args):
            return

    return Request_Handler


def run(server_class=HTTPServer, port=8333, log_file_path=LOG_FILE):

    np.random.seed(RANDOM_SEED)

    if not os.path.exists(SUMMARY_DIR):
        os.makedirs(SUMMARY_DIR)

    # make chunk combination options
    for combo in itertools.product([0,1,2,3,4,5], repeat=5):
        CHUNK_COMBO_OPTIONS.append(combo)

    with open(log_file_path, 'w') as log_file:

        s_batch = [np.zeros((S_INFO, S_LEN))]

        last_bit_rate = DEFAULT_QUALITY
        last_total_rebuf = 0
        # need this storage, because observation only contains total rebuffering time
        # we compute the difference to get

        video_chunk_count = 0

        input_dict = {'log_file': log_file,
                      'last_bit_rate': last_bit_rate,
                      'last_total_rebuf': last_total_rebuf,
                      'video_chunk_coount': video_chunk_count,
                      's_batch': s_batch}

        # interface to abr_rl server
        handler_class = make_request_handler(input_dict=input_dict)

        #serve on random port
        while True:
            try:
                server_address = ('localhost', port)
                httpd = server_class(server_address, handler_class)
            except socketserver.socket.error:
                port = random.randint(2000,65535)
            else:
                break
        print('Listening on port ' + str(port))
        sys.stdout.flush()
        httpd.serve_forever()


def main():
    if len(sys.argv) > 6:
        parser = argparse.ArgumentParser()
        parser.add_argument('port', help='port of the server', type=int)
        parser.add_argument('abr', help='abr under test', type=str)
        parser.add_argument('trace_file', help='traces file folder', type=str)
        parser.add_argument('timeout', help='timeout', type=int)
        parser.add_argument("result_dir", help='results directory', type=str)
        parser.add_argument('stream_id', help='stream id', type=int)
        parser.add_argument("--debug", action="store_true", help='If selected, logging also to debug level')
        parser.add_argument("--display", action="store_true", help='If selected, logging also to stderr')

        global args
        args = parser.parse_args()
        
        global TREE_FILENAME
        TREE_FILENAME = os.path.join(TREE_FOLDER, args.abr, 'classifier')
    
        form = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logfile = os.path.join(args.result_dir, "log/{}_abr_server_{}.log".format(args.abr,args.stream_id))
        


        llevel = logging.INFO
        if args.debug:
            llevel = logging.DEBUG

        global logger

        logger = logging.getLogger("{} ABR SERVER {}".format(args.abr, args.stream_id))
        logger.setLevel(llevel)

        fo = logging.FileHandler(logfile, mode = 'a')
        formatter = logging.Formatter(form)
        fo.setLevel(llevel)
        fo.setFormatter(formatter)
        logger.addHandler(fo)


        if args.display:
            so = logging.StreamHandler(sys.stderr)
            so.setLevel(llevel)
            so.setFormatter(formatter)
            logger.addHandler(so)
   

        log_file_p = os.path.join(args.result_dir, 'result')
        if not os.path.exists(log_file_p):
            os.makedirs(log_file_p, 0o777)

        log_file_abr_server = os.path.join(log_file_p, '{}_rewards_{}.log'.format(args.abr,args.stream_id))
        
        logger.info('Running with arguments passed {}'.format(args.trace_file))
        run(port=args.port, log_file_path=log_file_abr_server)
        logger.info('Listening on port ' + str(args.port))
    else:
        run()



if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Keyboard interrupted.")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
