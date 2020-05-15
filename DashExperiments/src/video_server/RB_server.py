#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
from http.server import BaseHTTPRequestHandler, HTTPServer
import socketserver

import base64
import urllib
import requests
import sys
import os
import logging
import json
import random

import argparse
from collections import deque
import numpy as np
import time
import datetime


VIDEO_BIT_RATE = [300,750,1200,1850,2850,4300]  # Kbps
BITRATE_REWARD = [1, 2, 3, 12, 15, 20]
BITRATE_REWARD_MAP = {0: 0, 300: 1, 750: 2, 1200: 3, 1850: 12, 2850: 15, 4300: 20}
M_IN_K = 1000.0
DEFAULT_QUALITY = 0  # default video quality without agent
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> this number of Mbps
SMOOTH_PENALTY = 1
TOTAL_VIDEO_CHUNKS = 48
SUMMARY_DIR = './results'
LOG_FILE = './results/log'
# in format of time_stamp bit_rate buffer_size rebuffer_time video_chunk_size future_download_time reward

def make_request_handler(input_dict):

    class Request_Handler(BaseHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            self.input_dict = input_dict
            self.log_file = input_dict['log_file']
            BaseHTTPRequestHandler.__init__(self, *args, **kwargs)

        def do_POST(self):
            content_length = int(self.headers['Content-Length'])
            post_data = json.loads(self.rfile.read(content_length))
            send_data = ""
            
            if len(post_data) == 1: # message comes from the controller that suggests the optimal bitrate
                self.send_response(200)
                self.send_header('Content-Type', 'text/plain')
                self.send_header('Content-Length', len(send_data))
                self.send_header('Access-Control-Allow-Origin', "*")
                self.end_headers()
                self.wfile.write(send_data.encode('utf-8'))
                return

            if ( 'lastquality' in post_data ):
                start_time = datetime.datetime.now()
                rebuffer_time = float(post_data['RebufferTime'] -self.input_dict['last_total_rebuf'])
                reward = \
                   VIDEO_BIT_RATE[post_data['lastquality']] / M_IN_K \
                   - REBUF_PENALTY * (post_data['RebufferTime'] - self.input_dict['last_total_rebuf']) / M_IN_K \
                   - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[post_data['lastquality']] -
                                                  self.input_dict['last_bit_rate']) / M_IN_K
                # reward = BITRATE_REWARD[post_data['lastquality']] \
                #         - 8 * rebuffer_time / M_IN_K - np.abs(BITRATE_REWARD[post_data['lastquality']] - BITRATE_REWARD_MAP[self.input_dict['last_bit_rate']])

                video_chunk_fetch_time = post_data['lastChunkFinishTime'] - post_data['lastChunkStartTime']
                if video_chunk_fetch_time == 0:
                    video_chunk_fetch_time = 0.9
                video_chunk_size = post_data['lastChunkSize']
                time = datetime.datetime.now()
                time_str = datetime.time(time.hour, time.minute, time.second, time.microsecond)
                # log wall_time, bit_rate, buffer_size, rebuffer_time, video_chunk_size, future_download_time, reward
                self.log_file.write(str(time_str) + '\t' +
                                    str(VIDEO_BIT_RATE[post_data['lastquality']]) + '\t' +
                                    str(post_data['buffer']) + '\t' +
                                    str(float(post_data['RebufferTime'] - self.input_dict['last_total_rebuf']) / M_IN_K) + '\t' +
                                    str(video_chunk_size) + '\t' +
                                    str(video_chunk_fetch_time) + '\t' +
                                    str(reward) + '\n')
                self.log_file.flush()

                self.input_dict['last_total_rebuf'] = post_data['RebufferTime']
                self.input_dict['last_bit_rate'] = VIDEO_BIT_RATE[post_data['lastquality']]

                if ( post_data['lastRequest'] == TOTAL_VIDEO_CHUNKS ):
                    send_data = ""  # send_data = "REFRESH" we don't want the video to restart
                    self.input_dict['last_total_rebuf'] = 0
                    self.input_dict['last_bit_rate'] = DEFAULT_QUALITY
                    self.log_file.write('\n')  # so that in the log we know where video ends
                    print("done_successful") # signal player finished a video
                    sys.stdout.flush()

            self.send_response(200)
            self.send_header('Content-Type', 'text/plain')
            self.send_header('Content-Length', len(send_data))
            self.send_header('Access-Control-Allow-Origin', "*")
            self.end_headers()
            self.wfile.write(send_data.encode('utf-8'))
            end_time = datetime.datetime.now()
            

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

    if not os.path.exists(SUMMARY_DIR):
        os.makedirs(SUMMARY_DIR)
        
    with open(log_file_path, 'w') as log_file:
        last_bit_rate = DEFAULT_QUALITY
        last_total_rebuf = 0 
        input_dict = {'log_file': log_file,
                      'last_bit_rate': last_bit_rate,
                      'last_total_rebuf': last_total_rebuf}

        handler_class = make_request_handler(input_dict=input_dict)
        server_address = ('localhost', port)
        httpd = server_class(server_address, handler_class)
        
        logger.info("Listening on port {}".format(str(port)))
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
        logging.debug("Keyboard interrupted.")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
