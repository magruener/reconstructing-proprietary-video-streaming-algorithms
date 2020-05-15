import subprocess, argparse, signal, os, sys, json, threading, multiprocessing, datetime, shutil, traceback
import psutil
import logging
from time import sleep
from tempfile import SpooledTemporaryFile as tempfile
from multiprocessing.connection import Client

# SCRIPTS: not parameters
#

BROWSER_SCRIPT = './src/run_browser_test.py'
HTTP_TRAFFIC_SCRIPT =  './src/http/http_traffic.py'
HTTP_SERVER_SCRIPT = './src/http/web-traffic-generator/http_traffic_server.py'
RELEASE_SCRIPT = './src/utils/release.sh' 
UDP_PROXY_SERVER_SCRIPT = './src/udp_video_transfer/proxy_server.py'
UDP_VIDEO_TRANSFER_SCRIPT = './src/udp_video_transfer/video_server.py'


#
#

RANDOM_SEED = 42
COOLDOWN_TIME = 15
controller_connection = None


# PORTS OFFSET
# STATIC

PROXY_SERVER_TCP_PORT_OFFSET = 8000 # Proxy servers are allocated TCP ports in the range 8000-8099
ABR_SERVER_TCP_PORT_OFFSET = 6000 # Proxy servers are allocated TCP ports in the range 6000-6099
PROXY_SERVER_UDP_PORT_OFFSET = 6100 # Proxy servers are allocated UDP ports in the range 6100-6199
VIDEO_SERVER_UDP_PORT_OFFSET = 6200 # Video servers are allocated UDP ports in the range 6200-6299


#
#

available_bw = multiprocessing.Value('d', 0) # variable shared between processes
ports = []
streams = {} # for each stream it contains data, such as number of chunks downloaded or buffer occupancy
proxy_server_procs = {}
video_server_procs = {}


def format_cmd(cmd_list):
    return [' '.join(cmd_list)]

# timeout if running for too long (t + 30s)
def timeout_handler(signum, frame):
    raise Exception("Timeout")


# end all subprocesses
def end_processes():
    try:
        logger.info("Trying to terminate http traffic generator")
        http_traffic_proc.terminate()
    except:
        logger.info("Termination usnsuccessful")
    
    try:
        logger.info("Trying to terminate http traffic client")
        http_traffic_client_proc.terminate()
    except:
        logger.info("Termination usnsuccessful")

    try:
        logger.info("Trying to terminate proxy servers")
        proxy_server_procs.terminate()
    except:
        logger.info("Termination usnsuccessful")
    
    try:
        logger.info("Trying to terminate video server procs")
        video_server_procs.terminate()
    except:
        logger.info("Termination usnsuccessful")
    
    


# start 'run_browser_test' with the necessary information for the logfile
def start_ABR(test_id, name, trace, abr, stream_id, duration, server_address, udp, quic, server_module):
    script = ['python', BROWSER_SCRIPT, abr, duration, server_address, args.result_dir, str(stream_id), trace, server_module]
    if args.debug:
        script.append("--debug")
    if args.display:
        script.append("--display")
    # print(' '.join(str(e) for e in script))
    # script = ['python', RUN_SCRIPT, abr, duration, server_address, './testresults/'+test_id+'/log_'+test_id+'_'+str(run_nr)+'_'+name]
    if udp:
        script.append('-u')
    if quic == 'true':
        script.append('-q')
    if args.browser:
        script.append('-b')

    cmd = ''
    for x in script:
        cmd += x + ' ' 
    # proc = subprocess.call(script, shell=False) # synchronous
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)  # asynchronous
    output = proc.stdout.readline()
    proc.wait()


#   read json config of test-setup and run it _repeat_n times

#def runTest(testcase):
#    udp = True if testcase['udp'] == 'true' else False
#
#    # number of repetitions
#    n = int(testcase['repeat_n'])
#    
#    logger.info("Repeating the experiment {} times".format(n))
#
#    # repeat this test_setup n times
#    
#    for i in range(n):
#        
#        logger.info("Repetition number {}".format(i+1))
#
#        # prepare jobs to be run
#        job_threads = []
#        j = 0
#        for job in testcase['jobs']:
#            server = server_config['server'][job['transport']]['address'] + ':' + \
#                     server_config['server'][job['transport']]['port']
#
#            t = threading.Timer(interval=float(job['start']), function=start_ABR,
#                                args=[testcase['test_id'], job['name'], testcase['trace'], job['abr'], j,
#                                      job['duration'], server, udp, job['quic'], job['server_module']])
#            job_threads.append(t)
#            j += 1
#
#            logger.info("Job number {} started".format(j))
#            logger.info("Server configurations: {}".format(server))
#            logger.info("Testcase id: {}, Job name: {}, Trace: {}, \
#                        ABR: {}, Duration:{}, UDP: {}, QUIC: {}, \
#                        Server Module :{}".format(testcase['test_id'], job['name'],\
#                        testcase['trace'], job['abr'], job['duration'], udp, job['quic'], job['server_module']))
#
#
#
#        abr = job['abr']
#        
#        num_servers = len(testcase['jobs'])
#        
#        if udp:
#            cwnd_folder = os.path.join(args.result_dir,'cwnd')
#            start_udp_infrastructure(num_servers, cwnd_folder)
#            logger.info("UDP infrastructure started")
#
#        global p_c, ports
#        manager = multiprocessing.Manager()
#        ports = manager.list()
#        
#        
#        p_c = multiprocessing.Process(target=run_traces, args=(abr, ports, udp, num_servers))
#        p_c.start()
#
#        # start jobs
#        for t in job_threads:
#            t.start()
#            logger.info("Job started")
#
#        for t in job_threads:
#            t.join()
#
#            # stop network shaping script
#        sleep(COOLDOWN_TIME)
#


# Modified runTest to include etherogeneous experiments

def runTest(testcase):

    # number of repetitions
    n = int(testcase['repeat_n'])
    
    logger.info("Repeating the experiment {} times".format(n))

    # repeat this test_setup n times
    
    for i in range(n):
        
        logger.info("Repetition number {}".format(i+1))

        # prepare jobs to be run
        job_threads = []
        j = 0
        progressive_udp = 0
        progressive_tcp = 0

        for job in testcase['jobs']:
            
            server = server_config['server'][job['transport']]['address'] + ':' + \
                     server_config['server'][job['transport']]['port']
            

            udp = True if job['udp'] == 'true' else False
            if udp:
                logger.info("Job {} has udp enabled".format(j))
            else:
                logger.info("Job {} doesn't have udp enabled".format(j))




            t = threading.Timer(interval=float(job['start']), function=start_ABR,
                                args=[testcase['test_id'], job['name'], testcase['trace'], job['abr'], j,
                                      job['duration'], server, udp, job['quic'], job['server_module']])
            
            job_threads.append(t)
            j += 1

            logger.info("Job number {} started".format(j))
            logger.info("Server configurations: {}".format(server))
            logger.info("Testcase id: {}, Job name: {}, Trace: {}, \
                        ABR: {}, Duration:{}, UDP: {}, QUIC: {}, \
                        Server Module :{}".format(testcase['test_id'], job['name'],\
                        testcase['trace'], job['abr'], job['duration'], udp, job['quic'], job['server_module']))


            if udp:
                progressive_udp += 1
                cwnd_folder = os.path.join(args.result_dir,'cwnd')
                start_udp_infrastructure(job_id, progressive_udp, cwnd_folder)
                logger.info("UDP infrastructure started for job {}".format(j))
            else:
                progressive_tcp += 1


        #abr = job['abr']
        
        #num_servers = len(testcase['jobs'])
        

        global p_c, ports
        manager = multiprocessing.Manager() # unsure it returns all the ports
        ports = manager.list()
        
        
        p_c = multiprocessing.Process(target=run_traces, args=(progressive_udp, progressive_tcp))
        p_c.start()

        # start jobs
        for t in job_threads:
            t.start()
            logger.info("Job started")

        for t in job_threads:
            t.join()

            # stop network shaping script
        sleep(COOLDOWN_TIME)



# starting udp for a particular job

def start_udp_infrastructure(job_id, progressive_udp, cwnd_folder):
        proxy_server_tcp_port = PROXY_SERVER_TCP_PORT_OFFSET + progressive_udp
        proxy_server_udp_port = PROXY_SERVER_UDP_PORT_OFFSET + progressive_udp
        video_server_udp_port = VIDEO_SERVER_UDP_PORT_OFFSET + progressive_udp

        logger.debug("proxy server tcp port: {}, proxy server udp port: {}, video server upd port: {}".format(proxy_server_tcp_port, proxy_server_udp_port, video_server_udp_port))
    
        proxy_server_script = ['python3', UDP_PROXY_SERVER_SCRIPT, \
                               str(proxy_server_tcp_port), str(proxy_server_udp_port), str(video_server_udp_port), args.result_dir, str(job_id)]
        if args.debug:
            proxy_server_script.append("--debug")
        if args.display:
            proxy_server_script.append("--display")
 
        
        proxy_server_proc = subprocess.Popen(proxy_server_script, shell=False)
        proxy_server_procs.append(proxy_server_proc)
        
        logger.info("Proxy server {} started".format(job_id))

        video_server_script = ['python2', UDP_VIDEO_TRANSFER_SCRIPT, \
                               str(video_server_udp_port), str(proxy_server_udp_port), str(job_id), args.result_dir]
        if args.debug:
            video_server_script.append("--debug")
        if args.display:
            video_server_script.append("--display")
 

        
        video_server_proc = subprocess.Popen(video_server_script, shell=False)
        video_server_procs.append(video_server_proc)

        logger.info("UDP video transfer {} started".format(job_id))


# For each stream over UDP we create one proxy server and one video server
#def start_udp_infrastructure(num_servers, cwnd_folder):
#    for i in range(num_servers):
#        proxy_server_tcp_port = PROXY_SERVER_TCP_PORT_OFFSET + i
#        proxy_server_udp_port = PROXY_SERVER_UDP_PORT_OFFSET + i
#        video_server_udp_port = VIDEO_SERVER_UDP_PORT_OFFSET + i
#
#        logger.debug("proxy server tcp port: {}, proxy server udp port: {}, video server upd port: {}".format(proxy_server_tcp_port, proxy_server_udp_port, video_server_udp_port))
#    
#        proxy_server_script = ['python3', UDP_PROXY_SERVER_SCRIPT, \
#                               str(proxy_server_tcp_port), str(proxy_server_udp_port), str(video_server_udp_port), args.result_dir, str(i)]
#        if args.debug:
#            proxy_server_script.append("--debug")
#        if args.display:
#            proxy_server_script.append("--display")
# 
#        
#        proxy_server_proc = subprocess.Popen(proxy_server_script, shell=False)
#        proxy_server_procs[i] = proxy_server_proc
#        
#        logger.info("Proxy server {} started".format(i))
#
#        video_server_script = ['python2', UDP_VIDEO_TRANSFER_SCRIPT, \
#                               str(video_server_udp_port), str(proxy_server_udp_port), str(i), args.result_dir]
#        if args.debug:
#            video_server_script.append("--debug")
#        if args.display:
#            video_server_script.append("--display")
# 
#
#        
#        video_server_proc = subprocess.Popen(video_server_script, shell=False)
#        video_server_procs[i] = video_server_proc
#
#        logger.info("UDP video transfer {} started".format(i))
#
#
def parse_trace():
    traces = [[],[]]
    with open(os.path.join(args.traces, testcase_data['trace'])) as fin:
        lines = fin.readlines()
        for line in lines:
            line = line.strip() 
            if bool(line):
                traces[0].append(float(line.split()[0]))
                traces[1].append(float(line.split()[1]))
    return traces


#def run_traces(ports, udp, abr, num_servers):
#    global time, available_bw
#
#    traces = parse_trace()
#    traces_length = len(traces[0])
#    i = 0
#    available_bw.value = traces[1][0]
#
#    os.system('sudo /sbin/tc qdisc del dev lo root')  # Make sure that all qdiscs are reset
#    if udp:
#        # If we run the video stream over UDP, we only want to throttle the UDP port. The reason being that the implementation
#        # requires the proxy server to have an arbitrarily quick TCP connection to the video player.
#        # TBF requires you to specify 'burst'. For 10mbit/s on Intel, you need at least 10kbyte buffer if you want to reach your configured rate!
#        os.system('sudo /sbin/tc qdisc add dev lo root handle 1: prio')
#        os.system('sudo /sbin/tc qdisc add dev lo parent 1:3 handle 30: tbf rate ' + str(
#            available_bw.value) + 'mbit latency 2000ms burst 20000')# peakrate ' + str(peakrate) +'mbit mtu 1024')
#
#        for j in range(num_servers):
#            os.system('sudo /sbin/tc filter add dev lo protocol ip parent 1:0 prio 3 u32 match ip sport ' + str(
#                VIDEO_SERVER_UDP_PORT_OFFSET + j) + ' 0xffff flowid 1:3')
#        os.system('sudo /sbin/tc filter add dev lo protocol ip parent 1:0 prio 3 u32 match ip sport ' + str(
#            5600) + ' 0xffff flowid 1:3')
#
#
#    else:
#        logger.info('Throttling bandwidth: ' + str(available_bw.value) + ' Mbps')
#        os.system('sudo ifconfig lo mtu 1500')  # set the mtu to the size of an ethernet frame to avoid jumbo frames
#        os.system('sudo /sbin/tc qdisc add dev lo root tbf rate ' + str(
#            available_bw.value) + 'mbit latency 200000ms burst 200000')
#
#    while True:
#        time = traces[0][i]
#        available_bw.value = traces[1][i]
#
#        time = datetime.datetime.now()
#        time_str = datetime.time(time.hour, time.minute, time.second, time.microsecond)
#        
#        
#        if udp:
#            os.system('sudo /sbin/tc qdisc change dev lo parent 1:3 handle 30: tbf rate ' + str(
#                available_bw.value) + 'mbit latency 2000ms burst 20000')
#        else:
#            os.system('sudo /sbin/tc qdisc change dev lo root tbf rate ' + str(
#                available_bw.value) + 'mbit latency 2000ms burst 20000')
#
#        logger.info("Throtteling started")
#
#        try:
#            logger.info("Sleeping for {}".format(traces[0][i + 1] - traces[0][i]))
#            sleep(traces[0][i + 1] - traces[0][i])
#            i = i + 1
#        except IndexError:
#            logger.info("Starting the trace again")
#            logger.info("Sleeping for {}".format(traces[0][0]))
#            sleep(traces[0][0])
#            i = 0
#
#        except KeyboardInterrupt:
#            logger.info("Throttling interrupted")
#            break
#



# to fix
def run_traces(progressive_udp, progressive_tcp):
    global time, available_bw

    traces = parse_trace()
    traces_length = len(traces[0])
    i = 0
    available_bw.value = traces[1][0]

    os.system('sudo /sbin/tc qdisc del dev lo root')  # Make sure that all qdiscs are reset
    
        # If we run the video stream over UDP, we only want to throttle the UDP port. The reason being that the implementation
        # requires the proxy server to have an arbitrarily quick TCP connection to the video player.
        # TBF requires you to specify 'burst'. For 10mbit/s on Intel, you need at least 10kbyte buffer if you want to reach your configured rate!
#    os.system('sudo /sbin/tc qdisc add dev lo root handle 1: prio')
#    os.system('sudo /sbin/tc qdisc add dev lo parent 1:3 handle 30: tbf rate ' + str(
#            available_bw.value) + 'mbit latency 2000ms burst 20000')# peakrate ' + str(peakrate) +'mbit mtu 1024')

#    for j in range(progressive_udp):
#            logger.info("Throttling on port {}".format(VIDEO_SERVER_UDP_PORT_OFFSET + j + 1))
#            os.system('sudo /sbin/tc filter add dev lo protocol ip parent 1:0 prio 3 u32 match ip sport ' + str(VIDEO_SERVER_UDP_PORT_OFFSET + j + 1) + ' 0xffff flowid 1:3')
#    
#    for j in range(progressive_tcp):
#            logger.info("Throttling on port {}".format(ABR_SERVER_TCP_PORT_OFFSET + j))
#            os.system('sudo /sbin/tc filter add dev lo protocol ip parent 1:0 prio 3 u32 match ip dport ' + str(ABR_SERVER_TCP_PORT_OFFSET + j) + ' 0xffff flowid 1:3')
#    
#    
#    os.system('sudo /sbin/tc filter add dev lo protocol ip parent 1:0 prio 3 u32 match ip sport ' + str(
#            5600) + ' 0xffff flowid 1:3')

    os.system('sudo ifconfig lo mtu 1500')  # set the mtu to the size of an ethernet frame to avoid jumbo frames
    os.system('sudo /sbin/tc qdisc add dev lo root tbf rate 1mbit latency 200000ms burst 200000')
    
    
    while True:
        time = traces[0][i]
        available_bw.value = traces[1][i]

        time = datetime.datetime.now()
        time_str = datetime.time(time.hour, time.minute, time.second, time.microsecond)
        
        if (available_bw.value <= 0):
            available_bw.value = 0.1

        os.system('sudo /sbin/tc qdisc change dev lo root tbf rate ' + str(available_bw.value) + 'mbit latency 2000ms burst 20000')
#

        
#        os.system('sudo /sbin/tc qdisc change dev lo parent 1:3 handle 30: tbf rate ' + str(
#                available_bw.value) + 'mbit latency 2000ms burst 20000')

        logger.debug("Throtteling started")

        try:
            logger.debug("Sleeping for {}".format(traces[0][i + 1] - traces[0][i]))
            sleep(traces[0][i + 1] - traces[0][i])
            i = i + 1
        except IndexError:
            logger.info("Starting the trace again")
            logger.debug("Sleeping for {}".format(traces[0][0]))
            sleep(traces[0][0])
            i = 0

        except KeyboardInterrupt:
            logger.info("Throttling interrupted")
            break

# This function must be called before the program terminates. Otherwise, the bandwidth will remain throttled.
def successful_termination():
    logger.info("Terminating throttling")
    end_process(p_c)
    
    logger.info('Bandwidth is no longer being throttled.')
    os.system('sudo ifconfig lo mtu 9000')
    os.system('sudo /sbin/tc qdisc del dev lo root')

def end_process(process):
 #   for proc in process.children(recursive=True):
 #       proc.send_signal(signal.SIGINT)
    process.kill()
    process.join()

def init():
    # parse server config
    #
    
    logger.info("loading server config json")
    global server_config
    with open(args.server_config) as json_data:
        server_config = json.load(json_data)
    logger.info("loading server config json complete")
    logger.debug("loaded server data: " + str(server_config))

    # parsing testcase data
    #

    global testcase_data
    logger.info('loading testcase json config')
    with open(args.testcase) as json_data:
            testcase_data = json.load(json_data)
    
    logger.info('loading testcase json config completed')
    logger.debug("loaded server data: " + str(testcase_data))
    
    logger.info('creating results folders')
    os.mkdir(os.path.join(args.result_dir,'traces'), 0o777)
    os.mkdir(os.path.join(args.result_dir,'tcpdump'), 0o777)
    os.mkdir(os.path.join(args.result_dir,'setup'), 0o777)
    os.mkdir(os.path.join(args.result_dir,'cwnd'), 0o777)
    
    logger.info('copying testcase information into setup folder')

    with open(os.path.join(args.result_dir,'setup', testcase_data['test_id'] + '.json'), 'w') as outfile:
            json.dump(testcase_data, outfile)

    if args.http_traffic:
        # generate background http traffic
        http_filename = os.path.join(args.result_dir, 'http')
        
        # starting http server
        global http_traffic_proc
        
        http_traffic_script = ['python', HTTP_SERVER_SCRIPT]
        http_traffic_proc = subprocess.Popen(format_cmd(http_traffic_script), shell=False)
        logger.info("HTTP server script started")
        
        #starting HTTP client
        global http_traffic_client_proc
        
        http_traffic_client_script = ['python', HTTP_TRAFFIC_SCRIPT , str(http_filename)]
        http_traffic_client_proc = subprocess.Popen(format_cmd(http_traffic_client_script), shell=False)
        logger.info("HTTP client started")

# define main function to capture interrupts
def main():
    
    
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('server_config', help='json file containing server information', type=str)
    parser.add_argument('testcase', help='testcase filename', type=str)
    parser.add_argument('traces', help='traces file folder', type=str)
    parser.add_argument('result_dir', help='result directory', type=str)

    parser.add_argument("--http_traffic", action="store_true", help='parameter to add background http traffic')
    parser.add_argument("--browser", action="store_true", help='parameter to not open browser')
    parser.add_argument("--debug", action="store_true", help='If selected, logging also to debug level')
    parser.add_argument("--display", action="store_true", help='If selected, logging also to stderr')
    
    # args need to be global for simplicity
    global args
    args = parser.parse_args()
   

    # creating results dir, if doesn't already exist
    # otherwise cleaning it
    
    if not os.path.exists(args.result_dir):
        os.makedirs(os.path.join(args.result_dir, 'log/'), 0o777)
    else:
        shutil.rmtree(args.result_dir)
        os.makedirs(os.path.join(args.result_dir, 'log/'),  0o777)
   
    # log file configuration
    # logging into the result directory
    # if debug is enabled, logging appears also in stdout

    form = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logfile = os.path.join(args.result_dir, "log/competitive_test.log")
    
    llevel = logging.INFO
    if args.debug:
        llevel = logging.DEBUG
    
    global logger

    logger = logging.getLogger("DASH ABR TEST")
    logger.setLevel(llevel)

    fo = logging.FileHandler(logfile, mode = 'a')
    formatter = logging.Formatter(form)
    fo.setLevel(llevel)
    fo.setFormatter(formatter)
    logger.addHandler(fo)

    if args.display is not None:
        so = logging.StreamHandler(sys.stderr)
        so.setLevel(llevel)
        so.setFormatter(formatter)
        logger.addHandler(so)
   

    # cleanup before the next experiment
    #
    logger.info("Deallocating all necessary ports")
    os.system(RELEASE_SCRIPT) # to deallocate all necessary ports
    logger.info("Deallocation successfull")

    # initialize new experiment
    #
    logger.info("Initializing experiment")
    init()
    logger.info("Initialization successfull")
    
    # run tests
    #
    logger.info("Start testcase {}".format(args.testcase))
    runTest(testcase_data)
    logger.info('Completed testcase {}'.format(args.testcase))


if __name__ == "__main__":
    try:
        main()
        successful_termination()
        end_processes()
        sys.exit(0)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupted.")
        try:
            successful_termination()
            end_processes()
            sys.exit(0)
        except SystemExit:
            successful_termination()
            end_processes()
            os._exit(0)
    except Exception as e:
            logger.info("Something went wrong.. trying to terminate the processes anyway")
            logger.info("Exception: {}".format(e))
            successful_termination()
            end_processes()
            sys.exit(-1)
        
