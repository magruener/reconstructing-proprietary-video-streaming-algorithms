#   runABR
#   script to do a single run of an ABR
#   by starting the localhost-server and a browser-session
#   arguments:
#   @abr_alg, @time, @server_address, @experiment_id

import os, sys, time, selenium, subprocess, signal, argparse, re, httplib, urllib, requests, psutil, traceback
from selenium import webdriver
from pyvirtualdisplay import Display
from selenium.webdriver.chrome.options import Options
from time import sleep
import logging

ABR_SERVER_PORT_OFFSET = 6000

# timeout if running for too long (t + 30s)
def timeout_handler(signum, frame):
	raise Exception("Timeout")
	
# end all subprocesses
def end_process(proc_pid):
    process = psutil.Process(proc_pid)
    for proc in process.children(recursive=True):
        proc.send_signal(signal.SIGINT)
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=10)
    except psutil.TimeoutExpired:
        print("timeout expired closing network script")

def wait_for_video_end(pipe_out, timeout):
    endtime = time.time() + timeout
    
    logger.info('Waiting for video to end')
    while time.time()<endtime:
        line = pipe_out.readline()
        if str.startswith(line, "done_successful"):
            return
    return

#main program
def run():
    #read input variables
    ABR_ALG = args.abr_alg #abr algorithm to execute
    TIME =  args.time_seconds# time to sleep ins seconds
    SERVER_ADDR = args.server_addr #server address to open
    STREAM_ID = str(args.stream_id)
    TRACE = args.trace
    EXP_ID = args.result_dir+'/log_' + ABR_ALG + '_' + TRACE + '_' + STREAM_ID #path to logsile

    #print >> sys.stderr, 'udp', args.udp
    if args.udp:
        url='http://localhost/' + 'myindex_' + ABR_ALG + '_udp.html'
    else:
        url='http://localhost/' + 'myindex_' + ABR_ALG + '.html'
        
    # timeout signal
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(TIME + 30)

    try:
        port = ABR_SERVER_PORT_OFFSET + args.stream_id
        

        # Note: all the video servers have to take the same params
        #
        #
        
        log_file_dir_abr_server = os.path.join(args.result_dir, 'result')
        if not os.path.exists(log_file_dir_abr_server):
            os.makedirs(log_file_dir_abr_server,0o777)
        
        python_v = 'python3'
        command = [python_v, args.server_module, str(port), ABR_ALG, EXP_ID, str(TIME), args.result_dir, STREAM_ID]

        if args.debug:
            command.append('--debug')
        if args.display:
            command.append('--display')

        global proc

        cmd = ''
        for x in command:
            cmd += x + ' '

        logger.info("Starting the server located at {}".format(command[1]))
        proc = subprocess.Popen([cmd], stdout=subprocess.PIPE, shell=True)
        
        sleep(10)

        url += '?p=' + str(port)
        
        print(port) # This has to be the only print statement up to this point. This is because every time we call print, 
                    # its string is passed to competitive_tests.py using pipes
        sys.stdout.flush()
        

        #r = requests.post('http://localhost:' + str(port), json={'suggested_bitrate': 4300})

        # to not display the page in browser (unless -b option)
        if args.show_browser:
            logger.info("Not displaying the browser")
            display = Display(visible=0, size=(300,400))
            display.start()

        #init chrome driver
	'''
        default_chrome_user_dir = 'abr_browser_dir/chrome_data_dir'
        chrome_user_dir = '/tmp/chrome_user_dir_id_'
        os.system('rm -r ' + chrome_user_dir)
        os.system('cp -r ' + default_chrome_user_dir + ' ' + chrome_user_dir)
        chrome_driver = 'abr_browser_dir/chromedriver'
	'''

        options = Options()
        options.add_argument('--ignore-certificate-errors')
        options.add_argument('--autoplay-policy=no-user-gesture-required')
        options.add_argument("--disable-infobars")
        options.add_argument('--disable-application-cache')
        options.add_argument('--media-cache-size=1')
        options.add_argument("--disk-cache-size=1")
        options.add_argument("--disable-web-security") # only needed when running tests over the UDP proxy
        options.add_argument("--explicitly-allowed-ports=6000")
        options.add_argument("--auto-open-devtools-for-tabs");
        
        logger.info("Options have been added to chrome driver")

        #enable quic
        if args.quic:
            logger.info("Enabling quic")
            options.add_argument('--no-proxy-server')
            options.add_argument('--enable-quic')
            options.add_argument('--quic-version=QUIC_VERSION_39')
            options.add_argument('--quic-host-whitelist="https://'+SERVER_ADDR+'" "https://'+SERVER_ADDR+'"')
            options.add_argument('--origin-to-force-quic-on='+SERVER_ADDR)


        # start chrome
        #driver=webdriver.Chrome(chrome_driver, chrome_options=options)
	driver_path = './src/chromedriver'
        driver=webdriver.Chrome(chrome_options=options, executable_path=driver_path)
        driver.set_page_load_timeout(30)
        driver.get(url)

        logger.info("Chrome driver started")        

        #run for @TIME seconds
        wait_for_video_end(pipe_out=proc.stdout, timeout=TIME)
        logger.info("Video ended")
        driver.quit()
        logger.info("Driver quitted")
        if args.show_browser:
            logger.info("Stopping display")
            display.stop()

        logger.info("Sending SIGINT to the video server")
        proc.kill()
        proc.wait()
        


    except Exception as e:
        logging.error(traceback.format_exc())
        try:             
            display.stop()
        except:
            logging.error(traceback.format_exc())
        try:
            driver.quit()
        except:
            logging.error(traceback.format_exc())
        try:
            proc.kill()
            proc.wait()
        except:
            logging.error(traceback.format_exc())

# define main function to capture interrupts
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('abr_alg', help='short name of abr algorithm', type=str)
    parser.add_argument('time_seconds', help='time in seconds', type=int)
    parser.add_argument('server_addr', help='server address, example localhost:2015', type=str)
    parser.add_argument('result_dir', help='results directory', type=str)
    parser.add_argument('-v','--video', help='name of the video to test', type=str, default='testVideo')
    parser.add_argument('stream_id', help='id of stream in case multiple are running in parallel', type=int)
    parser.add_argument('trace', help='name of the trace file used', type=str)
    parser.add_argument('server_module', help='path to the python server module to be called', type=str)
    parser.add_argument('-u', '--udp', help='use UDP connection', action='store_true')
    parser.add_argument('-q', '--quic', help='enable quic', action='store_true')
    parser.add_argument('-b', '--show_browser', help='show browser window', action='store_true')
    parser.add_argument("--debug", action="store_true", help='If selected, logging also to debug level')
    parser.add_argument("--display", action="store_true", help='If selected, logging also to stderr')
    
    global args
    args = parser.parse_args()
    

    form = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logfile = os.path.join(args.result_dir, "log/browser_test_" + str(args.stream_id) + ".log")
    
    llevel = logging.INFO
    if args.debug:
        llevel = logging.DEBUG

    global logger

    logger = logging.getLogger("BROWSER EXPERIMENT STREAM ID {}".format(args.stream_id))
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
   

    run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Keyboard interrupted.")
        try:
            proc.send_signal(signal.SIGINT)
            proc.wait()
        except:
            pass
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
