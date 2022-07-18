#!/usr/bin/python
import os
import sys
import logging
import subprocess as sp
import os.path as op
import yaml


def load_list_file(fname):
    with open(fname, 'r') as fp:
        lines = fp.readlines()
    result = [line.strip() for line in lines]
    if len(result) > 0 and result[-1] == '':
        result = result[:-1]
    return result

def init_logging():
    import socket
    if get_mpi_rank() == 0:
        level = logging.INFO
    else:
        level = logging.DEBUG
    logging.basicConfig(level=level,
        format='%(asctime)s.%(msecs)03d {} %(process)d %(filename)s:%(lineno)s %(funcName)10s(): %(message)s'.format(
            socket.gethostname()),
        datefmt='%m-%d %H:%M:%S',
    )

def copy_file(src, dest):
    tmp = dest + '.tmp'
    # we use rsync because it could output the progress
    cmd_run('rsync {} {} --progress'.format(src, tmp).split(' '))
    os.rename(tmp, dest)

def unzip(zip_file, target_folder):
    local_zip = '/tmp/code.zip'
    if zip_file.startswith('http'):
        cmd_run(['rm', '-rf', local_zip])
        cmd_run(['wget', zip_file, '-O', local_zip])
    else:
        copy_file(zip_file, local_zip)
    cmd_run(['unzip', local_zip, '-d', target_folder])

def cmd_run(cmd, working_directory='./', succeed=False,
        return_output=False, stdout=None, stderr=None):
    e = os.environ.copy()
    e['PYTHONPATH'] = '/app/caffe/python:{}'.format(e.get('PYTHONPATH', ''))
    # in the maskrcnn, it will download the init model to TORCH_HOME. By
    # default, it is /root/.torch, which is different among diferent nodes.
    # However, teh implementation assumes that folder is a share folder. Thus
    # only rank 0 do the data downloading. Here, we assume the output folder is
    # shared, which is the case in AML.
    e['TORCH_HOME'] = './output/torch_home'
    ensure_directory(e['TORCH_HOME'])
    logging.info('start to cmd run: {}'.format(' '.join(map(str, cmd))))
    for c in cmd:
        logging.info(c)
    if not return_output:
        try:
            p = sp.Popen(
                cmd, stdin=sp.PIPE,
                cwd=working_directory,
                env=e,
                stdout=stdout,
                stderr=stderr,
            )
            p.communicate()
            if succeed:
                logging.info('return code = {}'.format(p.returncode))
                assert p.returncode == 0
        except:
            if succeed:
                logging.info('raising exception')
                raise
    else:
        return sp.check_output(cmd)

def ensure_directory(path):
    if path == '' or path == '.':
        return
    if path != None and len(path) > 0:
        try:
            if not os.path.exists(path) and not op.islink(path):
                os.makedirs(path)
        except:
            pass

def root_user():
    import getpass
    user = getpass.getuser()
    logging.info('user = {}'.format(user))
    return user == 'root'

def user_path_sudo(cmd):
    path = os.environ['PATH']
    assert isinstance(cmd, list)
    cmd = ['env', 'PATH={}'.format(path)] + cmd
    if not root_user():
        cmd = ['sudo', '-E'] + cmd
    return cmd

def compile_qd(folder, compile_args):
    path = os.environ['PATH']

    cmd = ['pip', 'install', '-r', 'requirements.txt']
    cmd = user_path_sudo(cmd)
    cmd_run(cmd,
        working_directory=folder,
        succeed=False)

    compile_file = 'compile.aml.sh'
    cmd_run(['chmod', '+x', op.join(folder, compile_file)])
    cmd = ['env', 'PATH={}'.format(path), './{}'.format(compile_file)]
    if compile_args:
        cmd.append(compile_args)
    if not root_user():
        cmd.insert(0, 'sudo')
    cmd_run(cmd,
            working_directory=folder, succeed=False)

def get_mpi_rank():
    return int(os.environ.get('OMPI_COMM_WORLD_RANK', '0'))

def get_mpi_local_rank():
    return int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', '0'))

import fcntl

def acquireLock():
    ''' acquire exclusive lock file access '''
    locked_file_descriptor = open('/tmp/lockfile.LOCK', 'w+')
    fcntl.lockf(locked_file_descriptor, fcntl.LOCK_EX)
    return locked_file_descriptor

def releaseLock(locked_file_descriptor):
    ''' release exclusive lock file access '''
    locked_file_descriptor.close()


def parse_gpu_usage_dict(result):
    import re
    used = []
    p = '^\|.* ([0-9]*)MiB \/ *([0-9]*)MiB *\| *([0-9]*)\%.*Default \|$'
    for line in result.split('\n'):
        line = line.strip()
        r = re.match(p, line)
        if r != None:
            u = [int(g) for g in r.groups()]
            names = ['mem_used', 'mem_total', 'gpu_util']
            used.append({n: v for n, v in zip(names, u)})
    return used

def monitor():
    while True:
        cmd_result = cmd_run(['nvidia-smi'], return_output=True).decode()
        gpu_result = parse_gpu_usage_dict(cmd_result)
        logging.info('{}'.format(gpu_result))
        import shutil
        disk = shutil.disk_usage('/')
        info = []
        info.append('Total = {:.1f}G'.format(disk.total / 1024. ** 3))
        info.append('Free = {:.1f}G'.format(disk.free / 1024. ** 3))
        info.append('Used = {:.1f}G'.format(disk.used / 1024. ** 3))
        logging.info('disk usage of /: {}'.format(', '.join(info)))
        import time
        #cmd_run([sys.executable, 'connectivity_check.py', '13245'])
        time.sleep(60 * 30) # every 30 minutes

class MonitoringProcess(object):
    def __enter__(self):
        if get_mpi_local_rank() == 0:
            self.p = launch_monitoring_process()

    def __exit__ (self, type, value, tb):
        if get_mpi_local_rank() == 0:
            terminate_monitoring_process(self.p)

def terminate_monitoring_process(p):
    p.terminate()
    p.join()

def launch_monitoring_process():
    from multiprocessing import Process
    p = Process(target=monitor)
    p.daemon = True
    p.start()
    return p

def wrap_all(code_zip, code_root,
             folder_link, command,
             sleep_if_fail,
             compile_args,
             sleep_if_succeed,
             ):
    cmd_run([sys.executable, 'connectivity_check.py', '13245'])
    cmd_run(['ibstatus'])
    cmd_run(['grep', 'Port', '/etc/ssh/sshd_config'])
    cmd_run(['nvidia-smi'])
    cmd_run(['ifconfig'])
    cmd_run(['df', '-h'])
    cmd_run(['ls', '/dev'])

    lock_fd = acquireLock()
    logging.info('got the lock')
    # start the ssh server
    if not op.isdir(code_root):
        ensure_directory(code_root)
        # set up the code, models, output under qd
        logging.info('unzipping {}'.format(code_zip))
        unzip(code_zip, code_root)

        for target, source in folder_link.items():
            cmd_run(['rm', '-rf', target], code_root)
            cmd_run(['ln', '-s',
                     source,
                     op.join(code_root, target)])

        #cmd_run(['chmod', 'a+rw',
            #output_folder], succeed=False)

        #cmd_run(['ln', '-s', output_folder, op.join(code_root, 'output')])

        # compile the source code
        compile_qd(code_root, compile_args)
    releaseLock(lock_fd)

    cmd_run(['ls', '-llh'], code_root)
    # after the code is compiled, let's check the lib version
    cmd_run(['pip', 'freeze'])
    logging.info(command)
    if type(command) is str:
        command = list(command.split(' '))

    with MonitoringProcess():
        if len(command) > 0:
            try:
                cmd_run(command, working_directory=code_root,
                        succeed=True)
            except:
                if sleep_if_fail:
                    cmd_run(['sleep', 'infinity'])
                else:
                    raise
            if sleep_if_succeed:
                cmd_run(['sleep', 'infinity'])

def get_mpi_local_size():
    return int(os.environ.get('OMPI_COMM_WORLD_LOCAL_SIZE', '1'))

def load_from_yaml_str(s):
    return yaml.load(s)

def parse_args_to_dict():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--code_path', type=str)
    # in aml, tehre is no chance to provide option like 'store_true' or so.
    # Each option has to be key value pair
    parser.add_argument('--sleep_if_fail', type=int, default=0)
    parser.add_argument('--sleep_if_succeed', type=int, default=0)
    parser.add_argument('--compile_args', type=str, default='')
    # you can add multiple --xxx_folder, where xxx will be the folder name in
    # aml. Before, we hard-coded it to be data or model or output. Now, we can
    # use any name
    parser.add_argument('--command', type=str)
    args, unknown = parser.parse_known_args()
    param = vars(args)
    assert (len(unknown) % 2) == 0, unknown
    assert 'folder_link' not in param
    param['folder_link'] = {}
    for i in range(len(unknown) // 2):
        key = unknown[2 * i]
        value = unknown[2 * i + 1]
        while key.startswith('-'):
            key = key[1:]
        if not key.endswith('_folder'):
            logging.info('ignore {}={} because key not ends with _folder'.format(
                key,
                value))
            continue
        assert key not in param['folder_link'], (param, key)
        param['folder_link'][key[:-len('_folder')]] = value
    return param

def get_host_ip():
    import socket
    host_name = socket.gethostname()
    host_ip = socket.gethostbyname(host_name)
    return host_ip

def print_trace():
    import traceback
    traceback.print_exc()

def try_once(func):
    def func_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.info('ignore error \n{}'.format(str(e)))
            print_trace()
    return func_wrapper

@try_once
def is_bad_node():
    ip = get_host_ip()
    bad_list_fname = 'bad_node_ip.txt'
    if op.isfile(bad_list_fname):
        bad_node_ips = load_list_file(bad_list_fname)
        if ip in bad_node_ips:
            return True
        return False
    else:
        return False

def run():
    if is_bad_node():
        logging.info('this is a bad node and we will occupy it without running')
        cmd_run(['sleep', 'infinity'])
        return
    from pprint import pformat
    logging.info(pformat(sys.argv))
    dict_param = parse_args_to_dict()
    logging.info(pformat(dict_param))

    logging.info('start')

    for k in os.environ:
        logging.info('{} = {}'.format(k, os.environ[k]))

    qd_root = op.join('/tmp', 'code', 'act')

    wrap_all(dict_param['code_path'],
             qd_root,
             dict_param['folder_link'],
             dict_param['command'],
             dict_param['sleep_if_fail'],
             dict_param['compile_args'],
             sleep_if_succeed=dict_param['sleep_if_succeed'],
             )

if __name__ == '__main__':
    init_logging()
    #ensure_ssh_server_running()
    run()

