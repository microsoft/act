import yaml
import base64
import re
import logging
import os.path as op
import os
import shutil
import subprocess as sp
import psutil
from pprint import pformat
import sys
from collections import OrderedDict


def write_to_yaml_file(context, file_name):
    ensure_directory(op.dirname(file_name))
    with open(file_name, 'w') as fp:
        yaml.dump(context, fp, default_flow_style=False,
                encoding='utf-8', allow_unicode=True)

def url_to_file_by_curl(url, fname, bytes_start=None, bytes_end=None):
    ensure_directory(op.dirname(fname))
    if bytes_start == 0 and bytes_end == 0:
        cmd_run(['touch', fname])
        return
    if bytes_start is None:
        bytes_start = 0
    elif bytes_start < 0:
        size = get_url_fsize(url)
        bytes_start = size + bytes_start
        if bytes_start < 0:
            bytes_start = 0
    if bytes_end is None:
        # -f: if it fails, no output will be sent to output file
        if bytes_start == 0:
            cmd_run(['curl', '-f',
                url, '--output', fname])
        else:
            cmd_run(['curl', '-f', '-r', '{}-'.format(bytes_start),
                url, '--output', fname])
    else:
        # curl: end is inclusive
        cmd_run(['curl', '-f', '-r', '{}-{}'.format(bytes_start, bytes_end - 1),
            url, '--output', fname])

def limited_retry_agent(num, func, *args, **kwargs):
    for i in range(num):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.info('fails with \n{}: tried {}-th time'.format(
                e,
                i + 1))
            import time
            print_trace()
            if i == num - 1:
                raise
            time.sleep(5)

def ensure_remove_dir(d):
    is_dir = op.isdir(d)
    is_link = op.islink(d)
    if is_dir:
        if not is_link:
            shutil.rmtree(d)
        else:
            os.unlink(d)

def list_to_dict(l, idx, keep_one=False):
    result = OrderedDict()
    for x in l:
        if x[idx] not in result:
            result[x[idx]] = []
        y = x[:idx] + x[idx + 1:]
        if not keep_one and len(y) == 1:
            y = y[0]
        result[x[idx]].append(y)
    return result

def print_job_infos(all_job_info):
    all_key = [
        'cluster',
        'status',
        'appID-s',
        'result',
        'elapsedTime',
        'elapsedFinished',
        'mem_used',
        'gpu_util',
        'speed',
        'left']
    keys = ['data', 'net', 'expid']
    meta_keys = ['num_gpu']
    all_key.extend(keys)
    all_key.extend(meta_keys)

    # find the keys whose values are the same
    def all_equal(x):
        assert len(x) > 0
        return all(y == x[0] for y in x[1:])

    if len(all_job_info) > 1:
        equal_keys = [k for k in all_key if all_equal([j.get(k) for j in all_job_info])]
        if len(equal_keys) > 0:
            logging.info('equal key values for all jobs')
            print_table(all_job_info[0:1], all_key=equal_keys)
        all_key = [k for k in all_key if not all_equal([j.get(k) for j in all_job_info])]

    print_table(all_job_info, all_key=all_key)

def copy_file(src, dest):
    tmp = dest + '.tmp'
    # we use rsync because it could output the progress
    cmd_run('rsync {} {} --progress'.format(src, tmp).split(' '))
    os.rename(tmp, dest)

def natural_key(text):
    import re
    result = []
    for c in re.split(r'([0-9]+(?:[.][0-9]*)?)', text):
        try:
            result.append(float(c))
        except:
            continue
    return result

def natural_sort(strs, key=None):
    if key is None:
        strs.sort(key=natural_key)
    else:
        strs.sort(key=lambda s: natural_key(key(s)))

def compile_by_docker(src_zip, docker_image, dest_zip):
    # compile the qd zip file and generate another one by compiling. so that
    # there is no need to compile it again.
    src_fname = op.basename(src_zip)
    src_folder = op.dirname(src_zip)

    docker_src_folder = '/tmpwork'
    docker_src_zip = op.join(docker_src_folder, src_fname)
    docker_out_src_fname = src_fname + '.out.zip'
    docker_out_zip = op.join(docker_src_folder, docker_out_src_fname)
    out_zip = op.join(src_folder, docker_out_src_fname)
    docker_compile_folder = '/tmpcompile'
    cmd = ['docker', 'run',
            '-v', '{}:{}'.format(src_folder, docker_src_folder),
            docker_image,
            ]
    cmd.append('/bin/bash')
    cmd.append('-c')
    compile_cmd = [
            'mkdir -p {}'.format(docker_compile_folder),
            'cd {}'.format(docker_compile_folder),
            'unzip {}'.format(docker_src_zip),
            'bash compile.aml.sh',
            'zip -yrv x.zip *',
            'cp x.zip {}'.format(docker_out_zip),
            'chmod a+rw {}'.format(docker_out_zip),
            ]
    cmd.append(' && '.join(compile_cmd))
    cmd_run(cmd)
    ensure_directory(op.dirname(dest_zip))
    copy_file(out_zip, dest_zip)

def zip_qd(out_zip, options=None):
    ensure_directory(op.dirname(out_zip))
    cmd = [
        'zip',
        '-uyrv',
        out_zip,
        '*',
    ]
    if options:
        cmd.extend(options)
    else:
        cmd.extend([
            '-x',
            '\*src/CCSCaffe/\*',
            '-x',
            '\*src/build/lib.linux-x86_64-2.7/\*',
            '-x',
            '\*build/lib.linux-x86_64-2.7/\*',
            '-x',
            '\*build/temp.linux-x86_64-2.7/\*',
            '-x',
            '\*build/lib.linux-x86_64-3.5/\*',
            '-x',
            '\*build/temp.linux-x86_64-3.5/\*',
            '-x',
            '\*build/lib.linux-x86_64-3.7/\*',
            '-x',
            'assets\*',
            '-x',
            '\*build/temp.linux-x86_64-3.7/\*',
            '-x',
            '\*build/lib.linux-x86_64-3.6/\*',
            '-x',
            '\*build/temp.linux-x86_64-3.6/\*',
            '-x',
            '\*src/detectron2/datasets/\*',
            '-x',
            '\*src/CCSCaffe/models/\*',
            '-x',
            '\*src/CCSCaffe/data/\*',
            '-x',
            '\*src/CCSCaffe/examples/\*',
            '-x',
            '\*src/detectron2/output\*',
            '-x',
            'aux_data/yolo9k/\*',
            '-x',
            'visualization\*',
            '-x',
            'output\*',
            '-x',
            'data\*',
            '-x',
            '\*.build_release\*',
            '-x',
            '\*.build_debug\*',
            '-x',
            '\*.build\*',
            '-x',
            '\*tmp_run\*',
            '-x',
            '\*src/CCSCaffe/MSVC/\*',
            '-x',
            '\*.pyc',
            '-x',
            '\*.so',
            '-x',
            '\*.o',
            '-x',
            '\*src/CCSCaffe/docs/tutorial/\*',
            '-x',
            '\*src/CCSCaffe/matlab/\*',
            '-x',
            '\*.git\*',
            '-x',
            '\*wandb\*',
        ])
    cmd_run(cmd, working_dir=os.getcwd(), shell=True)

def get_user_name():
    import getpass
    return getpass.getuser()

def init_logging():
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)
    logger_fmt = logging.Formatter('%(asctime)s.%(msecs)03d %(process)d:%(filename)s:%(lineno)s %(funcName)10s(): %(message)s')
    ch.setFormatter(logger_fmt)

    root = logging.getLogger()
    root.handlers = []
    root.addHandler(ch)
    root.setLevel(logging.INFO)

def dict_has_path(d, p, with_type=False):
    ps = p.split('$')
    cur_dict = d
    while True:
        if len(ps) > 0:
            k = dict_parse_key(ps[0], with_type)
            if isinstance(cur_dict, dict) and k in cur_dict:
                cur_dict = cur_dict[k]
                ps = ps[1:]
            elif isinstance(cur_dict, list):
                try:
                    k = int(k)
                except:
                    return False
                cur_dict = cur_dict[k]
                ps = ps[1:]
            else:
                return False
        else:
            return True

def dict_update_nested_dict(a, b, overwrite=True):
    for k, v in b.items():
        if k not in a:
            dict_update_path_value(a, k, v)
        else:
            if isinstance(dict_get_path_value(a, k), dict) and isinstance(v, dict):
                dict_update_nested_dict(dict_get_path_value(a, k), v, overwrite)
            else:
                if overwrite:
                    dict_update_path_value(a, k, v)

def parse_iteration(file_name):
    patterns = [
        '.*model(?:_iter)?_([0-9]*)\..*',
        '.*model(?:_iter)?_([0-9]*)e\..*',
        '.*model(?:_iter)?_([0-9]*)$',
    ]
    for p in patterns:
        r = re.match(p, file_name)
        if r is not None:
            return int(float(r.groups()[0]))
    logging.info('unable to parse the iterations for {}'.format(file_name))
    return -2

def write_to_file(contxt, file_name, append=False):
    p = os.path.dirname(file_name)
    ensure_directory(p)
    if type(contxt) is str:
        contxt = contxt.encode()
    flag = 'wb'
    if append:
        flag = 'ab'
    with open(file_name, flag) as fp:
        fp.write(contxt)

def releaseLock(locked_file_descriptor):
    ''' release exclusive lock file access '''
    locked_file_descriptor.close()

def acquireLock(lock_f='/tmp/lockfile.LOCK'):
    ''' acquire exclusive lock file access '''
    import fcntl
    locked_file_descriptor = open(lock_f, 'w+')
    fcntl.lockf(locked_file_descriptor, fcntl.LOCK_EX)
    return locked_file_descriptor

def get_table_print_lines(a_to_bs, all_key):
    if len(a_to_bs) == 0:
        logging.info('no rows')
        return []
    if not all_key:
        all_key = []
        for a_to_b in a_to_bs:
            all_key.extend(a_to_b.keys())
        all_key = sorted(list(set(all_key)))
    all_width = [max([len(str(a_to_b.get(k, ''))) for a_to_b in a_to_bs] +
        [len(k)]) for k in all_key]
    row_format = ' '.join(['{{:{}}}'.format(w) for w in all_width])

    all_line = []
    line = row_format.format(*all_key)
    all_line.append(line.strip())
    for a_to_b in a_to_bs:
        line = row_format.format(*[str(a_to_b.get(k, '')) for k in all_key])
        all_line.append(line)
    return all_line

def print_table(a_to_bs, all_key=None, latex=False, **kwargs):
    if len(a_to_bs) == 0:
        return
    if not latex:
        all_line = get_table_print_lines(a_to_bs, all_key)
        logging.info('\n{}'.format('\n'.join(all_line)))
    else:
        from qd.latex_writer import print_simple_latex_table
        if all_key is None:
            all_key = list(set(a for a_to_b in a_to_bs for a in a_to_b))
            all_key = sorted(all_key)
        x = print_simple_latex_table(a_to_bs,
                all_key, **kwargs)
        logging.info('\n{}'.format(x))
        return x

def hash_sha1(s):
    import hashlib
    if type(s) is not str:
        s = pformat(s)
    return hashlib.sha1(s.encode('utf-8')).hexdigest()

def has_handle(fpath, opened_files=None):
    fpath = op.abspath(op.realpath(fpath))
    if opened_files is None:
        for proc in psutil.process_iter():
            try:
                for item in proc.open_files():
                    if fpath == item.path:
                        return True
            except Exception:
                pass
        return False
    else:
        return fpath in opened_files

def ensure_remove_file(d):
    if op.isfile(d) or op.islink(d):
        try:
            os.remove(d)
        except:
            pass

def query_all_opened_file_in_system():
    fs = []
    for proc in psutil.process_iter():
        for proc in psutil.process_iter():
            try:
                for item in proc.open_files():
                    fs.append(item.path)
            except Exception:
                pass
    return list(set(fs))

def dict_update_path_value(d, p, v):
    ps = p.split('$')
    while True:
        if len(ps) == 1:
            d[ps[0]] = v
            break
        else:
            if ps[0] not in d:
                d[ps[0]] = {}
            d = d[ps[0]]
            ps = ps[1:]

def get_all_path(d, with_type=False, leaf_only=True, with_list=True):
    assert not with_type, 'will not support'
    all_path = []

    if isinstance(d, dict):
        for k, v in d.items():
            all_sub_path = get_all_path(
                v, with_type, leaf_only=leaf_only, with_list=with_list)
            all_path.extend([k + '$' + p for p in all_sub_path])
            if not leaf_only or len(all_sub_path) == 0:
                all_path.append(k)
    elif (isinstance(d, tuple) or isinstance(d, list)) and with_list:
        for i, _v in enumerate(d):
            all_sub_path = get_all_path(
                _v, with_type,
                leaf_only=leaf_only,
                with_list=with_list,
            )
            all_path.extend(['{}$'.format(i) + p for p in all_sub_path])
            if not leaf_only or len(all_sub_path) == 0:
                all_path.append('{}'.format(i))
    return all_path

def load_from_yaml_file(file_name):
    with open(file_name, 'r') as fp:
        data = load_from_yaml_str(fp)
    while isinstance(data, dict) and '_base_' in data:
        b = op.join(op.dirname(file_name), data['_base_'])
        result = load_from_yaml_file(b)
        assert isinstance(result, dict)
        del data['_base_']
        all_key = get_all_path(data, with_list=False)
        for k in all_key:
            v = dict_get_path_value(data, k)
            dict_update_path_value(result, k, v)
        data = result
    return data

def load_from_yaml_str(s):
    return yaml.load(s, Loader=yaml.UnsafeLoader)

def decode_general_cmd(extraParam):
    re_result = re.match('.*python (?:scripts|src)/.*\.py -bp (.*)', extraParam)
    if re_result and len(re_result.groups()) == 1:
        ps = load_from_yaml_str(base64.b64decode(re_result.groups()[0]))
        return ps

def read_to_buffer(file_name):
    with open(file_name, 'rb') as fp:
        all_line = fp.read()
    return all_line

def decode_to_str(x):
    try:
        return x.decode('utf-8')
    except UnicodeDecodeError:
        return x.decode('latin-1')

def cmd_run(list_cmd,
            return_output=False,
            env=None,
            working_dir=None,
            stdin=sp.PIPE,
            shell=False,
            dry_run=False,
            silent=False,
            process_input=None,
            stdout=None,
            stderr=None,
            ):
    if not silent:
        logging.info('start to cmd run: {}'.format(' '.join(map(str, list_cmd))))
        if working_dir:
            logging.info(working_dir)
    # if we dont' set stdin as sp.PIPE, it will complain the stdin is not a tty
    # device. Maybe, the reson is it is inside another process.
    # if stdout=sp.PIPE, it will not print the result in the screen
    e = os.environ.copy()
    if 'SSH_AUTH_SOCK' in e:
        del e['SSH_AUTH_SOCK']
    if working_dir:
        ensure_directory(working_dir)
    if env:
        for k in env:
            e[k] = env[k]
    if dry_run:
        # we need the log result. Thus, we do not return at teh very beginning
        return
    if not return_output:
        #if env is None:
            #p = sp.Popen(list_cmd, stdin=sp.PIPE, cwd=working_dir)
        #else:
        p = sp.Popen(' '.join(list_cmd) if shell else list_cmd,
                     stdin=stdin,
                     env=e,
                     shell=shell,
                     stdout=stdout,
                     cwd=working_dir,
                     stderr=stderr,
                     )
        message = p.communicate(input=process_input)
        if p.returncode != 0:
            raise ValueError(message)
        return message
    else:
        if shell:
            message = sp.check_output(' '.join(list_cmd),
                    env=e,
                    cwd=working_dir,
                    shell=True)
        else:
            message = sp.check_output(list_cmd,
                                      env=e,
                                      cwd=working_dir,
                                      )
        if not silent:
            logging.info('finished the cmd run')
        return decode_to_str(message)

def get_url_fsize(url):
    result = cmd_run(['curl', '-sI', url], return_output=True)
    for row in result.split('\n'):
        ss = [s.strip() for s in row.split(':')]
        if len(ss) == 2 and ss[0] == 'Content-Length':
            size_in_bytes = int(ss[1])
            return size_in_bytes

def get_file_size(f):
    return os.stat(f).st_size

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
def try_delete(f):
    os.remove(f)

def ensure_directory(path):
    if path == '' or path == '.':
        return
    if path != None and len(path) > 0:
        assert not op.isfile(path), '{} is a file'.format(path)
        if not os.path.exists(path) and not op.islink(path):
            try:
                os.makedirs(path)
            except:
                if os.path.isdir(path):
                    # another process has done makedir
                    pass
                else:
                    raise

def concat_files(ins, out):
    ensure_directory(op.dirname(out))
    out_tmp = out + '.tmp'
    with open(out_tmp, 'wb') as fp_out:
        for i, f in enumerate(ins):
            logging.info('concating {}/{} - {}'.format(i, len(ins), f))
            with open(f, 'rb') as fp_in:
                shutil.copyfileobj(fp_in, fp_out, 1024*1024*10)
    os.rename(out_tmp, out)

def dict_parse_key(k, with_type):
    if with_type:
        if k[0] == 'i':
            return int(k[1:])
        else:
            return k[1:]
    return k

def dict_get_path_value(d, p, with_type=False):
    ps = p.split('$')
    cur_dict = d
    while True:
        if len(ps) > 0:
            k = dict_parse_key(ps[0], with_type)
            if isinstance(cur_dict, (tuple, list)):
                cur_dict = cur_dict[int(k)]
            else:
                cur_dict = cur_dict[k]
            ps = ps[1:]
        else:
            return cur_dict


