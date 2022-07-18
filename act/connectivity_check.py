import torch
import logging
import sys
import torch.distributed as dist
import os


def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    if get_mpi_size() == 1:
        return tensor
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(get_mpi_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

def init_logging():
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)
    logger_fmt = logging.Formatter('%(asctime)s.%(msecs)03d %(process)d:%(filename)s:%(lineno)s %(funcName)10s(): %(message)s')
    ch.setFormatter(logger_fmt)

    root = logging.getLogger()
    root.handlers = []
    root.addHandler(ch)
    root.setLevel(logging.INFO)

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

def destroy_process_group():
    if dist.is_initialized():
        dist.destroy_process_group()


def get_mpi_local_rank():
    if 'LOCAL_RANK' in os.environ:
        return int(os.environ['LOCAL_RANK'])
    return int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', '0'))
def get_mpi_size():
    if 'WORLD_SIZE' in os.environ:
        return int(os.environ['WORLD_SIZE'])
    return int(os.environ.get('OMPI_COMM_WORLD_SIZE', '1'))


def get_mpi_rank():
    if 'RANK' in os.environ:
        return int(os.environ['RANK'])
    return int(os.environ.get('OMPI_COMM_WORLD_RANK', '0'))

def get_master_node_ip():
    if 'AZ_BATCHAI_JOB_MASTER_NODE_IP' in os.environ:
        return os.environ['AZ_BATCHAI_JOB_MASTER_NODE_IP']
    elif 'MASTER_IP' in os.environ:
        return os.environ['MASTER_IP']
    else:
        return 'localhost'


def ensure_init_process_group(
        device_id=None, port=12345, backend='nccl'):
    if not dist.is_initialized():
        dist_url = 'tcp://{}:{}'.format(get_master_node_ip(),
                port)
        from datetime import timedelta
        init_param = {
            'backend': backend,
            'init_method': dist_url,
            'rank': get_mpi_rank(),
            'world_size': get_mpi_size(),
            'timeout': timedelta(days=10),
        }
        if device_id is None:
            device_id = get_mpi_local_rank()
        torch.cuda.set_device(device_id)
        logging.info(init_param)
        dist.init_process_group(**init_param)
        logging.info('initialized')


def test_nccl_gpu_all_gather(port):
    ensure_init_process_group(port=port, backend='nccl')
    for i in range(10):
        s = (i + 1) * 5
        x = torch.zeros((s, s), device='cuda')
        x = concat_all_gather(x)
        dist.barrier()
        dist.broadcast(x, 0)
        dist.barrier()
        dist.all_reduce(x)
        dist.barrier()
        x.sum().item()
    logging.info('concat_all_gather on nccl/cuda is OK. begin destropy')
    destroy_process_group()
    logging.info('process group destroyed')

def main(port):
    test_nccl_gpu_all_gather(port)

if __name__ == '__main__':
    init_logging()
    main(sys.argv[1])

