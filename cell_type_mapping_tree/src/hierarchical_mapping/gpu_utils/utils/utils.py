from enum import Enum

try:
    TORCH_AVAILABLE = False
    import torch  # type: ignore
    if torch.cuda.is_available():
        TORCH_AVAILABLE = True
        NUM_GPUS = torch.cuda.device_count()
        import torch.distributed as dist
except ImportError:
    TORCH_AVAILABLE = False
    NUM_GPUS = None


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count],
                             dtype=torch.float32,
                             device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        # fmtstr = '{name} \n ' + \
        #     '\tval={val' + self.fmt + '} \n' + \
        #     '\tavg={avg' + self.fmt + '} \n' + \
        #     '\tsum={sum' + self.fmt + '} \n' + \
        #     '\tcount={count' + self.fmt + '} \n'

        # fmtstr = f'{self.name} \
        # [{self.val}{self.fmt} \
        # ({self.avg}{self.fmt}) \
        # {self.sum}{self.fmt}]'

        fmtstr = '{name} [{val' + self.fmt + '} ({avg' + self.fmt + '}) {sum' + self.fmt + '}] '

        # fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('  '.join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def get_timers():

    timers = {}
    # timers["type_assignment"] = AverageMeter('type_assignment', ':6.3f')
    # timers["loop"] = AverageMeter('loop', ':6.3f')
    # timers["results"] = AverageMeter('Results', ':6.3f')
    # timers["run_type_assignment"] = AverageMeter('run_type_assignment', ':6.3f')
    # timers["assemble"] = AverageMeter('assemble', ':6.3f')
    # timers["choose_node"] = AverageMeter('choose_node', ':6.3f')
    # timers["tally_votes"] = AverageMeter('tally votes', ':6.3f')
    # timers["choose_node_p2"] = AverageMeter('cnp2', ':6.3f')
    # timers["togpu"] = AverageMeter('togpu', ':6.3f')
    # timers["tocpu"] = AverageMeter('tocpu', ':6.3f')
    # timers["sn1"] = AverageMeter('sn1', ':6.3f')
    # timers["sn2"] = AverageMeter('sn2', ':6.3f')
    # timers["matmul"] = AverageMeter('matmul', ':6.3f')
    # timers["meansqrt"] = AverageMeter('meansqrt', ':6.3f')
    # timers["looppreproc"] = AverageMeter('looppreproc', ':6.3f')
    # timers["correlation_nearest_neighbors"] = AverageMeter('correlation_nearest_neighbors', ':6.3f')
    # timers["tally_loop"] = AverageMeter('tally_loop', ':6.3f')
    # timers["argmax"] = AverageMeter('argmax', ':6.3f')
    # timers["correlationarraynge"] = AverageMeter('correlationarraynge', ':6.3f')
    # timers["votes_counter"] = AverageMeter('votes_counter', ':6.3f')
    # timers["dele"] = AverageMeter('dele', ':6.3f')
    # timers["correlation_dot"] = AverageMeter('correlation_dot', ':6.3f')
    return timers
