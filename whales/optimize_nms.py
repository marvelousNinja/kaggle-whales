# mask_threshold=0.5, min_area=0, pre_nms_threshold=0.1, pre_nms=500, post_nms_threshold=0.15, post_nms_topk=300,
#     matrix_nms_iter=4

# 20211211_235800_inp480x640_grid30x40_b6_limit100_nfocal
# out.txt

# python3 -m sartorius.fit --name inp480x640_grid30x40_b6_limit100_nfocal --grid-shape 30x40 --input-shape 480x640 --batch-size 6 --validation-batch-size 1 --mask-limit 100 --checkpoint-path data/experiments/20211211_235800_inp480x640_grid30x40_b6_limit100_nfocal/best.pth.serialized --steps-per-epoch 0

# define an objective function
def objective(args):
    case, val = args
    if case == 'case 1':
        return val
    else:
        return val ** 2

# define a search space
from hyperopt import hp
from hyperopt.pyll.base import scope
import hyperopt.pyll
import subprocess

space = {
    'matrix_nms_iter': scope.int(hp.randint('matrix_nms_iter', 15)),
    'min_area': scope.int(hp.randint('min_area', 300)),

    'pre_nms_threshold': hp.uniform('pre_nms_threshold', 0.0, 1.0),
    'post_nms_threshold_blend': hp.uniform('post_nms_threshold_blend', 0.0, 1.0), # TODO: pre larger than post

    'pre_nms': scope.int(hp.randint('pre_nms', 1000)),
    'post_nms_topk_blend': hp.uniform('post_nms_topk_blend', 0.0, 1.0)
}

def objective(args):
    def blend(start, finish, alpha):
        return start + (finish - start) * alpha

    post_nms_threshold = blend(args['pre_nms_threshold'], 1.0, args['post_nms_threshold_blend'])
    post_nms_topk = int(blend(args['pre_nms'], 0, args['post_nms_threshold_blend']))

    cmd = f'''python3 -m sartorius.fit --name inp480x640_grid30x40_b6_limit100_nfocal \
        --grid-shape 120x160 \
        --input-shape 480x640 \
        --batch-size 6 \
        --validation-batch-size 2 \
        --mask-limit 100 \
        --checkpoint-path data/experiments/20211227_223342_weighted_no0.5/best.pth.serialized \
        --steps-per-epoch 0 \
        --num_epochs 1 \
        --matrix_nms_iter {args['matrix_nms_iter']} \
        --min_area {args['min_area']} \
        --pre_nms_threshold {args['pre_nms_threshold']} \
        --post_nms_threshold {post_nms_threshold} \
        --pre_nms {args['pre_nms']} \
        --post_nms_topk {post_nms_topk}
    '''

    subprocess.check_call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    with open('out.txt', 'r') as f:
        out = float(f.read())

    print(out, args)

    return -out


print(hyperopt.pyll.stochastic.sample(space))

from hyperopt import fmin, tpe
best = fmin(objective, space, algo=tpe.suggest, max_evals=500)

import pdb; pdb.set_trace()
pass

# print best
# -> {'a': 1, 'c2': 0.01420615366247227}
# print hyperopt.space_eval(space, best)
# -> ('case 2', 0.01420615366247227}