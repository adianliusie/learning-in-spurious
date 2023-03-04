import os
import logging

from collections import defaultdict
from src.handlers.trainer import Trainer
from src.handlers.evaluater import Evaluator
from src.data.handler import DataHandler
from src.utils.general import save_json, load_json
from src.utils.parser import get_model_parser, get_train_parser

# Load logger
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.DEBUG)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    # get model and train args
    model_parser = get_model_parser()
    model_parser.add_argument('--num-seeds', type=int, default=3, help='number of seeds to have for this experiment')

    train_parser = get_train_parser()

    # Parse system input arguments
    model_args, moargs = model_parser.parse_known_args()
    train_args, toargs = train_parser.parse_known_args()

    # Making sure no unkown arguments are given
    assert set(moargs).isdisjoint(toargs), f"{set(moargs) & set(toargs)}"

    logger.info(model_args.__dict__)
    logger.info(train_args.__dict__)
    
    performances = defaultdict(dict)

    # Load any runs that have already been saved in past runs
    if os.path.isfile(os.path.join(model_args.path, 'curve.json')):
        performance_cache = load_json(os.path.join(model_args.path, 'curve.json'))
        for lim in performance_cache.keys():
            performances[int(lim)] = performance_cache[lim]

    for lim in [0, 40, 100, 400, 1_000, 4_000, 10_000, 40_000, 100_000]:
        # check whether enough data samples, and exit if so
        train_data = DataHandler.load_split(train_args.dataset, mode='train', bias=train_args.bias, lim=lim)

        if len(train_data) < lim-2:
            continue

        for seed_num in range(1, model_args.num_seeds+1):
            # skip runs already done in previous submissions
            if str(seed_num) in performances[lim]:
                continue

            #== Training ==========================================================================#
            # reset random seed as gets overwritten by other runs
            setattr(model_args, 'rand_seed', None)
            
            # artifically set the data truncation limit
            setattr(train_args, 'lim', lim)
            
            # create model
            exp_path = os.path.join(model_args.path, f'{lim}/seed-{seed_num}')
            trainer = Trainer(exp_path, model_args)
            
            # train the model
            trainer.train(train_args)
            
            #== Evaluation ========================================================================#
            seed_perf = {}

            # use small subsets of eval data, if dataset with huge evaluation test set given
            dataset = train_args.dataset
            dataset = f"{dataset}-s"  if dataset in ['imdb', 'amazon', 'yelp'] else dataset
            
            # get bias_detail
            bias_name, bias_acc, *bias_bounds = train_args.bias.split('-')
            bias_bounds = '-'.join(bias_bounds)

            # standard evaluation
            evaluator = Evaluator(exp_path, train_args.device)
            preds  = evaluator.load_preds(dataset, 'test')
            labels = evaluator.load_labels(dataset, 'test')
            acc = evaluator.calc_acc(preds, labels)
            seed_perf['acc'] = acc

            # biased evlauation (points where the shortcut property is correct)
            bias_preds  = evaluator.load_preds(dataset, 'test', f'all-{bias_name}-{bias_bounds}')
            bias_labels = evaluator.load_labels(dataset, 'test', f'all-{bias_name}-{bias_bounds}')
            bias_acc = evaluator.calc_acc(bias_preds, bias_labels)
            seed_perf['bias_acc'] = bias_acc

            # evaluation of unbiased points (points where the shortcut property is incorrect)
            inv_bias_preds  = evaluator.load_preds(dataset, 'test', f'inv-{bias_name}-{bias_bounds}')
            inv_bias_labels = evaluator.load_labels(dataset, 'test', f'inv-{bias_name}-{bias_bounds}')
            inv_bias_acc = evaluator.calc_acc(inv_bias_preds, inv_bias_labels)
            seed_perf['inv_bias_acc'] = inv_bias_acc

            # append results to overall performance curve
            print(f"LIM {lim} SEED {seed_num}:", seed_perf)
            performances[lim][seed_num] = seed_perf
            
            # delete model weights to not use up all disk space so quickly
            os.remove(
                os.path.join(exp_path, 'models/model.pt')
            )
            
            # save the updated performance
            save_json(
                performances, os.path.join(model_args.path, 'curve.json')
            )


######### python curve.py --bias lexical-0.7 --path curves --ood mnli
