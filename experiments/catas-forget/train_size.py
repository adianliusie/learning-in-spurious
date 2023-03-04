import os
import argparse
import logging
import numpy as np

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
    train_parser.add_argument('--ood', default=['rt', 'imdb-s', 'yelp-s'], type=str, nargs = '+', help='OOD dataset to evaluate models on')

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

    for lim in [0, 4, 10, 40, 100, 400, 1000, 4_000, 10_000, 40_000, 100_000]:
        # check whether enough data samples, and exit if so
        train_data = DataHandler.load_split(train_args.dataset, mode='train', bias='balanced', lim=lim)
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

            evaluator = Evaluator(exp_path, train_args.device)
            evaluator.setup_helpers()

            # IID evaluation
            preds  = evaluator.load_preds(train_args.dataset, 'test')
            labels = evaluator.load_labels(train_args.dataset, 'test')
            acc = evaluator.calc_acc(preds, labels)
            seed_perf[train_args.dataset] = acc

            # run evaluation on all OOD domains
            ood_acc = None
            ood_domains = getattr(train_args, 'ood')
            if ood_domains:
                for domain in ood_domains:
                    ood_preds  = evaluator.load_preds(domain, 'test')
                    ood_labels = evaluator.load_labels(domain, 'test')
                    
                    if domain == 'hans':
                        ood_preds = {idx: (pred == 0) for idx, pred in ood_preds.items()}
                        
                    ood_acc = evaluator.calc_acc(ood_preds, ood_labels)
                    seed_perf[domain] = ood_acc
            
            # run synonym inference if prompting
            if model_args.prompt_finetuning:
                for neg_word, pos_word in [('horrible', 'fantastic'), ('terrible','great'), ('poor', 'amazing')]:
                    evaluator.model.update_label_words([neg_word, pos_word])
                    # generate new probs for different label words
                    probs = evaluator.generate_probs(train_args.dataset, 'test')
                    preds = {}
                    for ex_id, probs in probs.items():
                        preds[ex_id] = int(np.argmax(probs, axis=-1))

                    labels = evaluator.load_labels(train_args.dataset, 'test')
                    acc = evaluator.calc_acc(preds, labels)
                    print(f'{neg_word} + {pos_word} acc: {acc:.2f}' )
                    seed_perf[f'{neg_word}_{pos_word}'] = acc

                #update back to original words (just in case)
                evaluator.model.update_label_words(model_args.label_words)

            # append results to overall performance curve
            performances[lim][seed_num] = seed_perf
            print(f"LIM {lim} SEED {seed_num}:", seed_perf)

            # delete model weights to not use up all disk space so quickly
            os.remove(
                os.path.join(exp_path, 'models/model.pt')
            )
            
            # save the updated performance
            save_json(
                performances, os.path.join(model_args.path, 'curve.json')
            )


######### python curve.py --bias lexical-0.7 --path curves --ood mnli
