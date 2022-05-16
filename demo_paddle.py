from argparse import ArgumentParser

import utils

from models.basic_model import CDEvaluator

import os

"""
quick start

sample files in ./samples

save prediction files in the ./samples/predict

"""


def get_args():
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--project_name', default='', type=str)
    parser.add_argument('--gpu_ids', type=str, default='-1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--checkpoint_root', default='ChangeFormer', type=str)
    parser.add_argument('--output_folder', default='samples-paddle/predict', type=str)

    # data
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--dataset', default='CDDataset', type=str)
    parser.add_argument('--data_name', default='off-data/test', type=str)

    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--split', default="demo", type=str)
    parser.add_argument('--img_size', default=256, type=int)

    # model
    parser.add_argument('--n_class', default=2, type=int)
    parser.add_argument('--embed_dim', default=256, type=int)
    parser.add_argument('--net_G', default='base_transformer_pos_s4_dd8', type=str,
                        help='ChangeFormerV6 | CD_SiamUnet_diff | SiamUnet_conc | Unet | DTCDSCN | base_resnet18 | base_transformer_pos_s4_dd8 | base_transformer_pos_s4_dd8_dedim8|')
    parser.add_argument('--checkpoint_name', default='best_ckpt.pdparams', type=str)

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = get_args()
    utils.get_device(args)
    # device = torch.device("cuda:%s" % args.gpu_ids[0]
    #                       if torch.cuda.is_available() and len(args.gpu_ids)>0
    #                     else "cpu")
    args.checkpoint_dir = os.path.join(args.checkpoint_root, args.project_name)
    os.makedirs(args.output_folder, exist_ok=True)

    log_path = os.path.join(args.output_folder, 'log_vis.txt')

    data_loader = utils.get_loader(args.data_name, img_size=args.img_size,
                                   batch_size=args.batch_size,
                                   split=args.split, is_train=False)

    model = CDEvaluator(args)
    model.load_checkpoint(args.checkpoint_name)
    model.eval()

    for i, batch in enumerate(data_loader):
        name = batch['name']
        print('process: %s' % name)
        score_map = model._forward_pass(batch)
        model._save_predictions()






