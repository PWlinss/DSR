import logging
import torch
from os import path as osp
import os
from copy import deepcopy
from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.utils import get_env_info, get_root_logger, get_time_str, make_exp_dirs
from basicsr.utils.options import dict2str, parse_options
import torch.nn.functional as F


def convert(root_path):
    # parse options, set distributed setting, set ramdom seed
    opt, _ = parse_options(root_path, is_train=False)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # create model
    modelrestrity = build_model(opt)
    model = deepcopy(modelrestrity)
    print("===================>> start convert <<===================")
    # for module in model.modules():

    # model convert to deploy
    for module in model.net_g.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    # print(modelrestrity.net_g)
    # print(model)
    # name = opt['name']

    print("===================>> complete convert <<===================")
    print("===================>> test <<===================")
    # test
    input_tensor = torch.randn(1,3,50,50)
    y_train = modelrestrity.net_g(input_tensor)
    y_deploy = model.net_g(input_tensor)
    print('output is ', y_deploy.size())
    print('===================>> The test diff is:{}'.format(((y_deploy - y_train) ** 2).sum()))
    print('The diff should be close to 0')
    # print(((y_deploy - y_train) ** 2).sum())

    print(model.net_g)
    # save model
    print('===================>> saving model <<===================')
    current_iter=-2 # convert mode
    model.save_network(model.net_g, 'net_g', current_iter)
    print('===================>> complete <<===================')




    ## test:myconvert
    # count = 0
    # count_bsconv = 0
    # for module in model.modules():
    #     count = count+1
    #     if hasattr(module, 'switch_to_deploy'):
    #         count_bsconv=count_bsconv+1
    #         print('----------------module({})---------------:'.format(count_bsconv))
    #         print(module)
    #         module.switch_to_deploy()
    #         print('----------------module({})---------------:'.format(count_bsconv))
    #         print(module)
    # print(model)
    # print('count:{}  count_bsconv:{}'.format(count,count_bsconv))



if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    convert(root_path)
