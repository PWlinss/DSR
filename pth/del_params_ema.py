import torch

dict = torch.load('models/DRP-LKDN_x4.pth')  # load the pth file
for key in list(dict.keys()):
    print(key)
# for key in list(dict.keys()):
#     if key == 'params':
#         del dict[key]

new_dict = {}

# for key in list(dict['params_ema'].keys()):
#     new_dict[key] = dict['params_ema'][key]
for key in list(dict['params'].keys()):
    new_dict[key] = dict['params'][key]

torch.save(new_dict, 'models/DRP-LKDN_del_x4.pth')

changed_dict = torch.load('models/DRP-LKDN_del_x4.pth')
for key in list(changed_dict.keys()):
    print(key)
