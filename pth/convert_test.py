import torch.nn.functional as F
import torch

def custom_padding(param, position,group=4):
    padding_length = param.shape[1]
    if(group == 4):
        if position == "first":
            return F.pad(param, (0, 0, 0, 0, 0, 3*padding_length, 0, 0))
        elif position == "second":
            return F.pad(param, (0, 0, 0, 0, padding_length, 2*padding_length, 0, 0))
        elif position == "third":
            return F.pad(param, (0, 0, 0, 0, 2*padding_length, padding_length, 0, 0))
        elif position == "forth":
            return F.pad(param, (0, 0, 0, 0, 3*padding_length, 0, 0, 0))
        
    elif(group == 2):
        if position == "up":
            return F.pad(param, (0, 0, 0, 0, 0, padding_length, 0, 0))
        elif position == "down":
            return F.pad(param, (0, 0, 0, 0, padding_length, 0, 0, 0))
        
    elif(group == 8):
        if position == "first":
            return F.pad(param, (0, 0, 0, 0, 0, 7*padding_length, 0, 0))
        elif position == "second":
            return F.pad(param, (0, 0, 0, 0, padding_length, 6*padding_length, 0, 0))
        elif position == "third":
            return F.pad(param, (0, 0, 0, 0, 2*padding_length, 5*padding_length, 0, 0))
        elif position == "forth":
            return F.pad(param, (0, 0, 0, 0, 3*padding_length, 4*padding_length, 0, 0))
        elif position == "fifth":
            return F.pad(param, (0, 0, 0, 0, 4*padding_length, 3*padding_length,0, 0))
        elif position == "sixth":
            return F.pad(param, (0, 0, 0, 0, 5*padding_length, 2*padding_length, 0, 0))
        elif position == "seventh":
            return F.pad(param, (0, 0, 0, 0, 6*padding_length, padding_length, 0, 0))
        elif position == "eighth":
            return F.pad(param, (0, 0, 0, 0, 7*padding_length, 0, 0, 0))

def process_grouped_params(layer_weight,group=4):
    # 将参数张量按照位置分成group组
    grouped_params = torch.chunk(layer_weight, group, dim=0)
    if(group == 4):
        # 分别对三个位置的参数进行填充操作
        group_first = custom_padding(grouped_params[0], "first",group)
        group_second = custom_padding(grouped_params[1], "second",group)
        group_third = custom_padding(grouped_params[2], "third",group)
        group_forth = custom_padding(grouped_params[3], "forth",group)
        return torch.cat((group_first, group_second, group_third,group_forth),dim=0)
    elif(group == 2):
        group_up = custom_padding(grouped_params[0], "up",group)
        group_down = custom_padding(grouped_params[1], "down",group)
        return torch.cat((group_up,group_down),dim=0)
    elif(group == 8):
        # 分别对三个位置的参数进行填充操作
        group_first = custom_padding(grouped_params[0], "first",group)
        group_second = custom_padding(grouped_params[1], "second",group)
        group_third = custom_padding(grouped_params[2], "third",group)
        group_forth = custom_padding(grouped_params[3], "forth",group)
        group_fifth = custom_padding(grouped_params[4], "fifth",group)
        group_sixth = custom_padding(grouped_params[5], "sixth",group)
        group_seventh = custom_padding(grouped_params[6], "seventh",group)
        group_eighth = custom_padding(grouped_params[7], "eighth",group)
        return torch.cat((group_first, group_second, group_third,group_forth,group_fifth,group_sixth,group_seventh,group_eighth),dim=0)


# 示例用法
# input_param = torch.randn(4, 1, 3, 3)  # 假设输入参数张量
# group1 = 4
# group2 = 2
# output_param1 = process_grouped_params(input_param,group1)
# output_param2 = process_grouped_params(input_param,group2)

# # print("input_size:{},group:{},output_size:{}".format(input_param.shape,group1,output_param1.shape))
# # print('output_param1:\n')
# # print(input_param)
# # print('output_param1:\n')
# # print(output_param1[0])
# # print(output_param1[1])
# # print(output_param1[2])
# # print(output_param1[3])
# print("input_size:{},group:{},output_size:{}".format(input_param.shape,group2,output_param2.shape))
# print(input_param)
# print('output_param2:\n')
# print(output_param2[0:2])
# print(output_param2[2:4])