import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})
import numpy as np
from sklearn import preprocessing

import pandas as pd
import glob
import ast
import argparse
import os

normalize = True
out_name = "male_vad_norm"
title = "Gender vad histogram-nomralized"
top = 0.3
buttom = 0 if normalize else -top


def get_from_multi_file(pattern):
    files = glob.glob(f'{pattern}*')
    dict_exp = {}
    for file in files:
        # print(file)

        with open(file) as f:
            lines = f.readlines()
            first_line = lines[0]
            attr = first_line[first_line.find("resnet50_")+9 :first_line.find("_train")]
            line_index  = -1
            while "Expectation of Abosulte Logit change:" not in lines[line_index]:
                line_index -= 1
            try:
                expectation = lines[line_index]
                dict_str = expectation[expectation.find('{'):]
                dict_exp_curr = ast.literal_eval(dict_str)
            except:
                line_index -= 1

                while "Expectation of Abosulte Logit change:" not in lines[line_index]:
                    line_index -= 1
                # line_index -= 26
                expectation = lines[line_index]
                dict_str = expectation[expectation.find('{'):]
                dict_exp_curr = ast.literal_eval(dict_str)

            for k in dict_exp_curr:
                if k not in dict_exp:
                    dict_exp[k] = float(dict_exp_curr[k])
                else:
                    dict_exp[k] = (dict_exp[k] + float(dict_exp_curr[k]))
    
    return {k:dict_exp[k]/len(files) for k in dict_exp}, attr

                    

def filter_string(input_str):
    afhq_filter = ['a ', 'with ', 'dog ', 'cat ']
    if 'dog' in  input_str or 'cat' in input_str:
        for c in afhq_filter:
            input_str = input_str.replace(c, '')
    else:            
        for c in ['a ', 'an ', 'face ',' face', 'with ', 'of ', ' individual',  'individual ', 'portrait ', 'person ', ' person']:
            input_str = input_str.replace(c, '')
    return input_str

def histogram(weights, output, title, top=None, buttom=0, normalize=True,verbose = False):
    os.makedirs(os.path.dirname(output), mode=0o777, exist_ok=True)
    list_of_attributes = weights.keys()
    attribute_list = []
    weight_filtered = {}
    for a in list_of_attributes:
        new_str = a
        attribute_list += [filter_string(new_str)]
        weight_filtered[filter_string(new_str)] = weights[a]

    attribute_list.sort()
    if verbose:
        print(attribute_list)

    if normalize:
        barchart = [abs(float(weight_filtered[x])) for x in attribute_list]
        barchart = barchart / np.sum(barchart)
    else:
        barchart = [float(weight_filtered[x]) for x in attribute_list]


    plt.figure(figsize=(12 * 0.8, 9 * 0.8))

    plt.gcf().subplots_adjust(bottom=0.35)
    plt.bar(attribute_list, barchart, color='cornflowerblue',alpha=0.85)
    plt.title(title)
    plt.xticks(rotation=75)
    if top == None:
        top = (round((2*max(barchart)  - 0.04), 1) + 0.1)/2

    plt.ylim(buttom, top)
    plt.savefig(output)
    plt.close()


def to_excel(input, out_file, normalize=True):

    os.makedirs(os.path.dirname(out_file), mode=0o777, exist_ok=True)
    if type(input) is str:
        file1 = open(input, 'r')
        Lines = file1.readlines()

        attrs = []
        for i in range(len(Lines)):
            line = Lines[i]
            if '[' not in line:
                attrs += [line[:-1]]
            else:
                break


        all_list = []
        curr_list = []
        for i in range(i, len(Lines)):
            line = Lines[i]
            if '[' in line:
                curr_list = []
                line = line[1:]
                
            curr_list  += line.split(' ')
            if curr_list[0] == '':
                curr_list = curr_list[1:]
            curr_list[-1] = curr_list[-1][:-1]

            if ']' in line:
                curr_list[-1] = curr_list[-1][:-1]
                curr_list = [float(x) for x in curr_list if x != '']
                if len(curr_list) != 15:
                    curr_list += [0]
                all_list += [curr_list]


        print(attrs)
        print(all_list)

        dic = dict(zip(attrs, all_list))
    elif type(input) is dict:
        key = [filter_string(k) for k in sorted(input.keys())]
        val = [float(input[k]) for k in sorted(input.keys())]

        if normalize:
            val = val / np.sum(val)

        dic = {'attr': key, 'value': val} 
    print(dic)
    df = pd.DataFrame(dic)
    writer = pd.ExcelWriter(out_file, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1', index=False)


    workbook  = writer.book
    worksheet = writer.sheets['Sheet1']
    format1 = workbook.add_format({'num_format': '0.0000000'})
    worksheet.set_column('A:I', None, format1)  # Adds formatting to column C

    writer.save()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--pattern','-p', type=str, default=None)
    parser.add_argument('--top','-t', type=str, default=None)
    parser.add_argument('--del_keys','-d', nargs='+', default=[])


    args = parser.parse_args()

    patterns  = [] + [args.pattern]

    # del_keys = ["Smiling", "Beard", "Pale",]
    del_keys = ["Black", "Brown", "Bald", "Male", "Eyeglasses", "Mouth", "Mustache","Young" ]

    del_keys += list(args.del_keys)

    print(f"removing keys {del_keys}")

    for file_pattern in patterns:
        logit, attr = get_from_multi_file(file_pattern)
        
        del_items = []
        for l in logit:
            for k in del_keys:
                if k in l:
                    del_items += [l]
                    break
                
        for d in del_items:
            logit.pop(d)

        
        histogram(logit, f"images/{file_pattern[:-4]}.png", f"{attr} classifier", top = args.top)
        to_excel({k: float(logit[k]) for k in logit}, f"images/excel/{file_pattern[:-4]}.xlsx")

