import os

import numpy as np
import PIL.Image
import PIL.ImageDraw, PIL.ImageFont
import torch
from torch.nn.functional import normalize


from stylegan.wrapper import Generator
from stylegan.embedding import get_delta_t
from stylegan.manipulator import Manipulator
from stylegan.mapper import get_delta_s, get_delta_s_top

import class_labels


from torchvision.models import resnet
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm


import clip
import argparse

from scripts.histogram import histogram, filter_string
from scripts.draw_label import draw_label

exp_list = ['ffhq', 'afhqdog', 'afhqcat', 'church', 'car']
draw_logit = True


def process_experiment(args):
    exp = args.experiment 
    sg_path = {
        x : f'./pretrained/{x}.pkl' for x in exp_list
    }
    fs3_path = {
        x : f'./tensor/fs3{x}.npy' for x in exp_list

    }

    neutral = {
        'ffhq': 'a face',
        'afhqcat': 'a cat',
        'church': 'a church',
        'car': 'a car',
        'afhqdog': 'a dog'
    }

    args.stylegan_path = sg_path[exp]
    args.fs3_path = fs3_path[exp]
    args.neutral = neutral[exp]
    return args


def findMaxS(style, dic):
    s_list = np.array([])
    for a in style:
        s_list= np.append(s_list, style[a].cpu().detach().numpy())
    
    top_ind = np.where(np.abs(s_list) > 1.5)

    for dd in top_ind[0]:
        d = int(dd)
        if d in dic:

            dic[d] = dic[d] + 1
        else:
            dic[d] = 1

    dic_list = sorted(dic.items(), key=lambda item: item[1])
    dic_list.reverse()
    print({k: v for k, v in dic_list[:30]})
    return dic


def prepare_label(args, device, beta_threshold=0.1):
    '''
    Prepares the label for the given experiment and target attribute.

    Args:
    - args: Namespace object containing the experiment, target_attr, num_attr, neutral, stylegan_path, and fs3_path.
    - device: Device to run the model on.
    - beta_threshold: Threshold value for beta or num of top channels .

    Returns:
    - delta_s_dict: Dictionary containing delta_s values for each target attribute.
    - label_dict: Edit weights, initialized to 0 here for following attacks.
    '''

    
    if args.target_attr == None:
        labels, label_beta =  getattr(class_labels, args.experiment)()
    else:
        labels, label_beta =  args.target_attr.split(','), {}

    labels = labels[:args.num_attr]

    neutral = args.neutral

    ckpt = args.stylegan_path
    G = Generator(ckpt, device)
    model, preprocess = clip.load("ViT-B/32", device=device)
    fs3 = np.load(args.fs3_path)
    manipulator = Manipulator(G, device)

    avg = 0

    delta_s_dict = {}
    for target in labels:
        classnames=[neutral, target]
        delta_t = get_delta_t(classnames, model)

        if beta_threshold < 5:
            if target not in label_beta:
                delta_s, num_channel = get_delta_s(fs3, delta_t, manipulator, beta_threshold=beta_threshold)
            else:
                delta_s, num_channel = get_delta_s(fs3, delta_t, manipulator, beta_threshold=label_beta[target])

            d_s_array = convert(delta_s)
            d_s_array =  normalize(torch.tensor(d_s_array), dim = 0)
            delta_s = convert(d_s_array, delta_s)

        else:
            delta_s, num_channel = get_delta_s_top(fs3, delta_t, manipulator, num_top=int(beta_threshold))
            d_s_array = convert(delta_s)
            d_s_array =  normalize(torch.tensor(d_s_array), dim = 0)
            delta_s = convert(d_s_array, delta_s)

        print(f"{target}:{num_channel}")
        avg += num_channel

        delta_s_dict[target] = delta_s

    print(f"average channel is {avg/len(labels)}")

    return delta_s_dict, {a:torch.tensor([0.], device=device, requires_grad=True) for a in labels}
    



def convert(s,sample_dict = {}):

    if type(s) is dict:
        output = []
        for x in s:
            output += list(s[x].detach().cpu().numpy())
        return np.array(output)
    else:
        dic = dict()
        ind = 0
        for layer in sample_dict: # 26
            dim = sample_dict[layer].shape[-1]

            dic[layer] = s[ind:ind+dim].to(sample_dict[layer].device)
            ind += dim
        return dic



custom_transform_clip = transforms.Compose([transforms.Resize(size=224), 
                                    transforms.CenterCrop(size=(224, 224)),
                                    transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))])
custom_transform_cls = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

custom_transform_resize = transforms.Compose([transforms.Resize(size=224)])

def main(args):

    args = process_experiment(args)

    print(args)

    np.random.seed(args.seed)
    device = torch.device(args.device)

    victim_model = resnet.resnet50(pretrained=False)
    num_fc_in_features = victim_model.fc.in_features
    victim_model.fc = torch.nn.Linear(num_fc_in_features, 1)
    victim_model.load_state_dict(torch.load(args.target_model_path, map_location='cpu'))
    victim_model.to(device)
    victim_model.eval()

    G = Generator(args.stylegan_path, device)

    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

    bcelogit_loss = nn.BCEWithLogitsLoss().to(device)
    ce_loss = nn.CrossEntropyLoss().to(device)
    # tv_loss = kornia.losses.TotalVariation().to(device)

    clip_text_consistent = clip.tokenize(args.clip_token).to(device)

    # attr = args.target_model_path[args.target_model_path.find("resnet50_")+9 :args.target_model_path.find("_trainfull")]
    attr = os.path.basename(args.target_model_path)
    if args.outdir == None:
        outdir = f'./output/{args.experiment}/{attr}/'
    else:
        outdir = args.outdir
    print(f"output folder is {outdir}")

    if args.mode == "single":
        os.makedirs(f'{outdir}/single/', mode=0o777, exist_ok=True)
    else:
        os.makedirs(f'{outdir}/multiple/', mode=0o777,exist_ok=True)

    cls_weight = args.cls_weight
    clip_weight = args.clip_weight
    tv_weight = args.tv_weight
    l2_weight = args.l2_weight


    ds_label_ori, ds_weights_ori = prepare_label(args, device, args.style_beta_or_channel)
    overall = {a:0 for a in ds_weights_ori}
    overall_changes = {a:0 for a in ds_weights_ori}
    overall_logit_changes = {a:0 for a in ds_weights_ori}

 

    def attack( target, styles , index , combination=[], text=False, save_img = True):
        ds_label = ds_label_ori.copy()
        ds_weights = ds_weights_ori.copy()
        ds_weights_adv =  ds_weights_ori.copy()

        single_mode =  target != ""  
        if type(target) is list:
            ds_label = { your_key: ds_label[your_key] for your_key in target }
            ds_weights = { your_key: ds_weights[your_key] for your_key in target }
            ds_weights_adv = { your_key: ds_weights_adv[your_key] for your_key in target }

        for attack_iter in range(args.attack_iter):
            s = styles.copy()

            if single_mode:
                ds_weights[target].requires_grad=True
                for x in s:
                    ds = ds_label[target][x]
                    s[x] = s[x] + ds_weights[target] * ds
            else:
                for target in ds_label:
                    ds_weights[target].requires_grad=True
                    for x in s:
                        ds = ds_label[target][x]
                        s[x] = s[x] + ds_weights[target] * ds

            # ================================ Generate Image and forward ===================================
            img = G.synthesis_from_stylespace(w, s)
            img_resize224 = custom_transform_resize(img)
            img_resize224 = (img_resize224 * 127.5 + 128).clamp(0, 255) / 255.0
            img_resize224 = custom_transform_cls(img_resize224)
            
            victim_logit = victim_model(img_resize224).type(torch.float64)
            clip_image = custom_transform_clip((img+1)/2)
            logits_per_image, logits_per_text = clip_model(clip_image, clip_text_consistent)
            clip_prob = logits_per_image.softmax(dim=-1)


            if attack_iter == 0:
                original_image = img.clone().detach()
                clip_pred = clip_prob.argmax(dim=-1).to(device)
                clip_pred.requires_grad = False
                ground_truth = (torch.sigmoid(victim_logit) > 0.5).type(torch.float64).to(device)
                ground_truth_inv = (torch.sigmoid(victim_logit) <= 0.5).type(torch.float64).to(device)
                gt = float(torch.sigmoid(victim_logit))
                
                ground_truth.requires_grad = False

                img_normalize = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                output_img = PIL.Image.fromarray(img_normalize[0].cpu().numpy(), 'RGB')
                output_img = output_img.resize((512,512))
                if draw_logit:
                    output_img = draw_label(output_img, f"{gt:.2f}" )
                output_img_ori = output_img

            # ================================ Backward ===================================


            cls_cost = bcelogit_loss(victim_logit, ground_truth_inv)
            clip_cost = ce_loss(logits_per_image, clip_pred)
            # tv_cost = tv_loss(img)

            victim_model.zero_grad()
            G.G.zero_grad()
            
            final_cost = cls_weight * cls_cost + clip_weight * clip_cost # + tv_weight * tv_cost
            final_cost.backward()

            # ================================ Draw Image ===================================
       
            if attack_iter == args.attack_iter - 1:
                if save_img:
                    img_normalize = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                    output_img = PIL.Image.fromarray(img_normalize[0].cpu().numpy(), 'RGB')
                    output_img = output_img.resize((512,512))
                    
                    if draw_logit:
                        output_img = draw_label(output_img, f"{float(torch.sigmoid(victim_logit)):.2f}" )
                    if single_mode:
                        output_img_ori.save(f'{outdir}/single/{index}--original.png')
                        output_img.save(f'{outdir}/single/{index}-{attack_iter:04d}-{target}.png')
                    else:
                        output_img_ori.save(f'{outdir}/multiple/{index}--original.png')
                        output_img.save(f'{outdir}/multiple/{index}-{attack_iter:04d}.png')

                    output_img.close()
                attacked_s = s.copy()

            # ================================ Next Iter ===================================

            s = styles.copy()
            if single_mode:
                ds_weights_adv[target] = ds_weights[target] - args.attack_step_size * ds_weights[target].grad
                ds_weights_adv[target] = ds_weights_adv[target].clamp(-args.attack_bound, args.attack_bound)
                ds_weights[target] = ds_weights_adv[target].detach()

            else:
                for target in ds_label:
                    ds_weights_adv[target] = ds_weights[target] - args.attack_step_size * ds_weights[target].grad
                    ds_weights_adv[target] = ds_weights_adv[target].clamp(-args.attack_bound, args.attack_bound)
                    ds_weights[target] = ds_weights_adv[target].detach()


        # ================================= ATTACK FINISHED =================================#####

        if single_mode:

            logit_changes[target] = float(torch.sigmoid(victim_logit)) - gt
            print(f"attr: {target} | logit changes: {float(logit_changes[target])} |   weight {float(ds_weights[target])}")
            
        else:
            print( f"logit changes: {float(torch.sigmoid(victim_logit)) - gt}")
            print(f"final logit: {float(torch.sigmoid(victim_logit))}")
            for target in ds_label:
                logit_changes[target] = float(torch.sigmoid(victim_logit)) - gt

            print({k:f"{float(ds_weights[k]):.6f}" for k in ds_weights})
        flipped = (float(torch.sigmoid(victim_logit)) > 0.5) != (gt > 0.5)

        del original_image
        return ds_weights, logit_changes, attacked_s, flipped


    all_flipped = 0
    for i in range(args.num_sample):
        
        truncation_psi = 0.7

        z_ori = np.random.randn(1, G.G.z_dim)

        z_ori = torch.from_numpy(z_ori).to(device)
        z = z_ori.clone().detach()
        w_ori = G.mapping(z, truncation_psi=truncation_psi, truncation_cutoff=8)
        styles = G.mapping_stylespace(w_ori)

        s = styles.copy()
        for x in s: 
            s[x].detach()
        w = w_ori.clone().detach()

        print(f"===========================Sample Index {i}==========================")


        ds_label, ds_weights = ds_label_ori.copy(), ds_weights_ori.copy()
        logit_changes = {}
        
        us_loc = [pos for pos, char in enumerate(args.target_model_path) if char == '_']
        classifier_name = args.target_model_path[us_loc[-2]: us_loc[-1]]
        if args.mode in "single_attribute":
            for target in ds_label:
                ds_weights, logit_changes, attacked_s, flipped = attack(target, styles, i, text=True)
                overall_logit_changes[target] += abs( float(logit_changes[target]) )
                overall[target] += float(ds_weights[target])
                overall_changes[target] += abs(float(ds_weights[target]))

            print("Expectation of Abosulte Logit change:", {k:f"{v/(i+1):.15f}" for k,v in sorted(overall_logit_changes.items(), key=lambda item: abs(item[1]))})
            
            # Since histogram function normalize the logits, we don't need to take average (v/{i+1}) here
            histogram(overall_logit_changes, f"{outdir}/barchart-logit-single.png", f"{args.experiment} on {classifier_name}" )
        
        else:
            ds_weights, logit_changes, attacked_s, flipped = attack( "", styles, i, text=True)
            for a in overall:
                overall[a] += float(ds_weights[a])
                overall_changes[a] += abs(float(ds_weights[a]))
            if flipped:
                all_flipped += 1
            print(f"current flip rate: {all_flipped /(i+1)}")

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ZOOM Hyperparameters')

    parser.add_argument('--seed', type=int, default=0, help='random seed')

    parser.add_argument('--device', type=str, default='cuda', help='cuda device')

    parser.add_argument('--target_model_path', type=str, required=True, help='Target(victim) model path')

    parser.add_argument('--clip_token', default=["a cat", "a dog"], help='Clip token, This is used to prevent the model cheating by edit the target attribute, so we use clip to maintain the original attribute unchanged')

    parser.add_argument('--cls_weight', type=float, default=50, help='Classification loss weight')

    parser.add_argument('--clip_weight', type=float, default=0.005, help='Clip loss weight')

    parser.add_argument('--tv_weight', type=float, default=0.001, help='Total Variation loss weight')

    parser.add_argument('--l2_weight', type=float, default=0, help='L2 loss weight')

    parser.add_argument('--num_sample', type=int, default=1000, help='Number of images sampled from StyleGAN')

    parser.add_argument('--attack_iter', type=int, default=100, help='Number of attack iterations per image')
    
    parser.add_argument('--num_attr', type=int, default=10000, help='Number of a first attr used in attr list, used in controlling attributes(only used for table1, for other experiments we use all attributes)')

    parser.add_argument('--attack_step_size', type=float, default=1, help='Attack step size')

    parser.add_argument('--attack_bound', type=float, default=20, help='Attack bound. This is set to clamp the weights and prevent outragous edits weights')

    parser.add_argument('--style_beta_or_channel', type=float, default=0.1, help='if this < 5, then it is beta, else it is num_channel ')

    parser.add_argument('--experiment', type=str, default='ffhq', choices=exp_list, help='experiment name')

    parser.add_argument('--outdir', type=str, default=None,help='output image place')

    parser.add_argument('--target_attr', type=str, default=None, help='single target')

    parser.add_argument('--mode', type=str, default='single', choices=["single", "multiple"], help='experiment mode, in single model, only one attribute is edited at a time and we use this mode to draw histograms and analysis. In multiple mode, ALL attributes are optimized at the same time to create the most powerful conterfactuals.')


    args = parser.parse_args()
    main(args)