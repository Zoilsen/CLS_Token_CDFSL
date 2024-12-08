# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
from methods import vision_transformer as vits


# --- VIT with dino ---
def vit_small(**kwargs):
        model = vits.vit_small()
        
        state_dict = torch.load('methods/dino_deitsmall16_pretrain.pth')

        #model.load_state_dict(state_dict, strict=True)
        model_state_dict = model.state_dict()
        num_loaded_params = 0
        for name in model_state_dict.keys():
            if name in state_dict.keys():
                if 'cls_token' in name:
                    continue

                model_state_dict[name] = state_dict[name]
                num_loaded_params += 1

        model.load_state_dict(model_state_dict, strict=True)

        return model


# --- VIT with pytorch implementation ---
def vit_b_im(**kwargs):
        import torchvision
        model = torchvision.models.vit_b_16(weights='IMAGENET1K_V1')       
        return model



model_dict = dict(VIT_S = vit_small,
                  VIT_B_im = vit_b_im)
