import argparse 
import time
import torch 
import numpy as np
import json
import sys

from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from PIL import Image

def load_model():
    model_info = torch.load(args.model_checkpoint)
    model = model_info['model']
    model.classifier = model_info['classifier']
    model.load_state_dict(model_info['state_dict'])
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    pil_image = Image.open(image)
    # TODO: Process a PIL image for use in a PyTorch model
    trans = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
    tensor_image = trans(pil_image)
    return tensor_image

def classify_image(image_path, topk=5):
    model = load_model()
    device = torch.device("cuda" if args.gpu else "cpu")
    model.to(device)
    topk=int(topk)
    img_torch = process_image(image_path).unsqueeze_(0).float()
    with torch.no_grad():
        if (args.gpu):
            output = model.forward(img_torch.cuda())
        else:
            output = model.forward(img_torch.cpu())
    probability = F.softmax(output.data,dim=1)
    return probability.topk(topk)
  
def read_categories():
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name
    
def display_prediction(results):
    cat_to_name = read_categories()
    prob = np.array(results[0][0])
    cat = [cat_to_name[str(index + 1)] for index in np.array(results[1][0])]
    for i in range(len(prob)):
        print('Predicted Category: ', cat[i], ' Probability: ', round(prob[i]*100, 2), '%')
    return None
              
def parse():
    parser = argparse.ArgumentParser(description="Let's make prediction!")
    parser.add_argument('image_input', help='image file to classifiy (required)')
    parser.add_argument('model_checkpoint', default='checkpoint.pth', help='model used for classification')
    parser.add_argument('--top_k', default=5, help='how many prediction categories to show [default 5].')
    parser.add_argument('--category_names', default='cat_to_name.json', help='file for category names')
    parser.add_argument('--gpu', default=True, action='store_true', help='gpu option')
    args = parser.parse_args()
    return args

def main():
    global args
    args = parse() 
    image_path = args.image_input
    prediction = classify_image(image_path, topk = args.top_k)
    display_prediction(prediction)
    return prediction

main()