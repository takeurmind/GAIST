import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import zipfile
import io
import csv
import time
import os
import intel_extension_for_pytorch as ipex
from datetime import datetime
classes = ["T-shirt/Top", "Trouser", "Pullover", "Dress", "Coat", 
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]

def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return preprocess(image).unsqueeze(0)

def perform_inference(model, preprocessed_image):
    with torch.no_grad():
        outputs = model(preprocessed_image)
        _, predicted = torch.max(outputs, 1)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
        return predicted, probabilities

def process_images_from_zip(file_path, model, device, csv_file_path):
    if device == 'cpu' or  device == 'cpuIpex':
        device = torch.device('cpu')
    elif device == 'gpu' :
        device = torch.device('cuda')


    with os.scandir(file_path) as ref, open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        for image_file in ref:
            if image_file.name.lower().endswith(('.png', '.jpg', '.jpeg')) :
                image = Image.open(file_path+'/'+image_file.name)
                preprocessed_image = preprocess_image(image)
                predicted, probabilities = perform_inference(model, preprocessed_image.to(device))

                _, top5_indices = torch.topk(probabilities, 5)
                top5_probabilities = probabilities[top5_indices]

                row = [image_file.name]
                for idx, prob in zip(top5_indices, top5_probabilities):
                    row.extend([classes[idx], f'{prob.item():.2f}%'])
                writer.writerow(row)

def call_model(mode,model_s):
    if mode == 'cpu' or mode =="cpuIpex":
        device = torch.device('cpu')
    elif mode == 'gpu' :
        device = torch.device('cuda')

    if model_s == 'resnet18':
        modelx = models.resnet18(pretrained=False)
        num_ftrs = modelx.fc.in_features
        modelx.fc = nn.Linear(num_ftrs, 10)
        modelx.load_state_dict(torch.load('./GAIST/model_pth/00_PyTorch_FashionMNIST_ResNet18_1.pth' , map_location=torch.device(device)))

    elif model_s == 'vgg16':
        modelx = models.vgg16(pretrained=False)
        modelx.classifier[6] = nn.Linear(4096, 10)
        
        modelx.load_state_dict(torch.load('./GAIST/model_pth/01_PyTorch_FashionMNIST_VGGNet16.pth' , map_location=torch.device(device)))

    elif model_s == 'mobilenet':
        modelx = models.mobilenet_v2(pretrained=False)
        num_ftrs = modelx.classifier[1].in_features
        modelx.classifier[1] = nn.Linear(num_ftrs, 10)
        modelx.load_state_dict(torch.load('./GAIST/model_pth/02_PyTorch_FashionMNIST_MoblieNet_1.pth' , map_location=torch.device(device)))

    modelx = modelx.to(device)
    modelx.eval()

    if mode =="cpuIpex" :
        torch.set_num_threads(4)
        modelx = ipex.optimize(modelx, weights_prepack=False)
        
    return modelx

def bench(zip_path,csv_path):
    mode = [ 'GAIST', 'GPU','CPU']
    model = ['MobileNet','ResNet18', 'VGG16']
    x = datetime.now()
    csv_path = csv_path + x.strftime("%m월 %d일 %H시 %M분") + '_' +'bechmark_data.csv'
    csv_filename = os.path.join(csv_path)
    for mode_index in mode:
        if mode_index == 'CPU' or  mode_index == 'GAIST':
            device = torch.device('cpu')
        elif mode_index == 'GPU' :
            device = torch.device('cuda')
        for model_index in model:
            if model_index == 'ResNet18':
                modelx = models.resnet18(pretrained=False)
                num_ftrs = modelx.fc.in_features
                modelx.fc = nn.Linear(num_ftrs, 10)
                modelx.load_state_dict(torch.load('./GAIST/model_pth/00_PyTorch_FashionMNIST_ResNet18_1.pth' , map_location=torch.device(device)))

            elif model_index == 'VGG16':
                modelx = models.vgg16(pretrained=False)
                modelx.classifier[6] = nn.Linear(4096, 10)
                
                modelx.load_state_dict(torch.load('./GAIST/model_pth/01_PyTorch_FashionMNIST_VGGNet16.pth' , map_location=torch.device(device)))

            elif model_index == 'MobileNet':
                modelx = models.mobilenet_v2(pretrained=False)
                num_ftrs = modelx.classifier[1].in_features
                modelx.classifier[1] = nn.Linear(num_ftrs, 10)
                modelx.load_state_dict(torch.load('./GAIST/model_pth/02_PyTorch_FashionMNIST_MoblieNet_1.pth' , map_location=torch.device(device)))

            modelx = modelx.to(device)
            modelx.eval()
            if mode =="GAIST" :
                torch.set_num_threads(4)
                modelx = ipex.optimize(modelx, weights_prepack=False)
            stime = time.time()
            total_elapsed_time = 0
            image_count = 0
            total_ac = 0 
            with os.scandir(zip_path) as ref:
                
                for image_file in ref:
                    if image_file.name.lower().endswith(('.png', '.jpg', '.jpeg')) :
                        start_time = time.time()
                        image = Image.open(zip_path+'/'+image_file.name)
                        preprocessed_image = preprocess_image(image)
                        predicted, probabilities = perform_inference(modelx, preprocessed_image.to(device))

                        _, top5_indices = torch.topk(probabilities, 1)
                        top5_probabilities = probabilities[top5_indices]
                        total_ac += top5_probabilities[0].item()
                        elapsed_time = time.time() - start_time
                        total_elapsed_time += elapsed_time
                        image_count += 1

            if not os.path.exists(csv_filename):
                with open(csv_filename, 'w', newline='') as csvfile:
                    csv_writer = csv.writer(csvfile, delimiter=',')
                    csv_writer.writerow(['PROCESSOR MODE' , 'MODEL', 'INFERENCE TIME' , 'ACCURACY SCORE' ])

            with open(csv_filename, 'a', newline='') as csvfile:
                csv_writer = csv.writer(csvfile, delimiter='\t')
                csv_writer.writerow([ mode_index, model_index, f"{total_elapsed_time:.2f}",  f"{total_ac/image_count:.2f}"])
    return csv_path