# Version plus chatgpt

import torch, torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision.models import vgg16
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np


#images sous forme de tenseurs flottants avec pixels normalisés sur [0,1]
im1 = torchvision.io.read_image("./coco/217730183_8f58409e7c_z.jpg").float()/255
im2 = torchvision.io.read_image("./coco/541870527_8fe599ec04_z.jpg").float()/255
im3 = torchvision.io.read_image("./coco/2124681469_7ee4868747_z.jpg",mode=torchvision.io.ImageReadMode.RGB).float()/255
im4 = torchvision.io.read_image("./coco/2711568708_89f2308b85_z.jpg").float()/255
im5 = torchvision.io.read_image("./coco/2928196999_acd5471d23_z.jpg").float()/255
im6 = torchvision.io.read_image("./coco/3016145160_497da1b387_z.jpg").float()/255
im7 = torchvision.io.read_image("./coco/4683642953_2eeda0820e_z.jpg").float()/255
im8 = torchvision.io.read_image("./coco/6911037487_cc68a9d5a4_z.jpg").float()/255
im9 = torchvision.io.read_image("./coco/8139728801_60c233660e_z.jpg").float()/255

#resize en carrés de 520x520px
im1 = torch.nn.functional.interpolate(im1.unsqueeze(0), size=520)[0]
im2 = torch.nn.functional.interpolate(im2.unsqueeze(0), size=520)[0]
im3 = torch.nn.functional.interpolate(im3.unsqueeze(0), size=520)[0]
im4 = torch.nn.functional.interpolate(im4.unsqueeze(0), size=520)[0]
im5 = torch.nn.functional.interpolate(im5.unsqueeze(0), size=520)[0]
im6 = torch.nn.functional.interpolate(im6.unsqueeze(0), size=520)[0]
im7 = torch.nn.functional.interpolate(im7.unsqueeze(0), size=520)[0]
im8 = torch.nn.functional.interpolate(im8.unsqueeze(0), size=520)[0]
im9 = torch.nn.functional.interpolate(im9.unsqueeze(0), size=520)[0]


# Charger le modèle cible (VGGNet)
model = vgg16(pretrained=True)
model.eval()


with torch.no_grad():
    x = torch.stack([im1,im2],dim=0)
    x = (W.transforms())(x).cuda()
    z = net(x)["out"] # prédiction des cartes de score de confiance
    classes = [0,8,12,15,    16,3,2,4,5]
    _,l = z[:,classes,:,:].max(1) # on ne garde que les classes fond, personne, chat et chien et les fausses
    l.requires_grad_(False)
    # exemple : z[:,0,:,:] donne le score du background sur chaque pixel 

# J'utilise la fonction de perte de segmentation de la librairie pytorch pour s'assurer 
# d'avoir des fonctions qui marchent
criterion = nn.CrossEntropyLoss()  # Fonction de perte
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Optimiseur (SGD avec un taux d'apprentissage de 0.01)

# Générer un exemple adversaire
def generate_adversarial_example(inputs, labels):
    inputs.requires_grad_()  # Activer le suivi des gradients sur les entrées
    optimizer.zero_grad()
    outputs = model(inputs.cpu())
    loss = criterion(outputs, labels.cpu())
    loss.backward()

    # Appliquer la perturbation aux entrées
    perturbation = 0.01 * inputs.grad.sign()
    inputs_adversarial = inputs + perturbation

    return inputs_adversarial

# Valider l'attaque sur l'exemple adversaire
def validate_attack(inputs_adversarial, labels):
    outputs_adversarial = model(inputs_adversarial.cpu())
    _, predicted = torch.max(outputs_adversarial, 1)
    accuracy = (predicted == labels.cpu()).float().mean()
    return accuracy.item()

# Charger le modèle cible (VGGNet)
model = vgg16(pretrained=True)
model.eval()

# Modifier les caractéristiques d'entrée
input_image = preprocess(input_image)

# Générer un exemple adversaire
adversarial_example = generate_adversarial_example(input_image, target_label)

# Valider l'attaque sur l'exemple adversaire
accuracy_adversarial = validate_attack(adversarial_example, target_label)

print(f"Accuracy on adversarial example: {accuracy_adversarial}")