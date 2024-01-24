# version plus Quentin

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
def VGGNetPrediction(classes, choix_img):
    model = vgg16(pretrained=True)
    model.eval()

    with torch.no_grad():
        x = torch.stack([imgs[i] for i in choix_img],dim=0)
        x = (W.transforms())(x).cuda() # ajuste notamment à 520 la taille
        _,l = net(x)["out"][:,classes,:,:].max(1) # prédiction des cartes de score de confiance
    return net, x, l


# ------------ traitement des arguments ------------

image_file_paths = [
    "./coco/217730183_8f58409e7c_z.jpg",
    "./coco/541870527_8fe599ec04_z.jpg",
    "./coco/2124681469_7ee4868747_z.jpg",
    "./coco/2711568708_89f2308b85_z.jpg",
    "./coco/2928196999_acd5471d23_z.jpg",
    "./coco/3016145160_497da1b387_z.jpg",
    "./coco/4683642953_2eeda0820e_z.jpg",
    "./coco/6911037487_cc68a9d5a4_z.jpg",
    "./coco/8139728801_60c233660e_z.jpg",
]

# ============ Affichage ============

#charge les images sous forme de liste de tenseurs
def preprocess_images(file_paths, size=520):
    images = []
    for file_path in file_paths:
        image = torchvision.io.read_image(file_path,mode=torchvision.io.ImageReadMode.RGB).float() / 255
        image = torch.nn.functional.interpolate(image.unsqueeze(0), size=size)[0]
        images.append(image)
    return images

#nombre d'images traitées
choix_img = [1,2]

imgs = preprocess_images(image_file_paths)

classes = [0,8,12,15]
net, x, l = ResnetPrediction(choix_img, classes)

# ============ Attaque avec DAG (Dense Adversary Generation) ============

#fonctions de perte
class targetedLoss(torch.nn.Module):
    def __init__(self):
        super(targetedLoss, self).__init__()

    def forward(self, z, condition, l, l_target):
        # Sélectionner les indices où condition est True
        indices = torch.where(condition)
        z_good = z[indices[0], l[indices], indices[1], indices[2]]
        z_bad = z[indices[0], l_target[indices], indices[1], indices[2]]

        loss = torch.sum(z_good - z_bad)

        return loss

class untargetedLoss(torch.nn.Module):
    def __init__(self):
        super(untargetedLoss, self).__init__()

    def forward(self, z, condition, l):
        # Sélectionner les indices où condition est True
        indices = torch.where(condition)
        z_good = z[indices[0], l[indices], indices[1], indices[2]]
        loss = torch.sum(z_good)
        return loss


#définir la liste des mauvaises prédictions
def permutation_sans_point_fixe(l, classes):
    lst = range(len(classes))
    perm = np.random.permutation(lst)
    while not all(x != y for x, y in zip(lst, perm)):
        perm = np.random.permutation(lst)
    
    value_mapping = {value: index for index, value in enumerate(perm)}
    with torch.no_grad():
        mapping_tensor = torch.tensor([value_mapping[val] for val in range(l.max() + 1)], dtype=l.dtype)
        l_target = mapping_tensor[l.cpu()]
        return l_target


def targetList(l, classes, random = True):
    if random:
        return permutation_sans_point_fixe(l.cpu(), classes)
    else :
        #Faire quelque chose de personnalisé

        return


def attack(net, x, targeted = False, l=[], classes=[], gamma = 0.5, maxIter = 3):
    xm = x.detach().requires_grad_()
    with torch.no_grad():
        if targeted :
            print("targeted attack")
            l_target = targetList(l, classes)
        else:
            print("untargeted attack")
        r = torch.zeros_like(x)
    m = 0

    net.cpu()
    net.zero_grad()
    print("debut boucle")

    while m < maxIter: 
        m += 1
        print("tour : " + str(m))
        
        zm = net(xm.cpu())["out"][:,classes,:,:].cpu().requires_grad_()
        
        with torch.no_grad():
            _,lm = zm.max(1)
            condition = torch.eq(lm.cuda(), l.cuda())
            somme = condition.sum()
            print("pixels correctement classés : " + str(somme.item()))
            
            if condition.sum() == 0: # tous les pixels sont mal classés : fin de l'algorithme
                break
                
        if targeted:
            loss = targetedLoss()
            loss = loss(zm.cuda(), condition.cuda(), l.cuda(), l_target.cuda())
        else:
            loss = untargetedLoss()
            loss = loss(zm.cuda(), condition.cuda(), l.cuda())

        loss.backward(retain_graph=False)

        rm= torch.zeros_like(x)
        indices = torch.where(condition)
        rm[indices[0],:,indices[1],indices[2]] = -xm.grad[indices[0],:,indices[1],indices[2]]

        rm = (gamma/rm.norm())*rm
        r += rm

        xm=xm.detach()
        xm += rm
        xm.grad = None
        xm.requires_grad_()
        
        del zm
        net.zero_grad()

    return r






# ============ MAIN ============


# ------------ traitement des arguments ------------

arguments = sys.argv[1:]

if len(arguments)==0:
    print("use: python3 clean_main.py [targeted] [load] [n° of save file]")
    print("Veuillez indiquer \"targeted\" pour une attaque targeted, ou n'importe quoi pour une untargeted.")
    sys.exit()

targeted = arguments[0] == "targeted"

load = False
try:
    load = arguments[1] == "load"
except IndexError as e:
    print("L'algorithme DAG va être lancé sans chargement de fichier de sauvegarde.")



# ------------ traitement des arguments ------------

image_file_paths = [
    "./coco/217730183_8f58409e7c_z.jpg",
    "./coco/541870527_8fe599ec04_z.jpg",
    "./coco/2124681469_7ee4868747_z.jpg",
    "./coco/2711568708_89f2308b85_z.jpg",
    "./coco/2928196999_acd5471d23_z.jpg",
    "./coco/3016145160_497da1b387_z.jpg",
    "./coco/4683642953_2eeda0820e_z.jpg",
    "./coco/6911037487_cc68a9d5a4_z.jpg",
    "./coco/8139728801_60c233660e_z.jpg",
]

#nombre d'images traitées
choix_img = [1,2]

imgs = preprocess_images(image_file_paths)

classes = [0,8,12,15]
net, x, l = ResnetPrediction(choix_img, classes)

if load :
    nb = int(input("Veuillez saisir le numéro du fichier à charger : "))
    try:
        r = torch.load('./saves/perturb' + str(nb)+'.pth')
    except FileNotFoundError as e:
        print("Le fichier n'existe pas.")
        sys.exit()
else :
    ask = int(input("Nombre maximum d'itérations : "))
    r = attack(net, x, targeted, l, classes, maxIter=ask)
    ask = input("Voulez-vous sauvegarder l'attaque? (y/n) : ")
    if ask == 'y':
        nb = 0
        while os.path.exists("./saves/perturb"+str(nb)+".pth"):
            nb += 1
        torch.save(r,"./saves/perturb"+str(nb)+".pth")

display(x,r, classes, l)