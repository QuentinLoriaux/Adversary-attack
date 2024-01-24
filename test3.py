# Version finale

import matplotlib.pyplot as plt
import torch, torchvision
import numpy as np
import sys
import os
from torchvision.models import vgg16


#NOTE : il faudra faire ça avec une 50aine d'img selon les consignes
#NOTE : Ici, on enlèvera les commentaires, on fera du code + propre et avec des fonctions
#       de manière à ce qu'on puisse facilement faire des tests
#       Faut implémenter d'autres réseaux aussi
#NOTE : Il faut faire de quoi sauvegarder les résultats sous forme d'image, et stocker les perturbations créées!

#NOTE : ça marche po très bien, ptet retirer le background du traitement.






# ============ Prédiction correcte ============

def VGGNetPrediction(choix_img, classes):
    # Charger le modèle VGGNet
    net = torchvision.models.vgg16(pretrained=True)
    net = net.cuda()

    with torch.no_grad():
        x = torch.stack([imgs[i] for i in choix_img], dim=0)
        x = torch.nn.functional.interpolate(x, size=(520, 520), mode='bilinear', align_corners=False)
        x = x.cuda()

        # Pour VGGNet, on peut utiliser directement le modèle sans transformation supplémentaire (W)
        _, l = net(x)[:, classes, :, :].max(1)  # prédiction des cartes de score de confiance

        return net, x, l

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



# a priori, y a moyen que des groupes se séparent et que des formes changent...
# ça fait plus de classes à afficher!

# ============ Affichage ============

#charge les images sous forme de liste de tenseurs
def preprocess_images(file_paths, size=520):
    images = []
    for file_path in file_paths:
        image = torchvision.io.read_image(file_path,mode=torchvision.io.ImageReadMode.RGB).float() / 255
        image = torch.nn.functional.interpolate(image.unsqueeze(0), size=size)[0]
        images.append(image)
    return images

def display(x, r, classes, lB):
    xr = x.cuda()+r.cuda()
    xr.cuda()
    _,lM = net(xr)["out"][:,classes,:,:].max(1)

    # visualisation des prédictions : il faut transformer les indices de classes en couleur
    class_to_color = {0: [0, 0, 0], 1: [255, 0, 0], 2: [0, 255, 0], 3: [0, 0, 255], 4: [0, 0, 255], 5: [0, 0, 255], 6: [0, 0, 255], 7: [0, 0, 255], 8: [0, 0, 255], 9: [0, 0, 255]}
    # Dans l'ordre de découpage du tenseur ligne 40
    # 0 : background
    # 1 : humain
    # 2 : chat
    # 3 : 

    rgb_tensorM = torch.zeros(x.shape)
    rgb_tensorB = torch.zeros(x.shape)
    for class_label, color in class_to_color.items():
        mask = (lM.cpu() == class_label).unsqueeze(1).float()
        rgb_tensorM += mask * torch.tensor(color).view(1, 3, 1, 1).float()
        mask = (lB.cpu() == class_label).unsqueeze(1).float()
        rgb_tensorB += mask * torch.tensor(color).view(1, 3, 1, 1).float()
    #mettre le bon nombre d'images
    imgBase = torch.cat([x[i] for i in range(x.size(0))],dim=-1).cuda()
    imgBSegmentation = torch.cat([rgb_tensorB[i] for i in range(x.size(0))],dim=-1).cuda()
    imgModified = torch.cat([xr[i] for i in range(x.size(0))],dim=-1).cuda()
    imgPerturbation = torch.cat([r[i] for i in range(x.size(0))],dim=-1).cuda()
    imgMSegmentation = torch.cat([rgb_tensorM[i] for i in range(x.size(0))],dim=-1).cuda()
    
    img = torch.cat([imgBase,imgBSegmentation, imgModified, imgPerturbation, imgMSegmentation],dim=1)
    img = img.cpu().numpy().transpose(1,2,0)

    dpi = plt.rcParams['figure.dpi']
    width_px = 1600
    height_px = 900
    plt.figure(figsize=(width_px/dpi, height_px/dpi))
    plt.imshow(img)
    plt.axis('off')
    plt.show()






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