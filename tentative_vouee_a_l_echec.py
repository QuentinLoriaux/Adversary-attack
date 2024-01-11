import matplotlib.pyplot as plt
import torch, torchvision

#NOTE : il faudra faire ça avec une 50aine d'img selon les consignes

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


# ============ Prédiction correcte ============

W = torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
net = torchvision.models.segmentation.deeplabv3_resnet50(weights=W)
net = net.cuda()

with torch.no_grad():
  x = torch.stack([im1,im2,im3,im4,im5,im6,im7,im8,im9],dim=0)
  x = (W.transforms())(x).cuda()
  z = net(x)["out"] # prédiction des cartes de score de confiance
  z = z[:,[0,8,12,15],:,:] # we keep only person, cat and dog class
   #exemple : z[:,0,:,:] donne le score du background sur chaque pixel
   l,ind = z.max(1) # on prend le meilleur score


# ============ Bagarre ============
#Je tente le DAG (Dense Adversary Generation), voir [p.3] Xie_Adversarial_Examples_for_ICCV_2017_paper.pdf



#définir la liste des mauvaises prédictions (par permutation parmi les mauvaises par ex)
#En fait, je crois que c'est déjà en quelques sortes une attaque "targeted"

#untargeted correspondrait simplement à une descente de gradient où on fait baisser le bon score jusqu'à ce que ce soit faux
#ça a l'air + facile du coup je vais faire ça


maxIter = 200
correctSet = 

while m<maxIter and correctSet[0]

#définir la fonction de perte à minimiser
# un truc du style  net(x+r)[out][:,BONS,:,:] - net(x+r)[out][:,FAUX,:,:]  avec r la perturbation
# à vérifier 



# a priori, y a moyen que des groupes se séparent et que des formes changent... ça fait plus de classes
# à afficher!

# ============ Affichage ============

# visualisation des prédictions : il faut transformer les indices de classes en couleur
class_to_color = {0: [120, 120, 0], 1: [255, 0, 0], 2: [0, 255, 0], 3: [0, 0, 255]}

rgb_tensor = torch.zeros(9,3,520,520).cuda()
for class_label, color in class_to_color.items():
    mask = (z == class_label).unsqueeze(1).float()
    rgb_tensor += mask * torch.tensor(color).view(1, 3, 1, 1).float().cuda()

visu = torch.cat([im1,im2,im3,im4,im5,im6,im7,im8,im9],dim=-1)
visubis = torch.cat([rgb_tensor[i] for i in range(9)],dim=-1).cpu()
visu = torch.cat([visu,visubis],dim=1)
visu = visu.cpu().numpy().transpose(1,2,0)

dpi = plt.rcParams['figure.dpi']
width_px = 1600
height_px = 400
plt.figure(figsize=(width_px/dpi, height_px/dpi))
plt.imshow(visu)
plt.axis('off')
plt.show()