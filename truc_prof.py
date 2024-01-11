
# ========================= SECTION 1 =========================

#déziper coco.zip et mettre le dossier "coco" dans le repository.


# ========================= SECTION 2 =========================

import matplotlib.pyplot as plt
import torch, torchvision


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





#Cette partie ne sert qu'à afficher les images de départ
import matplotlib.pyplot as plt

#concaténation, alignement selon la dernière coord (dim=-1) => horizontal
visu = torch.cat([im1,im2,im3,im4,im5,im6,im7,im8,im9],dim=-1)
visu = visu.cpu().numpy().transpose(1,2,0)
dpi = plt.rcParams['figure.dpi']
width_px = 1600
height_px = 200
plt.figure(figsize=(width_px/dpi, height_px/dpi))
plt.imshow(visu)
plt.axis('off')
#plt.show()

# ========================= SECTION 3 =========================

#On charge des poids précalculés pour des images du style de notre batch d'image.
#VOC = Visual Object Classes
W = torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
#On crée une instance du modèle DeepLabV3 qu'on va plus tard utiliser 
net = torchvision.models.segmentation.deeplabv3_resnet50(weights=W)
net = net.cuda()
#['__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

#no_grad pour pas calculer des calculs de gradient dans le code (économie de mémoire)
with torch.no_grad():
  #On crée une liste de tenseurs indexés sur la dimension 0
  x = torch.stack([im1,im2,im3,im4,im5,im6,im7,im8,im9],dim=0)
  #Transformations obscures sur la liste avec les poids
  x = (W.transforms())(x).cuda()
  z = net(x)["out"] # prédiction des cartes de score de confiance
  z = z[:,[0,8,12,15],:,:] # we keep only person, cat and dog class, 0=background
  l,z = z.max(1) # on prend le meilleur score

# print("score")
# print(l)
# print("index")
# print(z)





# visualisation des prédictions : il faut transformer les indices de classes en couleur

couleur = torch.zeros(9,3,520,520).cuda()
couleur[:,0,:,:] = (z==1).float() # red for cat
couleur[:,1,:,:] = (z==2).float() # green for dog
couleur[:,2,:,:] = (z==3).float() # blue for person

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