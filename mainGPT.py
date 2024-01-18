import torch
import torchvision
import numpy as np

# # Définir la fraction de mémoire à allouer (par exemple, 0.9 pour 90%)
# memory_fraction = 0.8

# # Allouer la mémoire sur le GPU avec la fraction spécifiée
# torch.cuda.set_per_process_memory_fraction(memory_fraction)

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


# Charger le modèle DeepLabV3_ResNet50 pré-entraîné
W = torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
net = torchvision.models.segmentation.deeplabv3_resnet50(weights=W).cuda()

# Charger les images sur le GPU
im1, im2, im3, im4, im5, im6 = [im.cuda() for im in [im1, im2, im3, im4, im5, im6]]

# Transformer les images
x = torch.stack([im1, im2, im3, im4, im5, im6], dim=0).cuda()
x = (W.transforms())(x)

# Obtenir les prédictions du modèle
classes = [0, 8, 12, 15,     16, 3, 2, 4, 5]
with torch.no_grad():
    x = x.cuda()
    z = net.cuda()(x)["out"]  # prédiction des cartes de score de confiance
    _, l = z[:, classes, :, :].max(1)

# Définir la liste des mauvaises prédictions
perm = torch.randperm(len(classes)).cuda()
l_prime = perm[l]


# Définir la fonction de perte
class AdversarialLoss(torch.nn.Module):
    def __init__(self):
        super(AdversarialLoss, self).__init__()

    def forward(self, z, condition, l, l_prime):
        return torch.sum(z[condition, l[condition]] - z[condition, l_prime[condition]])

# Paramètres
gamma = 0.5
maxIter = 200

# Initialisations
xm = x.clone().requires_grad_()
r = torch.zeros_like(x)


del x
net.zero_grad()
torch.cuda.empty_cache()
# Boucle d'optimisation
print("début boucle")
for m in range(maxIter):
    xm.cuda()
    zm = net(xm)["out"][:, classes, :, :]
    _, lm = zm.max(1)

    condition = (lm == l)
    if not condition.any():  # Tous les pixels sont erronés
        break

    print("a")
    loss = AdversarialLoss()(zm, condition, l, l_prime)
    print("b")
    loss.backward()
    print("c")
    rm = -(gamma / xm.grad.norm()) * xm.grad
    r = r+rm
    xm = xm + rm
    xm.grad.zero_()

    del zm
    net.zero_grad()
    torch.cuda.empty_cache()

    print(m + 1)

print(r[:10,:10,:10])
