import torch
from torch import nn
from torchvision.models.vgg import vgg16, vgg19
from torchvision.models.resnet import resnet101


class GeneratorLoss(nn.Module):

    def __init__(self, weight_perception=0.006, weight_adversarial=0.001,
                 weight_image=1, network="vgg16"):
        super(GeneratorLoss, self).__init__()
        self.network = network
        self.weight_image = weight_image
        self.weight_adversarial = weight_adversarial
        self.weight_perception = weight_perception
        if network == "vgg16":
            pretrained_net = vgg16(pretrained=True)
            loss_network = nn.Sequential(
                *list(pretrained_net.features)[:31]).eval()
        elif network == "vgg19":
            pretrained_net = vgg19(pretrained=True)
            loss_network = nn.Sequential(
                *list(pretrained_net.features)[:37]).eval()
        elif network == "resnet101":
            pretrained_net = resnet101(pretrained=True)
            loss_network = nn.Sequential(
                *list(pretrained_net.classifier.children())[:-1]).eval()
        elif network == "vgg16vgg19":
            pretrained_net_vgg16 = vgg16(pretrained=True)
            loss_network = nn.Sequential(
                *list(pretrained_net_vgg16.features)[:31]).eval()
            pretrained_net_vgg19 = vgg19(pretrained=True)
            loss_network_vgg19 = nn.Sequential(
                *list(pretrained_net_vgg19.features)[:27]).eval5()
            for param in loss_network_vgg19.parameters():
                param.requires_grad = False
            self.loss_network_vgg19 = loss_network_vgg19

        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()

    def forward(self, out_error, out_images, target_images):
        # Adversarial Loss
        adversarial_loss = torch.mean(1 - out_error)
        # Perception Loss
        if self.network == "vgg16vgg19":
            perception_loss = (self.mse_loss(self.loss_network(
                out_images), self.loss_network(target_images)) + self.mse_loss(self.loss_network_vgg19(
                    out_images), self.loss_network_vgg19(target_images))) / 2
        else:
            perception_loss = self.mse_loss(self.loss_network(
                out_images), self.loss_network(target_images))
        # Image Loss
        image_loss = self.mse_loss(out_images, target_images)
        return self.weight_image * image_loss + self.weight_adversarial * adversarial_loss + self.weight_perception * perception_loss


if __name__ == "__main__":
    g_loss = GeneratorLoss()
    print(g_loss)
