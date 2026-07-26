"""
d2lcore/GAN - Generative Adversarial Network training functions.

d2lcore functions:
  - update_D(X, Z, net_D, net_G, loss, trainer_D)
  - update_G(Z, net_D, net_G, loss, trainer_G)
"""

import torch

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


class CdlUpdateD:
    """Update discriminator in GAN training.

    d2lcore: update_D(X, Z, net_D, net_G, loss, trainer_D)
    Returns the discriminator loss value.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "X": ("cdlTensor",),
                "Z": ("cdlTensor",),
                "net_D": ("cdlModel",),
                "net_G": ("cdlModel",),
            },
            "hidden": {
                "prompt": "PROMPT",
            }
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("loss_D",)
    FUNCTION = "execute"
    CATEGORY = "ComfyDL/GAN"

    def execute(self, prompt=None, X=None, Z=None, net_D=None, net_G=None):
        if X is None or Z is None or net_D is None or net_G is None:
            return (0.0,)

        batch_size = X.shape[0]
        ones = torch.ones((batch_size,), device=X.device)
        zeros = torch.zeros((batch_size,), device=X.device)
        loss_fn = torch.nn.BCEWithLogitsLoss()

        # Simple SGD trainer
        trainer_D = torch.optim.SGD(net_D.parameters(), lr=0.01)

        trainer_D.zero_grad()
        real_Y = net_D(X)
        fake_X = net_G(Z)
        fake_Y = net_D(fake_X.detach())
        loss_D = (loss_fn(real_Y, ones.reshape(real_Y.shape)) +
                   loss_fn(fake_Y, zeros.reshape(fake_Y.shape))) / 2
        loss_D.backward()
        trainer_D.step()
        return (float(loss_D.item()),)


NODE_CLASS_MAPPINGS["CdlUpdateD"] = CdlUpdateD
NODE_DISPLAY_NAME_MAPPINGS["CdlUpdateD"] = "Update Discriminator"


class CdlUpdateG:
    """Update generator in GAN training.

    d2lcore: update_G(Z, net_D, net_G, loss, trainer_G)
    Returns the generator loss value.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "Z": ("cdlTensor",),
                "net_D": ("cdlModel",),
                "net_G": ("cdlModel",),
            },
            "hidden": {
                "prompt": "PROMPT",
            }
        }

    RETURN_TYPES = ("FLOAT", "cdlTensor")
    RETURN_NAMES = ("loss_G", "fake_X")
    FUNCTION = "execute"
    CATEGORY = "ComfyDL/GAN"

    def execute(self, prompt=None, Z=None, net_D=None, net_G=None):
        if Z is None or net_D is None or net_G is None:
            return (0.0, torch.zeros(1))

        batch_size = Z.shape[0]
        ones = torch.ones((batch_size,), device=Z.device)
        loss_fn = torch.nn.BCEWithLogitsLoss()

        trainer_G = torch.optim.SGD(net_G.parameters(), lr=0.01)

        trainer_G.zero_grad()
        fake_X = net_G(Z)
        fake_Y = net_D(fake_X)
        loss_G = loss_fn(fake_Y, ones.reshape(fake_Y.shape))
        loss_G.backward()
        trainer_G.step()
        return (float(loss_G.item()), fake_X.detach())


NODE_CLASS_MAPPINGS["CdlUpdateG"] = CdlUpdateG
NODE_DISPLAY_NAME_MAPPINGS["CdlUpdateG"] = "Update Generator"
