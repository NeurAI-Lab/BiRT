import torchvision.transforms.functional as F
import numpy as np


class IncrementalRotation(object):
    """
    Defines an incremental rotation for a numpy array.
    """

    def __init__(self, init_deg: int = 0, increase_per_iteration: float = 0.006) -> None:
        """
        Defines the initial angle as well as the increase for each rotation
        :param init_deg:
        :param increase_per_iteration:
        """
        self.increase_per_iteration = increase_per_iteration
        self.iteration = 0
        self.degrees = init_deg

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Applies the rotation.
        :param x: image to be rotated
        :return: rotated image
        """
        degs = (self.iteration * self.increase_per_iteration + self.degrees) % 360
        self.iteration += 1
        return F.rotate(x, degs)

    def set_iteration(self, x: int) -> None:
        """
        Set the iteration to a given integer
        :param x: iteration index
        """
        self.iteration = x