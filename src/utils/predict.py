from torchvision import transforms
import torch

from src.models.face_age_module import FaceAgeModule

from pathlib import Path


class Predict:
    """
    This class is used for loading the trained face age model and making predictions on a given image.
    """

    def __init__(self):
        """
        Initializes the Predict class by loading the trained model, setting it to evaluation mode, and freezing its parameters.
        Also creates the image preprocessing pipeline using the torchvision library.
        """

        ckpt_path = Path("models/best-checkpoint.ckpt")
        assert ckpt_path.exists(), f"Model checkpoint not found at: '{ckpt_path}'"

        self.model = FaceAgeModule.load_from_checkpoint(ckpt_path, map_location=torch.device('cpu'))
        self.model.eval()
        self.model.freeze()
        transform_list = [
            transforms.ToTensor(),
            transforms.Resize((100, 100)),
            transforms.Resize((224, 224)),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
        self.transform = transforms.Compose(transform_list)

    def predict(self, image) -> float:
        """
        Predict the age of a face in an image using a pre-trained model.
        Args:
            image (image): An image of a face.
        Returns:
            float: The predicted age of the face in the image.
        """
        img = self.transform(image)
        img = img.reshape(1, 3, 224, 224)
        prediction = self.model.forward(img)
        prediction_rescaled = prediction * 80
        prediction_rescaled = prediction_rescaled.clip(1, 80)
        return prediction_rescaled.item()
