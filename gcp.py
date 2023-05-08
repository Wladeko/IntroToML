import os

import cv2
import mediapipe as mp
import pyrootutils
from PIL import Image

root = pyrootutils.setup_root(
    __file__, indicator=".project-root", pythonpath=True, cwd=False
)

from pathlib import Path
from typing import Any, List

import cv2
import pytorch_lightning as pl
import requests
import torch
from google.cloud import storage
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates
from PIL import Image
from torchmetrics import MeanAbsoluteError as MAE
from torchmetrics import MeanMetric, MinMetric
from torchvision import transforms

from src.models import architectures
from src.models.face_age_module import FaceAgeModule
from src.utils.functions import inference_picture
from src.utils.predict import Predict

## Gloabl model variable
model = None


# Classes
class FaceAgeModule(pl.LightningModule):
    """
    FaceAgeModule is a PyTorch Lightning module for training a model to predict the age of a face in an image.
    It uses a pre-trained model (either SimpleConvNet_100x100, SimpleConvNet_224x224, or PretrainedEfficientNet)
    and fine-tunes it on the input dataset. The module has several methods for training, validation, and testing,
    as well as for logging metrics such as mean absolute error (MAE) and loss.
    """

    def __init__(
        self,
        net: str = "EffNet_224x224",
        rescale_age_by: int = 80.0,
        loss_fn: str = "MSELoss",
    ):
        """
        Initializes the FaceAgeModule with the specified rescale value for the labels.
        The rescale value is used to convert the predicted age value from a range of [0,1] to [1, rescale_age_by].
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters()

        # architecture
        if net == "SimpleConvNet_100x100":
            self.net = architectures.SimpleConvNet_100x100()
        elif net == "SimpleConvNet_224x224":
            self.net = architectures.SimpleConvNet_224x224()
        elif net == "EffNet_224x224":
            self.net = architectures.PretrainedEfficientNet()
        else:
            raise ValueError(f"Unknown net: {net}")

        # loss function
        if loss_fn == "MSELoss":
            self.criterion = torch.nn.MSELoss()
        elif loss_fn == "SmoothL1Loss":
            self.criterion = torch.nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unknown loss functions: {loss_fn}")

        # metric objects for calculating and averaging MAE across batches
        self.train_mae = MAE()
        self.val_mae = MAE()
        self.test_mae = MAE()

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation maeuracy
        self.val_mae_best = MinMetric()

    def forward(self, x: torch.Tensor):
        """
        The forward method is called during training, validation, and testing to make predictions on the input data.
        It takes in a tensor 'x' and returns the model's predictions.
        """
        return self.net(x)

    def predict(self, batch):
        """
        The predict method is called to make predictions on a single batch of data.
        It takes in a batch of data as input and returns the model's predictions.
        """
        x, y = batch
        preds = self.forward(x)
        preds = preds.clip(0, 1)
        return preds

    def on_train_start(self):
        """
        The on_train_start method is called before the training process begins.
        It resets the val_mae_best metric to ensure that it doesn't store any values from the validation step sanity checks.
        """
        self.val_mae_best.reset()

    def model_step(self, batch: Any):
        """
        The model_step method is called during training, validation, and testing to make predictions and calculate loss.
        It takes in a batch of input data and returns the calculated loss and predictions.
        """
        x, y = batch
        preds = self.forward(x)
        loss = self.criterion(preds, y)

        # clip prediction to [0-1]
        preds = preds.clip(0, 1)

        # rescale prediction from [0-1] to [1-80]
        if self.hparams.rescale_age_by:
            preds = preds * self.hparams.rescale_age_by
            y = y * (self.hparams.rescale_age_by - 1) + 1  # y = y*79 + 1

        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        """
        The training_step method is called during training to calculate the loss and update the model's parameters.
        It also logs the loss and mean absolute error metric to track training progress.
        """
        loss, preds, targets = self.model_step(batch)

        self.train_loss(loss)
        self.train_mae(preds, targets)
        self.log(
            "train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "train/mae", self.train_mae, on_step=False, on_epoch=True, prog_bar=True
        )

        return {"loss": loss}

    def validation_step(self, batch: Any, batch_idx: int):
        """
        The validation_step method is called during validation to calculate the loss and update metrics.
        It also logs the loss and mean absolute error metric to track validation progress.
        """
        loss, preds, targets = self.model_step(batch)

        self.val_loss(loss)
        self.val_mae(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/mae", self.val_mae, on_step=False, on_epoch=True, prog_bar=True)

    def validation_epoch_end(self, outputs: List[Any]):
        """
        The validation_epoch_end method is called at the end of each validation epoch.
        It updates the best mean absolute error metric and logs it.
        """
        self.val_mae_best(self.val_mae.compute())
        self.log("val/mae_best", self.val_mae_best.compute(), prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        """
        The test_step method is called during testing to calculate the loss and update metrics.
        It also logs the loss and mean absolute error metric to track data, and returns the calculated loss, predictions, and targets.
        """
        loss, preds, targets = self.model_step(batch)

        self.test_loss(loss)
        self.test_mae(preds, targets)
        self.log(
            "test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log("test/mae", self.test_mae, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        """
        The configure_optimizers method is used to configure the optimizers used for training.
        This method should return a single optimizer or a list of optimizers.
        In this implementation, it returns an instance of the Adam optimizer with a learning rate of 0.01.
        """
        return torch.optim.Adam(self.parameters(), lr=0.01)


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

        self.model = FaceAgeModule.load_from_checkpoint(
            ckpt_path, map_location=torch.device("cpu")
        )
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


# Prediction
def inference_picture(
    image, mp_drawing, mp_face_detection, face_detection, model, SIZE=0.1
):
    """
    Infers the age of a person in a given image using a model and a face detection library.

    Args:
        image (ndarray): The image to be processed.
        mp_drawing (obj): The object of multiperson drawing.
        mp_face_detection (obj): The object of multiperson face detection.
        face_detection (obj): The object of face detection.
        model (obj): The age inference model.
        SIZE (float): The size of the bounding box.

    Returns:
        ndarray: The image with age inferences added.
    """

    # Get the dimensions of the image
    image_rows, image_cols, _ = image.shape

    # Run face detection on the image
    results = face_detection.process(image)

    # Make the image writable
    image.flags.writeable = True

    ages = []
    # Check if any faces were detected
    if results.detections:
        # print(results.detections)
        for detection in results.detections:
            # Get the bounding box of the face
            relative_bounding_box = detection.location_data.relative_bounding_box
            rect_start_point = _normalized_to_pixel_coordinates(
                relative_bounding_box.xmin,
                relative_bounding_box.ymin,
                image_cols,
                image_rows,
            )
            rect_end_point = _normalized_to_pixel_coordinates(
                relative_bounding_box.xmin + relative_bounding_box.width,
                relative_bounding_box.ymin + relative_bounding_box.height,
                image_cols,
                image_rows,
            )

            if rect_start_point is not None and rect_end_point is not None:
                width = rect_end_point[0] - rect_start_point[0]
                height = rect_end_point[1] - rect_start_point[1]
                resized_width_params = (
                    int(rect_start_point[0] - width * SIZE),
                    int(rect_end_point[0] + width * SIZE),
                )

                resized_height_params = (
                    int(rect_start_point[1] - height * SIZE),
                    int(rect_end_point[1] + height * SIZE),
                )

                image_height, image_width, _ = image.shape
                resized_width_params = (
                    max(0, resized_width_params[0]),
                    min(image_width, resized_width_params[1]),
                )
                resized_height_params = (
                    max(0, resized_height_params[0]),
                    min(image_height, resized_height_params[1]),
                )

                # Crop the image to the bounding box
                cropped_image = image[
                    resized_height_params[0] : resized_height_params[1],
                    resized_width_params[0] : resized_width_params[1],
                ]

                # saving analysed image - used for debugging
                # cv2.imwrite("image.jpg", cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))

                # Convert the cropped image to a PIL Image
                face = Image.fromarray(cropped_image)
                face = face.convert("RGB")

                # Run the age inference model on the face
                prediction = model.predict(face)
                ages.append(prediction)

                # Add the age inference to the image
                image = cv2.putText(
                    image,
                    f"Age: {int(prediction)}",
                    (rect_start_point[0], rect_start_point[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            # Draw the detection on the image
            mp_drawing.draw_detection(image, detection)
    return image, ages


def predict(file):
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

    # output_folder = "combined_images"
    # os.makedirs(output_folder, exist_ok=True)

    with mp_face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.5
    ) as face_detection:
        model = Predict()

        import numpy as np

        image = Image.open(file)

        image = np.array(image)

        image, ages = inference_picture(
            image,
            mp_drawing,
            mp_face_detection,
            face_detection,
            model,
            SIZE=0.1,
        )

        # output_path = os.path.join(output_folder, file.replace("_input", "_response"))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # cv2.imwrite(output_path, image)
        return image, ages


# GCP related
def download_model_file():
    from google.cloud import storage

    # Model Bucket details
    BUCKET_NAME = "web-app-model"
    PROJECT_ID = "alien-striker-386011"
    GCS_MODEL_FILE = "best-checkpoint.ckpt"

    # Initialise a client
    client = storage.Client(PROJECT_ID)

    # Create a bucket object for our bucket
    bucket = client.get_bucket(BUCKET_NAME)

    # Create a blob object from the filepath
    blob = bucket.blob(GCS_MODEL_FILE)

    folder = "/tmp/"
    if not os.path.exists(folder):
        os.makedirs(folder)
    # Download the file to a destination
    blob.download_to_filename(folder + "best-checkpoint.ckpt")

    return folder + "best-checkpoint.ckpt"


def download_input_file(file):
    from google.cloud import storage

    # Model Bucket details
    BUCKET_NAME = "web-app-uploads"
    PROJECT_ID = "alien-striker-386011"
    GCS_INPUT_FILE = file

    # Initialise a client
    client = storage.Client(PROJECT_ID)

    # Create a bucket object for our bucket
    bucket = client.get_bucket(BUCKET_NAME)

    # Create a blob object from the filepath
    blob = bucket.blob(GCS_INPUT_FILE)

    folder = "/tmp/"
    if not os.path.exists(folder):
        os.makedirs(folder)
    # Download the file to a destination
    blob.download_to_filename(folder + input)


# Main entry point for the cloud function
def main(request):
    # Use the global model variable
    global model

    if not model:
        download_model_file()

    # Get the features sent for prediction
    params = request.get_json()

    if (params is not None) and ("features" in params):
        # Run a test prediction
        pred_species = model.predict(np.array([params["features"]]))
        return "Complete"

    else:
        return "Nothing sent for prediction"
