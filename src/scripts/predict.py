import pathlib
import sys

if sys.version_info >= (3, 8):
    from typing import Final
else:
    from typing_extensions import Final

from src.ml.data.cifar10 import Cifar10DataModule
from src.ml.models.classifier import (
    ClassificationPredictor,
    ClassificationPredictorInputDto,
    ClassificationPredictorOutputDto,
    Classifier,
)


def predict(
    model_path: pathlib.Path,
    image_path: pathlib.Path,
    topk: int = 3,
    use_gpu: bool = False,
) -> ClassificationPredictorOutputDto:

    # Set up predictor
    labels: Final = Cifar10DataModule.get_labels()
    classifier: Final = Classifier.load_from_checkpoint(str(model_path))
    if use_gpu:
        classifier.to("cuda")
    predictor: Final = ClassificationPredictor(classifier, labels)

    # Preprocess image
    transform: Final = Cifar10DataModule.get_transform(train=False, to_tensor=True)
    input_dto = ClassificationPredictorInputDto.from_image_files(
        [image_path], transform, topk
    )
    if use_gpu:
        input_dto = input_dto.to_cuda()

    return predictor.predict(input_dto)


if __name__ == "__main__":
    model_path: Final = pathlib.Path("models/sample-checkpoint.ckpt")
    image_path: Final = pathlib.Path("tests/assets/sample_input_image.png")

    import argparse

    parser: Final = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(model_path),
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default=str(image_path),
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=3,
    )
    parser.add_argument("--use_gpu", action="store_true")
    args: Final = parser.parse_args()

    output_dto: Final = predict(
        args.model_path, args.image_path, args.topk, args.use_gpu
    )
    print(output_dto.to_json())
