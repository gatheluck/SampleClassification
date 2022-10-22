import base64
import http
import json
import pathlib
import sys
from io import BytesIO
from typing import Any, Dict, TypedDict

if sys.version_info >= (3, 8):
    from typing import Final
else:
    from typing_extensions import Final

from src.ml.data.cifar10 import Cifar10DataModule
from src.ml.models.classifier import (
    ClassificationPredictor,
    ClassificationPredictorInputDto,
    Classifier,
)


class _Response(TypedDict):
    statusCode: int
    headers: Dict[str, str]
    body: str


def _response_json(
    http_status: http.HTTPStatus,
    body: Dict,
) -> _Response:
    return {
        "statusCode": int(http_status),
        "headers": {
            "Content-Type": "application/json",
        },
        "body": json.dumps(body),
    }


def predict_handler(event: Dict, context: Any) -> _Response:

    try:
        binary_image: Final = base64.b64decode(event["body"])
    except Exception:
        raise ValueError()

    # Set up predictor
    model_path: Final = pathlib.Path(
        "sample_classification/models/sample-checkpoint.ckpt"
    )
    labels: Final = Cifar10DataModule.get_labels()
    classifier: Final = Classifier.load_from_checkpoint(str(model_path))
    predictor: Final = ClassificationPredictor(classifier, labels)

    transform: Final = Cifar10DataModule.get_transform(train=False, to_tensor=True)
    topk: Final = 3

    # Run prediction
    input_dto = ClassificationPredictorInputDto.from_images(
        [BytesIO(binary_image)], transform, topk
    )
    output_dto: Final = predictor.predict(input_dto)

    return _response_json(http.HTTPStatus.OK, output_dto.to_json())
