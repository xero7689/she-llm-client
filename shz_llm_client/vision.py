import base64
from io import BytesIO

import boto3
from PIL import Image


def image_to_base64(image_path):
    """
    Convert an image to Base64 format.

    :param image_path: Path to the image file
    :return: Base64 encoded string of the image
    """
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_string


def s3_image_to_base64(bucket_name, object_key):
    """
    Convert an image from S3 to Base64 format.

    :param bucket_name: The name of the S3 bucket
    :param object_key: The key of the image object in the S3 bucket
    :return: Base64 encoded string of the image
    """
    # Initialize the S3 client
    s3 = boto3.client("s3")

    # Get the object from S3
    response = s3.get_object(Bucket=bucket_name, Key=object_key)

    # Read the object's content
    image_data = response["Body"].read()

    # Encode the image to Base64
    encoded_string = base64.b64encode(image_data).decode("utf-8")

    return encoded_string


def resize_image(file, max_short_side=768, max_long_side=1568) -> BytesIO:
    """Resize the image if it exceeds the maximum short side or long side

    This function is for the limitation of OpenAI's vision support.
    The maximum short side is 768 and the maximum long side is 2000.
    """
    # Open the image
    image = Image.open(file)
    width, height = image.size

    # Determine the scaling factor
    if width < height:
        short_side, long_side = width, height
    else:
        short_side, long_side = height, width

    # Check if the image needs resizing
    if short_side > max_short_side or long_side > max_long_side:
        # Determine new dimensions
        if short_side > max_short_side:
            scale_factor = max_short_side / short_side
        else:
            scale_factor = max_long_side / long_side

        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)

        # Resize the image
        image = image.resize((new_width, new_height))

    # Save the image to a BytesIO object
    image_io = BytesIO()
    image_format = file.name.split(".")[-1].upper()
    if image_format == "JPG":
        image_format = "JPEG"
    image.save(image_io, format=image_format)
    image_io.seek(0)

    return image_io
