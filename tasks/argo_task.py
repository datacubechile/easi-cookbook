import logging
from json import loads
from json.decoder import JSONDecodeError

from .common import s3_upload_file

class ArgoTask:
    """
    A generic Argo task, that parses its json inputs and stores them as attributes.
    """

    # Input parameters to ignore
    IGNORE_INPUT_PARAM = (
        "package-path",
        "package-repo",
        "package-branch",
        "package-secret",
        "git_image",
    )

    def __init__(self, input_params: [{str, str}], ignore = "DEFAULT") -> None:
        """Parse and store input params as object attributes."""
        self._logger = logging.getLogger(self.__class__.__name__)

        if ignore == "DEFAULT":
            ignore = self.IGNORE_INPUT_PARAM

        for param in input_params:
            if param.get("name", "") in ignore:
                continue
            try:
                setattr(self, param["name"], loads(param["value"]))
            except JSONDecodeError:
                setattr(self, param["name"], param["value"])

    def s3_upload_file(self, path, bucket, key):
        """Upload a file to an S3 object.

        A wrapper around `common.s3_upload_file`.
        """
        s3_upload_file(file_name=path, bucket=bucket, object_name=key)
