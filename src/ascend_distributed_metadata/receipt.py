"""Optional durable sink for consumed ADM recovery receipts."""

from hashlib import sha256
import json
import os
from pathlib import Path


class FileReceiptSink:
    """Write one content-addressed receipt file after a routing read.

    The configured directory must already exist. A repeated write of the
    same receipt is idempotent; a partial or different existing file fails
    closed so the pending receipt is not acknowledged.
    """

    def __init__(self, directory: str):
        if not directory:
            raise ValueError("receipt_directory_required")
        self.directory = Path(directory)
        if not self.directory.is_dir():
            raise ValueError("receipt_directory_missing")

    def __call__(self, receipt: dict) -> None:
        payload = (json.dumps(receipt, sort_keys=True,
                              separators=(",", ":")) + "\n").encode()
        name = "adm-recovery-" + sha256(payload).hexdigest() + ".json"
        dir_fd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            try:
                fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                             os.O_NOFOLLOW, 0o600, dir_fd=dir_fd)
            except FileExistsError:
                if (self.directory / name).read_bytes() != payload:
                    raise ValueError("receipt_file_conflict")
                return
            with os.fdopen(fd, "wb") as output:
                output.write(payload)
                output.flush()
                os.fsync(output.fileno())
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
