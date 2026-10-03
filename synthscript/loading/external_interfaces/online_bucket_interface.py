from __future__ import annotations

import os
import urllib.parse
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import requests
from dotenv import load_dotenv
from tqdm.auto import tqdm

from synthscript.loading.external_interfaces.external_interface import ExternalInterface
from synthscript.shared.path_bundle import PathBundle


class OnlineBucketInterface(ExternalInterface):
    """Bucket download interface. Uses paths provided by a PathBundle instance."""

    def __init__(
        self,
        bucket_url: str | None = None,
        folder: str | None = None,
        online: bool = True,
        extension_wanted: str = ".png",
        what_downloading: Literal[
            "raw_images", "background_images", "stroke_images"
        ] = "raw_images",
    ) -> None:
        if not bucket_url:
            if "BUCKET_URL" in os.environ:
                bucket_url = str(os.getenv("BUCKET_URL"))
            else:
                raise ValueError(
                    "Either a bucket_url is provided or one can be found in the "
                    "environment variables (as BUCKET_URL)."
                )
        self.bucket_url = self._normalize_bucket_url(bucket_url)
        self.folder = self._normalize_folder(folder)
        self._timeout = 15
        self._type_downloading = what_downloading

        if not extension_wanted.startswith("."):
            extension_wanted = f".{extension_wanted}"
        self.extension = extension_wanted

        self.images_url_path = self.bucket_url

    @staticmethod
    def _normalize_folder(folder: str | None) -> str | None:
        if not folder:
            return None
        clean = folder.strip().strip("/").replace("\\", "/")
        return f"{clean}/" if clean else None

    def corresponding_path_accesor(self, paths: PathBundle) -> Callable[[str], Path]:
        match self._type_downloading:
            case "raw_images":
                return paths.get_raw_image_path
            case "background_images":
                return paths.get_background_image_path
            case "stroke_images":
                return paths.get_stroke_image_path
            case _:
                raise ValueError("Unsupported type_downloading")

    @classmethod
    def from_env(
        cls,
        bucket_url: str | None = None,
        folder: str | None = None,
        env_var: str = "BUCKET_URL",
    ) -> OnlineBucketInterface:
        """Generate an instance using the environment for missing data."""
        try:
            load_dotenv()
        except Exception:
            print("Could not load the dotenv.")

        bucket_url = bucket_url if bucket_url is not None else os.getenv(env_var)
        if not bucket_url:
            raise ValueError(
                f"Did not find {env_var} in the .env or environment variables"
            )
        return cls(bucket_url=bucket_url, folder=folder)

    @staticmethod
    def _normalize_bucket_url(url: str) -> str:
        clean = url.strip().strip('"').strip("'")
        if not clean.endswith("/"):
            clean += "/"
        return clean

    def _object_url(self, object_name: str) -> str:
        # Quote path components while preserving directory separators
        quoted_name = urllib.parse.quote(object_name, safe="/")
        return urllib.parse.urljoin(self.bucket_url, quoted_name)

    def test_connection_successful(self) -> bool:
        try:
            params: dict[str, str] = {"format": "json"}
            if self.folder:
                params["prefix"] = self.folder
            resp = requests.get(self.bucket_url, params=params, timeout=self._timeout)
            resp.raise_for_status()
            print("OBI connection successful.")
            return True
        except Exception:
            print("OBI connection unsuccessful.")
            return False

    def _list_bucket_objects(self) -> list[dict]:
        objects: list[dict] = []
        start: str | None = None

        while True:
            params: dict[str, str] = {"format": "json"}
            if self.folder:
                params["prefix"] = self.folder
            if start:
                params["start"] = start

            resp = requests.get(self.bucket_url, params=params, timeout=self._timeout)
            resp.raise_for_status()
            payload = resp.json()

            page_objects = payload.get("objects", []) or []
            objects.extend(page_objects)

            start = payload.get("nextStartWith")
            if not start:
                break

        return objects

    def _compute_pending_objects(self, paths: PathBundle) -> dict[str, str]:
        objects = self._list_bucket_objects()
        pending: dict[str, str] = {}

        for obj in objects:
            raw_name = obj.get("name")
            if not raw_name:
                continue

            decoded_name = urllib.parse.unquote(str(raw_name))
            path_str = decoded_name.replace("\\", "/")

            if self.folder and not path_str.startswith(self.folder):
                continue

            p = Path(path_str)
            if p.suffix.lower() != self.extension:
                continue

            page_name = p.stem
            local_img = self.corresponding_path_accesor(paths)(page_name)
            if not local_img.exists():
                pending.setdefault(page_name, decoded_name)

        return pending

    def parts_managed(self):
        return {self._type_downloading}

    def parts_required(self):
        return set()

    def _download_image(self, paths, page_name, object_name) -> str:
        with requests.Session() as session:
            img_url = self._object_url(object_name)
            img_resp = session.get(img_url, timeout=self._timeout)
            img_resp.raise_for_status()

            local_img = self.corresponding_path_accesor(paths)(page_name)
            local_img.parent.mkdir(parents=True, exist_ok=True)
            img = cv2.imdecode(
                np.frombuffer(img_resp.content, dtype=np.uint8),
                cv2.IMREAD_GRAYSCALE,
            )
            if img is None:
                raise ValueError(f"Downloaded image {img_url!r} could not be decoded.")
            encoded, buffer = cv2.imencode(local_img.suffix, img)
            if not encoded:
                raise ValueError(f"Downloaded image {img_url!r} could not be encoded.")
            local_img.write_bytes(buffer.tobytes())
            return page_name

    def setup(self, paths: PathBundle) -> None:
        """Download pending bucket images if the interface is online."""

        pending = self._compute_pending_objects(paths)
        if not pending:
            return
        target = self.corresponding_path_accesor(paths)("*")
        print(
            f" - Downloading images into {target!s}"
        )

        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(lambda item: self._download_image(paths, *item), item)
                for item in pending.items()
            ]
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f" Downloading {self._type_downloading}...",
            ):
                future.result()
