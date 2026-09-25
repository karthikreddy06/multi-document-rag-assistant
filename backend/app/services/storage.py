"""
Persistent File Storage Service.
Provides an interchangeable storage abstraction for local filesystem (development)
and Supabase Storage (production).
Enforces user-scoped storage path conventions: users/{user_id}/documents/{document_id}/{filename}
"""

from abc import ABC, abstractmethod
import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import urllib.parse
import httpx

from app.config import settings
from app.utils.logger import setup_logger

logger = setup_logger("services.storage")


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent directory traversal and remove illegal characters."""
    clean = os.path.basename(filename).strip()
    return clean or "document.bin"


def get_document_storage_path(user_id: str, document_id: str, filename: str) -> str:
    """Build standard user-isolated storage object path."""
    safe_name = sanitize_filename(filename)
    uid = user_id or "legacy_user"
    return f"users/{uid}/documents/{document_id}/{safe_name}"


class BaseStorageService(ABC):
    """Abstract file storage backend interface."""

    def upload_file(
        self,
        file_bytes: bytes,
        filename: str,
        user_id: str,
        document_id: str,
        content_type: Optional[str] = None,
    ) -> str:
        """Store file using the standard user-isolated convention: users/{user_id}/documents/{document_id}/{filename}."""
        path = get_document_storage_path(user_id=user_id, document_id=document_id, filename=filename)
        return self.upload(file_bytes=file_bytes, file_path=path, content_type=content_type)

    @abstractmethod
    def upload(self, file_bytes: bytes, file_path: str, content_type: Optional[str] = None) -> str:
        """Store file bytes at the specified path and return storage path."""
        pass

    @abstractmethod
    def download(self, file_path: str) -> bytes:
        """Read and return raw file bytes from storage."""
        pass

    @abstractmethod
    def get_view_url(self, file_path: str, expires_in: int = 3600) -> Optional[str]:
        """Generate a time-limited signed URL for viewing/downloading the file (if supported)."""
        pass

    @abstractmethod
    def delete(self, file_path: str) -> bool:
        """Delete file at the specified storage path."""
        pass

    @abstractmethod
    def exists(self, file_path: str) -> bool:
        """Check whether the file exists in storage."""
        pass


class LocalStorageService(BaseStorageService):
    """Local filesystem storage backend for development and test environments."""

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = (base_dir or settings.upload_abs_path).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_path(self, file_path: str) -> Path:
        resolved = (self.base_dir / file_path).resolve()
        if not str(resolved).startswith(str(self.base_dir)):
            raise ValueError(f"Directory traversal detected for path: {file_path}")
        return resolved

    def upload(self, file_bytes: bytes, file_path: str, content_type: Optional[str] = None) -> str:
        target = self._resolve_path(file_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(file_bytes)
        logger.info(f"LocalStorage: Saved {len(file_bytes)} bytes to '{target}'")
        return file_path

    def download(self, file_path: str) -> bytes:
        target = self._resolve_path(file_path)
        if not target.exists():
            raise FileNotFoundError(f"File not found on local disk: '{target}'")
        return target.read_bytes()

    def get_view_url(self, file_path: str, expires_in: int = 3600) -> Optional[str]:
        # Local filesystem does not emit signed cloud URLs; backend serves inline via API
        return None

    def delete(self, file_path: str) -> bool:
        try:
            target = self._resolve_path(file_path)
            if target.exists():
                target.unlink()
                logger.info(f"LocalStorage: Deleted '{target}'")
                return True
            return False
        except Exception as e:
            logger.warning(f"LocalStorage: Error deleting '{file_path}': {e}")
            return False

    def exists(self, file_path: str) -> bool:
        try:
            target = self._resolve_path(file_path)
            return target.exists() and target.is_file()
        except Exception:
            return False


class SupabaseStorageService(BaseStorageService):
    """
    Cloud persistent file storage backend using Supabase Storage REST API.
    Uses secret server-side service-role key for authenticated uploads and signed URL generation.
    """

    def __init__(
        self,
        supabase_url: Optional[str] = None,
        service_role_key: Optional[str] = None,
        bucket: Optional[str] = None,
    ):
        self.supabase_url = (supabase_url or settings.supabase_url).rstrip("/")
        self.service_role_key = (service_role_key or settings.supabase_service_role_key or "").strip()
        self.bucket = bucket or settings.supabase_storage_bucket or "rag-files"
        self._bucket_verified = False

    def _get_headers(self, content_type: Optional[str] = None) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.service_role_key}",
            "apikey": self.service_role_key,
        }
        if content_type:
            headers["Content-Type"] = content_type
        return headers

    def _ensure_bucket(self) -> None:
        """Verify that the private storage bucket exists or create it idempotently."""
        if self._bucket_verified:
            return
        if not self.service_role_key:
            return

        try:
            url = f"{self.supabase_url}/storage/v1/bucket/{self.bucket}"
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(url, headers=self._get_headers())
                if resp.status_code == 200:
                    self._bucket_verified = True
                    return

                # Create private bucket if not found
                create_url = f"{self.supabase_url}/storage/v1/bucket"
                payload = {"id": self.bucket, "name": self.bucket, "public": False}
                c_resp = client.post(create_url, headers=self._get_headers("application/json"), json=payload)
                if c_resp.status_code in (200, 201):
                    logger.info(f"SupabaseStorage: Created private bucket '{self.bucket}'.")
                self._bucket_verified = True
        except Exception as e:
            logger.warning(f"SupabaseStorage: Note while verifying bucket '{self.bucket}': {e}")
            self._bucket_verified = True

    def upload(self, file_bytes: bytes, file_path: str, content_type: Optional[str] = None) -> str:
        self._ensure_bucket()
        clean_path = file_path.lstrip("/")
        mime = content_type or mimetypes.guess_type(clean_path)[0] or "application/octet-stream"

        url = f"{self.supabase_url}/storage/v1/object/{self.bucket}/{clean_path}"
        headers = self._get_headers(mime)
        headers["x-upsert"] = "true"

        with httpx.Client(timeout=30.0) as client:
            resp = client.post(url, headers=headers, content=file_bytes)
            if resp.status_code not in (200, 201):
                # Try PUT if POST returned 400 for existing object
                resp = client.put(url, headers=headers, content=file_bytes)
                if resp.status_code not in (200, 201):
                    logger.error(f"SupabaseStorage upload failed ({resp.status_code}): {resp.text}")
                    raise RuntimeError(f"SupabaseStorage upload failed for '{clean_path}': {resp.status_code}")

        logger.info(f"SupabaseStorage: Uploaded {len(file_bytes)} bytes to '{self.bucket}/{clean_path}'")
        return clean_path

    def download(self, file_path: str) -> bytes:
        clean_path = file_path.lstrip("/")
        url = f"{self.supabase_url}/storage/v1/object/authenticated/{self.bucket}/{clean_path}"
        headers = self._get_headers()

        with httpx.Client(timeout=30.0) as client:
            resp = client.get(url, headers=headers)
            if resp.status_code == 200:
                return resp.content
            # Fallback to direct object endpoint
            fallback_url = f"{self.supabase_url}/storage/v1/object/{self.bucket}/{clean_path}"
            resp = client.get(fallback_url, headers=headers)
            if resp.status_code == 200:
                return resp.content

        raise FileNotFoundError(f"SupabaseStorage: Object not found: '{self.bucket}/{clean_path}'")

    def get_view_url(self, file_path: str, expires_in: int = 3600) -> Optional[str]:
        """Generate time-limited signed URL for authenticated inline view or download."""
        clean_path = file_path.lstrip("/")
        url = f"{self.supabase_url}/storage/v1/object/sign/{self.bucket}/{clean_path}"
        headers = self._get_headers("application/json")
        payload = {"expiresIn": expires_in}

        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.post(url, headers=headers, json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    signed_url_path = data.get("signedURL") or data.get("signedUrl")
                    if signed_url_path:
                        if signed_url_path.startswith("http"):
                            return signed_url_path
                        return f"{self.supabase_url}/storage/v1{signed_url_path}"
        except Exception as e:
            logger.warning(f"SupabaseStorage: Error generating signed URL for '{clean_path}': {e}")
        return None

    def delete(self, file_path: str) -> bool:
        clean_path = file_path.lstrip("/")
        url = f"{self.supabase_url}/storage/v1/object/{self.bucket}/{clean_path}"
        headers = self._get_headers()

        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.delete(url, headers=headers)
                return resp.status_code in (200, 204)
        except Exception as e:
            logger.warning(f"SupabaseStorage: Error deleting '{clean_path}': {e}")
            return False

    def exists(self, file_path: str) -> bool:
        clean_path = file_path.lstrip("/")
        url = f"{self.supabase_url}/storage/v1/object/authenticated/{self.bucket}/{clean_path}"
        headers = self._get_headers()
        headers["Range"] = "bytes=0-0"

        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(url, headers=headers)
                return resp.status_code in (200, 206)
        except Exception:
            return False


# Singleton instance
_storage_service: Optional[BaseStorageService] = None


def get_storage_service() -> BaseStorageService:
    """Return configured storage service (Supabase Storage in production or LocalStorage fallback)."""
    global _storage_service
    if _storage_service is None:
        if settings.is_supabase_storage:
            logger.info("StorageService: Initializing Supabase Storage backend.")
            _storage_service = SupabaseStorageService()
        else:
            logger.info("StorageService: Initializing LocalStorage backend (fallback).")
            _storage_service = LocalStorageService()
    return _storage_service


def reset_storage_service() -> None:
    """Reset storage service singleton (used for tests and configuration reloads)."""
    global _storage_service
    _storage_service = None
