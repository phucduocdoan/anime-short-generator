from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from anime_short.config import YouTubeConfig
from anime_short.youtube.auth import get_credentials
from anime_short.youtube.metadata import VideoMetadata


@dataclass(frozen=True)
class YouTubeUploadResult:
    video_id: str
    url: str
    privacy_status: str
    publish_at: datetime | None


class YouTubeService:
    def __init__(self, config: YouTubeConfig, headless: bool = False, oauth_port: int = 8080):
        from googleapiclient.discovery import build

        self.config = config
        creds = get_credentials(
            token_file=config.token_file,
            client_secret_file=config.client_secret_file,
            headless=headless,
            port=oauth_port,
        )
        self._youtube = build("youtube", "v3", credentials=creds)

    def upload(
        self,
        video_path: Path,
        metadata: VideoMetadata,
        privacy_status: str,
        publish_at: datetime | None = None,
    ) -> YouTubeUploadResult:
        from googleapiclient.http import MediaFileUpload

        if privacy_status not in {"private", "unlisted", "public"}:
            raise ValueError("privacy_status must be private, unlisted, or public")
        if not video_path.exists():
            raise FileNotFoundError(video_path)

        status_body: dict[str, object] = {
            "privacyStatus": privacy_status,
            "selfDeclaredMadeForKids": self.config.self_declared_made_for_kids,
            "containsSyntheticMedia": self.config.contains_synthetic_media,
        }
        scheduled_at = None
        if publish_at is not None:
            scheduled_at = publish_at.astimezone(UTC)
            status_body["privacyStatus"] = "private"
            status_body["publishAt"] = scheduled_at.strftime("%Y-%m-%dT%H:%M:%S.000Z")

        snippet: dict[str, object] = {
            "title": metadata.title,
            "description": metadata.description,
            "categoryId": self.config.category_id,
        }
        if metadata.tags:
            snippet["tags"] = metadata.tags

        request = self._youtube.videos().insert(
            part="snippet,status",
            body={"snippet": snippet, "status": status_body},
            media_body=MediaFileUpload(
                str(video_path),
                mimetype="video/mp4",
                resumable=True,
                chunksize=10 * 1024 * 1024,
            ),
            notifySubscribers=self.config.notify_subscribers,
        )

        response = None
        while response is None:
            _, response = request.next_chunk()

        video_id = response["id"]
        return YouTubeUploadResult(
            video_id=video_id,
            url=f"https://youtu.be/{video_id}",
            privacy_status=str(status_body["privacyStatus"]),
            publish_at=scheduled_at,
        )
