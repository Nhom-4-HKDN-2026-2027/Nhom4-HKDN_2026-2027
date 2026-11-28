import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from pydantic import BaseModel, Field


def compute_target_time(base_time_str: Any) -> str | None:
    """Tính toán mốc thời gian dự báo đích (target_time = event_time + 1h)."""
    if not base_time_str:
        return None
    try:
        clean_str = str(base_time_str).replace("Z", "+00:00")
        dt = datetime.fromisoformat(clean_str)
        target_dt = dt + timedelta(hours=1)
        if target_dt.tzinfo:
            return target_dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        return target_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return None


class PredictionRequest(BaseModel):
    model_name: str | None = Field(default=None, description="Tên mô hình (tùy chọn)")
    timestamp: str | None = Field(default=None, description="Thời điểm gửi dự báo (tùy chọn)")
    features: list[dict[str, Any]] | list[list[float]] = Field(
        ..., description="Dữ liệu quan trắc đầu vào để dự báo nồng độ PM2.5."
    )


class PredictionResponse(BaseModel):
    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="Mã định danh lượt gọi API"
    )
    status: str = Field(default="success", description="Trạng thái xử lý")
    model_name: str
    model_version: str = Field(
        default="1", description="Phiên bản mô hình đang phục vụ trong Registry (VD: '1', '2')"
    )
    model_alias: str | None = None
    target_time: str | list[str] | None = Field(
        default=None, description="Thời điểm đích của dự báo PM2.5 (event_time + 1h)"
    )
    predictions: list[float] = Field(..., description="Nồng độ PM2.5 dự báo (µg/m³)")
