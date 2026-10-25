"""Данные полного распознавания и отдельного пользовательского текста."""
from dataclasses import dataclass, field

@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str


@dataclass
class Document:
    transcript: str
    text: str = ""
    source: str = ""
    model_id: str = ""
    segments: list[Segment] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)


class AppError(Exception):
    """Ожидаемая ошибка с понятным сообщением пользователю."""
