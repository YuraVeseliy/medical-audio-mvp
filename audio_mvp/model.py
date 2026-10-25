"""Данные полного распознавания и отдельного пользовательского текста."""
from dataclasses import dataclass, field

SECTIONS = {"complaints": "Жалобы", "anamnesis": "Анамнез"}


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Evidence:
    section: str
    quote: str
    start: int
    end: int


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
