"""Общие данные: исходный текст и проверяемые цитаты."""
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
    evidence: list[Evidence] = field(default_factory=list)
    sections: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    source: str = ""


class AppError(Exception):
    """Ожидаемая ошибка с понятным сообщением пользователю."""
