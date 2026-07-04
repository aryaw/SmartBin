import logging
import asyncio
from datetime import datetime, timezone

from sqlalchemy import Column, Integer, String, DateTime, Text, event, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncAttrs
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.config import DATABASE_URL

logger = logging.getLogger(__name__)

engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
    pool_recycle=3600,
)

async_session = async_sessionmaker(engine, expire_on_commit=False)


class Base(AsyncAttrs, DeclarativeBase):
    pass


class Detection(Base):
    __tablename__ = "detections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_type: Mapped[str] = mapped_column(String(16), nullable=False)
    total_objects: Mapped[int] = mapped_column(Integer, default=0)
    organik: Mapped[int] = mapped_column(Integer, default=0)
    non_organik: Mapped[int] = mapped_column(Integer, default=0)
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)
    result_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    raw_json: Mapped[dict | None] = mapped_column(JSONB, nullable=True)


async def init_db(retries: int = 3, delay: float = 1.0):
    for attempt in range(retries):
        try:
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("Database initialized")
            return
        except Exception as e:
            if attempt < retries - 1:
                logger.warning(f"DB init attempt {attempt + 1} failed: {e}, retrying in {delay}s")
                await asyncio.sleep(delay)
                delay *= 2
            else:
                logger.error(f"DB init failed after {retries} attempts: {e}")
                raise


async def save_detection(data: dict, retries: int = 2) -> int | None:
    for attempt in range(retries):
        try:
            async with async_session() as session:
                det = Detection(**data)
                session.add(det)
                await session.commit()
                return det.id
        except Exception as e:
            if attempt < retries - 1:
                logger.warning(f"DB save attempt {attempt + 1} failed: {e}, retrying")
                await asyncio.sleep(0.5)
            else:
                logger.error(f"DB save failed after {retries} attempts: {e}")
                return None


async def check_db() -> bool:
    try:
        async with async_session() as session:
            await session.execute(text("SELECT 1"))
        return True
    except Exception:
        return False