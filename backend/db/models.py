from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime
from database import Base

class Blog(Base):
    __tablename__ = "blogs"
    __table_args__ = {'extend_existing': True}

    id = Column(Integer, primary_key=True, index=True)
    topic = Column(String, nullable=False)
    markdown = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    references = relationship("db.models.Reference", back_populates="blog", cascade="all, delete")
    images = relationship("db.models.Image", back_populates="blog", cascade="all, delete")

class Reference(Base):
    __tablename__ = "references"
    __table_args__ = {'extend_existing': True}

    id = Column(Integer, primary_key=True, index=True)
    blog_id = Column(Integer, ForeignKey("blogs.id"))
    title = Column(String)
    url = Column(String)

    blog = relationship("db.models.Blog", back_populates="references")

class Image(Base):
    __tablename__ = "images"
    __table_args__ = {'extend_existing': True}

    id = Column(Integer, primary_key=True, index=True)
    blog_id = Column(Integer, ForeignKey("blogs.id"))
    url = Column(String)
    alt = Column(String)
    license = Column(String)

    blog = relationship("db.models.Blog", back_populates="images")
