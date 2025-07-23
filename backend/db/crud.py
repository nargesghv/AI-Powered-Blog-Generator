from sqlalchemy.orm import Session
from . import models, schemas

def create_blog(db: Session, blog: schemas.BlogCreate):
    db_blog = models.Blog(topic=blog.topic, markdown=blog.markdown)
    db.add(db_blog)
    db.commit()
    db.refresh(db_blog)

    for ref in blog.references:
        db_ref = models.Reference(**ref.dict(), blog_id=db_blog.id)
        db.add(db_ref)

    for img in blog.images:
        db_img = models.Image(**img.dict(), blog_id=db_blog.id)
        db.add(db_img)

    db.commit()
    db.refresh(db_blog)
    return db_blog

def get_blog(db: Session, blog_id: int):
    return db.query(models.Blog).filter(models.Blog.id == blog_id).first()

def list_blogs(db: Session, skip: int = 0, limit: int = 10):
    return db.query(models.Blog).offset(skip).limit(limit).all()
