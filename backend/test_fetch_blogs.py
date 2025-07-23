# test_fetch_blogs.py

from database import SessionLocal
from db.models import Blog

db = SessionLocal()

blogs = db.query(Blog).all()
for blog in blogs:
    print(f"Blog ID: {blog.id}")
    print(f"Topic: {blog.topic}")
    print(f"Created: {blog.created_at}")
    print(f"Markdown Preview: {blog.markdown[:200]}...\n")
    print("References:")
    for ref in blog.references:
        print(f" - {ref.title} ({ref.url})")
    print("Images:")
    for img in blog.images:
        print(f" - {img.url} (alt: {img.alt})")
    print("=" * 40)

db.close()
