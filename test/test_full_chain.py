import os
from dotenv import load_dotenv
from agents.blog_agents import blog_chain

load_dotenv()

# Step 1: Initial state
initial_state = {
    "topic": "How AI is Transforming Renewable Energy"
}

# Step 2: Run the agent pipeline
result = blog_chain.invoke(initial_state)

# Step 3: Check if final_post exists
final_post = result.get("final_post")
if not final_post:
    print("❌ Error: final_post not found in result.")
    exit()

# Step 4: Ensure directory exists
output_dir = r"C:/Users/nariv/Desktop/Git/A blog writter by Multi_agent AI system/blog_data"
os.makedirs(output_dir, exist_ok=True)

# Step 5: Save file
output_path = os.path.join(output_dir, "final_blog.md")
with open(output_path, "w", encoding="utf-8") as f:
    f.write(final_post)

print(f"✅ Blog saved to: {output_path}")

# Step 6: Print blog
print("\n✅ Final Blog Post:")
print("-" * 80)
print(final_post)