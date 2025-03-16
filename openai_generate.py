import instructor
from pydantic import BaseModel
from openai import OpenAI


# Define your desired output structure
class UserInfo(BaseModel):
    sentence: str
    emoji: str


# Patch the OpenAI client
client = instructor.from_openai(OpenAI())

# Extract structured data from natural language
user_info = client.chat.completions.create(
    model="gpt-4o-mini",
    response_model=UserInfo,
    messages=[
        {
            "role": "user",
            "content": "Give me a sentece and write its emoji sentence counterpart.",
        }
    ],
)

print(user_info.sentence)
# > John Doe
print(user_info.emoji)
# > 30
