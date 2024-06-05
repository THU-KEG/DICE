from openai import OpenAI
url = "https://chatapi.onechat.fun/v1"
api_key = "sk-ERMje880txuZZ3XTE226B89902Eb4631B26676A682698291"
client = OpenAI(base_url=url, api_key=api_key)
model_names = ["gpt-4", "gpt-4-0613", "gpt-4-1106-preview", "gpt-3.5-turbo", "gpt-3.5-turbo-0613", "gpt-3.5-turbo-1106"]
response = client.chat.completions.create(
                model=model_names[0],
                temperature=0,
                messages=[
                    {"role": "system", "content": "Assistant is a large language model trained by OpenAI."},
                    {"role": "user", "content": "Please introduce yourself in twenty words."}
                ]
            )
print(response.choices[0].message.content)