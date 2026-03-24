import os
import base64
import json
import time
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

image_dir = "data/test/div2k"   # 你的PNG文件夹
output_file = "data/test/div2k_caption.json"

# 断点续传：优先加载已有结果
results = {}
if os.path.exists(output_file):
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            loaded = json.load(f)
            if isinstance(loaded, dict):
                results = loaded
                print(f"Loaded existing results: {len(results)} entries")
            else:
                print(f"Warning: {output_file} is not a JSON object, start from empty results.")
    except Exception as e:
        print(f"Warning: failed to load {output_file}, start from empty results. Error: {e}")

for filename in sorted(os.listdir(image_dir)):
    if filename.lower().endswith(".png"):
        if filename in results:
            print(f"Skipping (already exists): {filename}")
            continue

        path = os.path.join(image_dir, filename)
        print(f"Processing: {filename}")

        try:
            base64_image = encode_image(path)

            completion = client.chat.completions.create(
                model="qwen3.5-plus",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant. "
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Please describe this picture in detail with 40 words. Do not provide any description about feelings."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                },
                            }
                        ],
                    }
                ],
                max_tokens=60,
                extra_body={"vl_high_resolution_images":True}  # 打开思考模式
            )

            result = completion.choices[0].message.content
            completion_tokens = completion.usage.completion_tokens
            results[filename] = result

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            token_text = completion_tokens if completion_tokens is not None else "N/A"
            print(f"[TOKENS] {filename}: completion_tokens={token_text}")
            print(f"[SAVED] {filename}: {result}")

            time.sleep(1)  # 限速，防止被限流

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            results[filename] = "ERROR"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"[SAVED] {filename}: ERROR")

# 最终再保存一次（确保完整）
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("Done!")
