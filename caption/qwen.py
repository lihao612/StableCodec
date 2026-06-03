import os
import base64
import json
import io
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from openai import OpenAI

MAX_RETRIES = 3
RETRY_BASE_DELAY = 10
MAX_WORKERS = int(os.getenv("QWEN_MAX_WORKERS", "5"))
REQUEST_INTERVAL = float(os.getenv("QWEN_REQUEST_INTERVAL", "10"))
COMPRESS_QUALITY = int(os.getenv("QWEN_COMPRESS_QUALITY", "80"))

_thread_local = threading.local()

prompt_40 = """
Describe the visual content as accurately as possible to help recreate an image that closely matches the original. Do not refer to "this image" or "this picture". Use short, clear, specific phrases.

First, list the main items in order from top to bottom, left to right, describing their position, shape, and appearance.

Then, describe the details considering the following aspects:
- Feature Correspondence: Distinct features (edges, corners, textures, etc.)
- Geometric Consistency: Spatial relationships between features
- Photometric Consistency: Appearance in terms of lighting, color, and intensity
- Style Consistency: Image styles (photograph, painting, drawing, diagram, medical image)
- Semantic Consistency: Objects and their parts maintaining their identity and meaning
- Structural Integrity: Overall structure of the objects in the images

Note:
Keep your response within 40 words or fewer. If it exceeds this limit, the text will be truncated to 40 words. Feel free to shorter phrases or incomplete sentences, if it helps to include important details.
"""

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return "image/png", base64.b64encode(f.read()).decode("utf-8")


def encode_image_as_jpeg(image_path, quality=COMPRESS_QUALITY):
    with Image.open(image_path) as img:
        rgb_img = img.convert("RGB")
        buffer = io.BytesIO()
        rgb_img.save(buffer, "JPEG", optimize=True, quality=quality)
        compressed_bytes = buffer.getvalue()
    return "image/jpeg", base64.b64encode(compressed_bytes).decode("utf-8")


def get_client():
    if not hasattr(_thread_local, "client"):
        _thread_local.client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
    return _thread_local.client


def save_results(output_path, data):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def contains_chinese(text):
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def should_regenerate(existing_result):
    if not isinstance(existing_result, str):
        return True

    normalized = existing_result.strip()
    if not normalized:
        return True
    if normalized == "ERROR":
        return True
    if not normalized.endswith("."):
        return True
    # if contains_chinese(normalized):
    #     return True

    return False


def request_caption_with_retry(base64_image, image_mime_type, filename):
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            completion = get_client().chat.completions.create(
                model="qwen3-vl-plus",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt_40
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{image_mime_type};base64,{base64_image}"
                                },
                            }
                        ],
                    }
                ],
                max_tokens=77,  # 77是sd模型分词器最大输入
                extra_body={
                    "vl_high_resolution_images": True, # 不压缩输入
                    }  # 默认不打开思考模式
            )
            return completion
        except Exception as e:
            last_error = e
            if attempt == MAX_RETRIES:
                break

            delay = RETRY_BASE_DELAY * attempt
            print(
                f"[RETRY] {filename}: attempt {attempt}/{MAX_RETRIES} failed: {e}. "
                f"Retrying in {delay}s..."
            )
            time.sleep(delay)

    raise last_error


def process_image(image_dir, filename, use_compressed_image=False):
    path = os.path.join(image_dir, filename)
    if use_compressed_image:
        image_mime_type, base64_image = encode_image_as_jpeg(path)
        print(f"[COMPRESS] {filename}: sending compressed JPEG, quality={COMPRESS_QUALITY}")
    else:
        image_mime_type, base64_image = encode_image(path)

    completion = request_caption_with_retry(base64_image, image_mime_type, filename)
    result = completion.choices[0].message.content
    completion_tokens = completion.usage.completion_tokens

    if REQUEST_INTERVAL > 0:
        time.sleep(REQUEST_INTERVAL)

    return filename, result, completion_tokens


image_dir = "data/test_ori/clic2020"   # 你的PNG文件夹
output_file = "data/test_ori/clic2020_caption.json"  # 输出结果的JSON文件

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

pending_files = []
for filename in sorted(os.listdir(image_dir)):
    if not filename.lower().endswith(".png"):
        continue
    if filename in results:
        if should_regenerate(results[filename]):
            print(f"Re-generating: {filename}")
            pending_files.append((filename, False))  # 优先使用压缩图重试
        else:
            print(f"Skipping (already exists): {filename}")
        continue
    pending_files.append((filename, False))

print(f"Pending images: {len(pending_files)}")
print(f"Using max workers: {MAX_WORKERS}")

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    future_to_filename = {}
    for filename, use_compressed_image in pending_files:
        print(f"Queued: {filename}")
        future = executor.submit(process_image, image_dir, filename, use_compressed_image)
        future_to_filename[future] = filename

    for future in as_completed(future_to_filename):
        filename = future_to_filename[future]
        try:
            result_filename, result, completion_tokens = future.result()
            results[result_filename] = result
            save_results(output_file, results)

            token_text = completion_tokens if completion_tokens is not None else "N/A"
            print(f"[TOKENS] {result_filename}: completion_tokens={token_text}")
            print(f"[SAVED] {result_filename}: {result}")
        except Exception as e:
            print(f"[FAILED] {filename}: exhausted retries. Error: {e}")
            results[filename] = "ERROR"
            save_results(output_file, results)
            print(f"[SAVED] {filename}: ERROR")

# 最终再保存一次（确保完整）
save_results(output_file, results)

print("Done!")
