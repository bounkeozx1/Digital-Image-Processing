import mimetypes
import os
import google.generativeai as genai

def generate(image_path):
    # 1. ตั้งค่า API Key
    api_key = "AIzaSyDxiKISlz75wpVijQVVWIeU5e4Z44N42Cs"
    
    genai.configure(api_key=api_key)

    if not os.path.exists(image_path):
        print(f"Error: ไม่พบไฟล์ที่ {image_path}")
        return

    # 2. เลือกโมเดล
    model = genai.GenerativeModel("gemini-2.0-flash")

    # 4. เตรียม Content
    with open(image_path, "rb") as f:
        image_data = {
            "mime_type": mimetypes.guess_type(image_path)[0] or "image/jpeg",
            "data": f.read()
        }

    print("--- กำลังวิเคราะห์รูปภาพด้วย Gemini ---")

    try:
        # 5. เรียกใช้ generate_content
        response = model.generate_content([
            "Detect traffic signs, car signs, and license plates. Extract all text and describe them.",
            image_data
        ])
        print(response.text)
    except Exception as e:
        print(f"เกิดข้อผิดพลาด: {e}")

    print("--------------------------")

if __name__ == "__main__":
    target_image = "Python/girl.jpg"
    generate(target_image)