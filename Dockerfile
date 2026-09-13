# ใช้ Python 3.9 เป็นพื้นฐาน
FROM python:3.9-slim

# ติดตั้ง FFmpeg ที่จำเป็นสำหรับจัดการไฟล์เสียง .m4a / .mp4
RUN apt-get update && \
    apt-get install -y ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# กำหนดพื้นที่ทำงานใน Container
WORKDIR /app

# ก๊อปปี้ไฟล์ requirements และติดตั้ง
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ก๊อปปี้ไฟล์ทั้งหมด (app.py, index.html) เข้าไปใน Container
COPY . .

# เปิด Port 8000
EXPOSE 8000

# คำสั่งรันเซิร์ฟเวอร์ FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
