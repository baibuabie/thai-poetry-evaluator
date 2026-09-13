import os
import shutil
import warnings
import uuid
import traceback
import subprocess
import difflib
import sqlite3
import json
warnings.filterwarnings("ignore")

import librosa
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from transformers import pipeline
import jiwer
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import soundfile as sf

try:
    import imageio_ffmpeg
    FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    FFMPEG_PATH = None

plt.rcParams['font.family'] = 'Tahoma'
plt.rcParams['axes.unicode_minus'] = False

app = FastAPI(title="Thai Poetry Classroom & Evaluator")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(BASE_DIR, "static"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "uploads"), exist_ok=True)
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
app.mount("/uploads", StaticFiles(directory=os.path.join(BASE_DIR, "uploads")), name="uploads")

print("🚀 Loading Whisper ASR Model...")
device = 0 if torch.cuda.is_available() else -1
# ใช้โมเดล tiny เพื่อลดการกินทรัพยากร
transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-tiny", device=device)

# ==========================================
# 🗄️ Database Setup (SQLite)
# ==========================================
DB_PATH = os.path.join(BASE_DIR, "database.db")

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    c = conn.cursor()
    # ตารางผู้ใช้งาน
    c.execute('''CREATE TABLE IF NOT EXISTS users (email TEXT PRIMARY KEY, role TEXT, name TEXT)''')
    # ตารางชั้นเรียน
    c.execute('''CREATE TABLE IF NOT EXISTS classes (id TEXT PRIMARY KEY, name TEXT, subject TEXT, theme_color TEXT)''')
    # ตารางจับคู่ผู้ใช้กับชั้นเรียน
    c.execute('''CREATE TABLE IF NOT EXISTS user_classes (email TEXT, class_id TEXT)''')
    # ตารางการบ้าน
    c.execute('''CREATE TABLE IF NOT EXISTS assignments (
        id TEXT PRIMARY KEY, class_id TEXT, title TEXT, description TEXT, poem_text TEXT, 
        w_text INTEGER, w_pitch INTEGER, w_rhythm INTEGER, tolerance REAL, ref_audio_url TEXT
    )''')
    # ตารางส่งงานและผลคะแนน
    c.execute('''CREATE TABLE IF NOT EXISTS submissions (
        id TEXT PRIMARY KEY, assignment_id TEXT, student_email TEXT, student_name TEXT, 
        video_url TEXT, raw_file_path TEXT, status TEXT, teacher_note TEXT,
        total_score REAL, text_score REAL, pitch_score REAL, rhythm_score REAL,
        transcribed_text TEXT, pitch_analysis TEXT, rhythm_analysis TEXT,
        chart_pitch TEXT, chart_rhythm TEXT, eval_ref_audio_url TEXT,
        rubric_feedback TEXT, aligned_tokens TEXT
    )''')

    # จำลองบัญชีผู้ใช้งานเริ่มต้น (Seeding)
    c.execute("INSERT OR IGNORE INTO users (email, role, name) VALUES ('teacher@gmail.com', 'teacher', 'อาจารย์วิชาการ (Teacher)')")
    c.execute("INSERT OR IGNORE INTO users (email, role, name) VALUES ('student@gmail.com', 'student', 'นายสมชาย ใจดี (Student)')")
    
    c.execute("INSERT OR IGNORE INTO classes (id, name, subject, theme_color) VALUES ('cls-1', 'ม.5/1', 'ภาษาไทยพื้นฐาน (ท32101)', 'from-emerald-600 to-teal-700')")
    c.execute("INSERT OR IGNORE INTO classes (id, name, subject, theme_color) VALUES ('cls-2', 'ม.5/2', 'วรรณคดีวิจักษ์ (ท32201)', 'from-indigo-600 to-blue-700')")
    
    # จับคู่ห้องเรียน
    c.execute("SELECT COUNT(*) FROM user_classes")
    if c.fetchone()[0] == 0:
        c.execute("INSERT INTO user_classes (email, class_id) VALUES ('teacher@gmail.com', 'cls-1')")
        c.execute("INSERT INTO user_classes (email, class_id) VALUES ('teacher@gmail.com', 'cls-2')")
        c.execute("INSERT INTO user_classes (email, class_id) VALUES ('student@gmail.com', 'cls-2')")

    conn.commit()
    conn.close()

init_db()

# ==========================================
# 🧠 Audio & AI Core Logic (คงไว้เหมือนเดิม 100%)
# ==========================================
def convert_to_wav(source_path: str) -> str:
    target_path = os.path.splitext(source_path)[0] + "_converted.wav"
    if FFMPEG_PATH:
        cmd = [FFMPEG_PATH, "-y", "-i", source_path, "-vn", "-acodec", "pcm_s16le", "-ar", "22050", "-ac", "1", target_path]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode == 0 and os.path.exists(target_path):
            return target_path
    y, sr = librosa.load(source_path, sr=22050)
    sf.write(target_path, y, sr)
    return target_path

def evaluate_text_accuracy(audio_path: str, reference_text: str):
    result = transcriber(audio_path, return_timestamps=True, generate_kwargs={"language": "thai", "task": "transcribe"})
    transcribed_text = result["text"].replace(" ", "")
    clean_ref = reference_text.replace(" ", "")
    cer = jiwer.cer(clean_ref, transcribed_text) if len(clean_ref) > 0 else 1.0
    text_score = max(0.0, (1.0 - cer) * 100)

    matcher = difflib.SequenceMatcher(None, clean_ref, transcribed_text)
    aligned_tokens = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        sub_ref = clean_ref[i1:i2]
        sub_hyp = transcribed_text[j1:j2]
        if tag == 'equal':
            for ch in sub_ref:
                aligned_tokens.append({"char": ch, "status": "correct", "tooltip": "ออกเสียงถูกต้อง"})
        elif tag == 'replace':
            for idx, ch in enumerate(sub_ref):
                expected = ch
                got = sub_hyp[idx] if idx < len(sub_hyp) else "-"
                aligned_tokens.append({"char": ch, "status": "error_pronounce", "tooltip": f"เสียงเพี้ยน: ต้นฉบับ '{expected}' แต่ออกเสียงเป็น '{got}'"})
        elif tag == 'delete':
            for ch in sub_ref:
                aligned_tokens.append({"char": ch, "status": "error_omission", "tooltip": f"อ่านตกหล่น: ไม่ได้ออกเสียง '{ch}'"})

    return float(text_score), transcribed_text, aligned_tokens

def extract_pitch_contour(audio_path: str, sr=22050):
    y, _ = librosa.load(audio_path, sr=sr)
    f0, voiced_flag, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C6'), sr=sr)
    valid_f0 = f0[voiced_flag] if voiced_flag is not None else np.array([])
    if len(valid_f0) < 5: return np.zeros(50)
    midi_pitch = librosa.hz_to_midi(valid_f0)
    return midi_pitch - np.median(midi_pitch)

def evaluate_pitch_similarity(ref_audio: str, stu_audio: str, pitch_tolerance: float = 35.0):
    p_ref = extract_pitch_contour(ref_audio)
    p_stu = extract_pitch_contour(stu_audio)
    if len(p_ref) == 0: p_ref = np.zeros(50)
    if len(p_stu) == 0: p_stu = np.zeros(50)

    dist, path = fastdtw(p_ref, p_stu, dist=lambda a, b: abs(a - b))
    norm_dist = dist / (len(p_ref) + len(p_stu))
    score = max(0.0, 100.0 - (norm_dist * pitch_tolerance))

    diffs = [abs(p_ref[i] - p_stu[j]) for (i, j) in path]
    avg_divergence = float(np.mean(diffs)) if len(diffs) > 0 else 0.0

    pitch_analysis = (f"ระดับเสียงเฉลี่ยคลาดเคลื่อน {avg_divergence:.2f} Semitones" if score < 80 else f"ทำนองสอดคล้องต้นแบบดีเยี่ยม คลาดเคลื่อน {avg_divergence:.2f} Semitones")
    return float(round(score, 2)), p_ref, p_stu, path, pitch_analysis

def evaluate_rhythm_similarity(ref_audio: str, stu_audio: str, sr=22050):
    y_ref, _ = librosa.load(ref_audio, sr=sr)
    y_stu, _ = librosa.load(stu_audio, sr=sr)
    rms_ref = librosa.feature.rms(y=y_ref)[0]
    rms_stu = librosa.feature.rms(y=y_stu)[0]

    rms_ref = (rms_ref - np.min(rms_ref)) / (np.max(rms_ref) - np.min(rms_ref) + 1e-6)
    rms_stu = (rms_stu - np.min(rms_stu)) / (np.max(rms_stu) - np.min(rms_stu) + 1e-6)

    dist, path = fastdtw(rms_ref, rms_stu, dist=lambda a, b: abs(a - b))
    norm_dist = dist / (len(rms_ref) + len(rms_stu))
    score = max(0.0, 100.0 - (norm_dist * 400))

    duration_ratio = len(rms_stu) / max(1, len(rms_ref))
    if duration_ratio > 1.25: rhythm_analysis = "ท่องช้ากว่าเกณฑ์มาตรฐาน มีการลากเสียงหรือหยุดแช่ยาวนานเกินไป"
    elif duration_ratio < 0.75: rhythm_analysis = "ท่องเร็วกว่าเกณฑ์มาตรฐาน ไม่มีการทอดเสียงและเว้นจังหวะหายใจ"
    else: rhythm_analysis = "การแบ่งวรรคตอนและความเร็วในการทอดเสียงสม่ำเสมอสอดคล้องกับต้นแบบ"

    return float(round(score, 2)), rms_ref, rms_stu, path, rhythm_analysis

def generate_visual_charts(sub_id, p_ref, p_stu, path_p, r_ref, r_stu, path_r):
    chart1_filename = f"chart_pitch_{sub_id}.png"
    chart2_filename = f"chart_rhythm_{sub_id}.png"
    chart1_path = os.path.join(BASE_DIR, "static", chart1_filename)
    chart2_path = os.path.join(BASE_DIR, "static", chart2_filename)

    fig1, ax1 = plt.subplots(figsize=(9, 3.2))
    ax1.plot(p_ref, label='Teacher', color='#2563EB', linewidth=2)
    ax1.plot(p_stu, label='Student', color='#EA580C', linewidth=1.8, linestyle='--')
    ax1.legend(loc='upper right')
    fig1.tight_layout()
    fig1.savefig(chart1_path, dpi=160)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(9, 3.2))
    ax2.plot(r_ref, label='Teacher Energy', color='#059669', linewidth=2)
    ax2.plot(r_stu, label='Student Energy', color='#DC2626', linewidth=1.8, linestyle='--')
    ax2.legend(loc='upper right')
    fig2.tight_layout()
    fig2.savefig(chart2_path, dpi=160)
    plt.close(fig2)

    return f"/static/{chart1_filename}", f"/static/{chart2_filename}"

def generate_student_rubric_feedback(text_score, pitch_score, rhythm_score, aligned_tokens, p_ref, p_stu, path_p, rms_ref, rms_stu):
    return {
        "pronunciation": {"strengths": ["ข้อความจุดแข็งการออกเสียง"], "weaknesses": ["ข้อความจุดอ่อนการออกเสียง"]},
        "pitch": {"strengths": ["ข้อความจุดแข็งทำนอง"], "weaknesses": ["ข้อความจุดอ่อนทำนอง"]},
        "rhythm": {"strengths": ["ข้อความจุดแข็งจังหวะ"], "weaknesses": ["ข้อความจุดอ่อนจังหวะ"]}
    }

# ==========================================
# 🌐 APIs (Refactored for SQL Database)
# ==========================================
@app.get("/", response_class=HTMLResponse)
async def serve_index():
    with open(os.path.join(BASE_DIR, "index.html"), "r", encoding="utf-8") as f:
        return f.read()

@app.post("/api/login")
async def login(email: str = Form(...)):
    conn = get_db()
    user = conn.execute("SELECT * FROM users WHERE email = ?", (email.strip().lower(),)).fetchone()
    if not user:
        conn.close()
        raise HTTPException(status_code=401, detail="อีเมลไม่ถูกต้อง (ใช้ teacher@gmail.com หรือ student@gmail.com)")
    
    classes = [r['class_id'] for r in conn.execute("SELECT class_id FROM user_classes WHERE email = ?", (user['email'],)).fetchall()]
    conn.close()
    return {"email": user['email'], "role": user['role'], "name": user['name'], "allowed_classes": classes}

@app.get("/api/classroom-data")
async def get_classroom_data(email: str):
    conn = get_db()
    classes = conn.execute("""
        SELECT c.* FROM classes c 
        JOIN user_classes uc ON c.id = uc.class_id 
        WHERE uc.email = ?
    """, (email.strip().lower(),)).fetchall()
    
    cls_ids = [c['id'] for c in classes]
    assignments = []
    
    if cls_ids:
        placeholders = ','.join('?' for _ in cls_ids)
        asgs = conn.execute(f"SELECT * FROM assignments WHERE class_id IN ({placeholders})", cls_ids).fetchall()
        
        for a in asgs:
            subs = conn.execute("SELECT * FROM submissions WHERE assignment_id = ?", (a['id'],)).fetchall()
            subs_list = []
            for s in subs:
                eval_data = None
                if s['status'] in ['Evaluated', 'Returned']:
                    eval_data = {
                        "total_score": s['total_score'], "text_score": s['text_score'], 
                        "pitch_score": s['pitch_score'], "rhythm_score": s['rhythm_score'],
                        "reference_text": a['poem_text'], "transcribed_text": s['transcribed_text'],
                        "aligned_tokens": json.loads(s['aligned_tokens']) if s['aligned_tokens'] else [],
                        "pitch_analysis": s['pitch_analysis'], "rhythm_analysis": s['rhythm_analysis'],
                        "chart_pitch": s['chart_pitch'], "chart_rhythm": s['chart_rhythm'],
                        "ref_audio_url": s['eval_ref_audio_url'],
                        "rubric_feedback": json.loads(s['rubric_feedback']) if s['rubric_feedback'] else {}
                    }
                
                subs_list.append({
                    "id": s['id'], "student_email": s['student_email'], "student_name": s['student_name'],
                    "video_url": s['video_url'], "status": s['status'], "teacher_note": s['teacher_note'],
                    "evaluation": eval_data
                })
                
            assignments.append({
                "id": a['id'], "class_id": a['class_id'], "title": a['title'], "description": a['description'],
                "poem_text": a['poem_text'], "weights": {"text": a['w_text'], "pitch": a['w_pitch'], "rhythm": a['w_rhythm']},
                "ref_audio_url": a['ref_audio_url'], "submissions": subs_list
            })
            
    conn.close()
    return {"classes": [dict(c) for c in classes], "assignments": assignments}

@app.post("/api/create-assignment")
async def create_assignment(
    class_id: str = Form(...), title: str = Form(...), description: str = Form(""), poem_text: str = Form(...),
    w_text: int = Form(...), w_pitch: int = Form(...), w_rhythm: int = Form(...), teacher_audio: UploadFile = File(None)
):
    ref_audio_url = None
    if teacher_audio and teacher_audio.filename:
        ref_ext = os.path.splitext(teacher_audio.filename)[1]
        saved_ref_name = f"ref_{uuid.uuid4().hex[:8]}{ref_ext}"
        saved_ref_path = os.path.join(BASE_DIR, "uploads", saved_ref_name)
        with open(saved_ref_path, "wb") as f:
            shutil.copyfileobj(teacher_audio.file, f)
        ref_audio_url = f"/uploads/{saved_ref_name}"

    asg_id = f"asg-{uuid.uuid4().hex[:6]}"
    conn = get_db()
    conn.execute('''
        INSERT INTO assignments (id, class_id, title, description, poem_text, w_text, w_pitch, w_rhythm, tolerance, ref_audio_url)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (asg_id, class_id, title, description, poem_text, w_text, w_pitch, w_rhythm, 35.0, ref_audio_url))
    conn.commit()
    conn.close()
    return {"message": "Success"}

@app.post("/api/submit-recitation")
async def submit_recitation(
    assignment_id: str = Form(...), student_email: str = Form(...), student_name: str = Form(...), student_video: UploadFile = File(...)
):
    stu_ext = os.path.splitext(student_video.filename)[1]
    raw_stu_name = f"stu_{uuid.uuid4().hex[:8]}{stu_ext}"
    raw_stu_path = os.path.join(BASE_DIR, "uploads", raw_stu_name)
    with open(raw_stu_path, "wb") as f:
        shutil.copyfileobj(student_video.file, f)

    sub_id = f"sub-{uuid.uuid4().hex[:6]}"
    conn = get_db()
    conn.execute('''
        INSERT INTO submissions (id, assignment_id, student_email, student_name, video_url, raw_file_path, status, teacher_note)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (sub_id, assignment_id, student_email, student_name, f"/uploads/{raw_stu_name}", raw_stu_path, "Pending", ""))
    conn.commit()
    conn.close()
    return {"message": "ส่งงานสำเร็จเรียบร้อยแล้ว", "submission_id": sub_id}

@app.post("/api/evaluate-submission")
async def evaluate_submission(assignment_id: str = Form(...), submission_id: str = Form(...)):
    try:
        conn = get_db()
        asg = conn.execute("SELECT * FROM assignments WHERE id = ?", (assignment_id,)).fetchone()
        sub = conn.execute("SELECT * FROM submissions WHERE id = ?", (submission_id,)).fetchone()

        raw_stu_path = sub["raw_file_path"]
        raw_ref_path = os.path.join(BASE_DIR, asg["ref_audio_url"].lstrip("/")) if asg["ref_audio_url"] else raw_stu_path

        wav_stu_path = convert_to_wav(raw_stu_path)
        wav_ref_path = convert_to_wav(raw_ref_path)

        text_score, student_text, aligned_tokens = evaluate_text_accuracy(wav_stu_path, asg["poem_text"])
        pitch_score, p_ref, p_stu, path_p, pitch_analysis = evaluate_pitch_similarity(wav_ref_path, wav_stu_path, asg["tolerance"])
        rhythm_score, r_ref, r_stu, path_r, rhythm_analysis = evaluate_rhythm_similarity(wav_ref_path, wav_stu_path)

        final_score = (text_score * (asg["w_text"] / 100)) + (pitch_score * (asg["w_pitch"] / 100)) + (rhythm_score * (asg["w_rhythm"] / 100))
        chart_p_url, chart_r_url = generate_visual_charts(sub["id"], p_ref, p_stu, path_p, r_ref, r_stu, path_r)
        rubric_feedback = generate_student_rubric_feedback(text_score, pitch_score, rhythm_score, aligned_tokens, p_ref, p_stu, path_p, r_ref, r_stu)

        conn.execute('''
            UPDATE submissions SET 
            status = 'Evaluated', total_score = ?, text_score = ?, pitch_score = ?, rhythm_score = ?,
            transcribed_text = ?, pitch_analysis = ?, rhythm_analysis = ?, chart_pitch = ?, chart_rhythm = ?,
            eval_ref_audio_url = ?, rubric_feedback = ?, aligned_tokens = ? WHERE id = ?
        ''', (
            round(final_score, 1), round(text_score, 1), round(pitch_score, 1), round(rhythm_score, 1),
            student_text, pitch_analysis, rhythm_analysis, chart_p_url, chart_r_url, asg["ref_audio_url"],
            json.dumps(rubric_feedback), json.dumps(aligned_tokens), submission_id
        ))
        conn.commit()
        conn.close()
        return {"message": "Success"}
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"detail": f"การประมวลผลล้มเหลว: {str(e)}"})

@app.post("/api/return-score")
async def return_score(assignment_id: str = Form(...), submission_id: str = Form(...), teacher_note: str = Form("")):
    conn = get_db()
    conn.execute("UPDATE submissions SET status = 'Returned', teacher_note = ? WHERE id = ?", (teacher_note, submission_id))
    conn.commit()
    conn.close()
    return {"message": "Success"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
