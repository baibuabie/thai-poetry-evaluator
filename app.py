import os
import shutil
import warnings
import uuid
import traceback
import subprocess
import difflib
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
transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-small", device=device)

# --- Role-based Database ---
db = {
    "users": {
        "teacher@gmail.com": {
            "role": "teacher", 
            "name": "อาจารย์วิชาการ (Teacher)",
            "classes": ["cls-1", "cls-2"]
        },
        "student@gmail.com": {
            "role": "student", 
            "name": "นายสมชาย ใจดี (Student)",
            "classes": ["cls-2"]
        }
    },
    "classes": [
        {"id": "cls-1", "name": "ม.5/1", "subject": "ภาษาไทยพื้นฐาน (ท32101)", "theme_color": "from-emerald-600 to-teal-700"},
        {"id": "cls-2", "name": "ม.5/2", "subject": "วรรณคดีวิจักษ์ (ท32201)", "theme_color": "from-indigo-600 to-blue-700"}
    ],
    "assignments": []
}

def convert_to_wav(source_path: str) -> str:
    target_path = os.path.splitext(source_path)[0] + "_converted.wav"
    if FFMPEG_PATH:
        cmd = [
            FFMPEG_PATH, "-y", "-i", source_path,
            "-vn", "-acodec", "pcm_s16le", "-ar", "22050", "-ac", "1",
            target_path
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode == 0 and os.path.exists(target_path):
            return target_path

    y, sr = librosa.load(source_path, sr=22050)
    sf.write(target_path, y, sr)
    return target_path

def evaluate_text_accuracy(audio_path: str, reference_text: str):
    result = transcriber(
        audio_path,
        return_timestamps=True,
        generate_kwargs={"language": "thai", "task": "transcribe"}
    )
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
                aligned_tokens.append({
                    "char": ch,
                    "status": "error_pronounce",
                    "tooltip": f"เสียงเพี้ยน: ต้นฉบับ '{expected}' แต่ออกเสียงเป็น '{got}'"
                })
        elif tag == 'delete':
            for ch in sub_ref:
                aligned_tokens.append({
                    "char": ch,
                    "status": "error_omission",
                    "tooltip": f"อ่านตกหล่น: ไม่ได้ออกเสียง '{ch}'"
                })

    return float(text_score), transcribed_text, aligned_tokens

def extract_pitch_contour(audio_path: str, sr=22050):
    y, _ = librosa.load(audio_path, sr=sr)
    f0, voiced_flag, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C6'), sr=sr)
    valid_f0 = f0[voiced_flag] if voiced_flag is not None else np.array([])
    if len(valid_f0) < 5:
        return np.zeros(50)
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

    pitch_analysis = (
        f"ระดับเสียงเฉลี่ยคลาดเคลื่อน {avg_divergence:.2f} Semitones โดยพบจุดเอื้อนสูง-ต่ำเบี่ยงเบนจากเส้นต้นแบบของครู"
        if score < 80 else
        f"ทำนองและการเอื้อนสอดคล้องกับต้นแบบดีเยี่ยม คลาดเคลื่อนเฉลี่ยเพียง {avg_divergence:.2f} Semitones"
    )
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
    if duration_ratio > 1.25:
        rhythm_analysis = "ท่องช้ากว่าเกณฑ์มาตรฐาน มีการลากเสียงหรือหยุดแช่ในบางวรรคยาวนานเกินไป"
    elif duration_ratio < 0.75:
        rhythm_analysis = "ท่องเร็วกว่าเกณฑ์มาตรฐาน ไม่มีการทอดเสียงและเว้นจังหวะหายใจระหว่างวรรค"
    else:
        rhythm_analysis = "การแบ่งวรรคตอนและความเร็วในการทอดเสียงสม่ำเสมอสอดคล้องกับต้นแบบ"

    return float(round(score, 2)), rms_ref, rms_stu, path, rhythm_analysis

def generate_visual_charts(sub_id, p_ref, p_stu, path_p, r_ref, r_stu, path_r):
    chart1_filename = f"chart_pitch_{sub_id}.png"
    chart2_filename = f"chart_rhythm_{sub_id}.png"
    chart1_path = os.path.join(BASE_DIR, "static", chart1_filename)
    chart2_path = os.path.join(BASE_DIR, "static", chart2_filename)

    fig1, ax1 = plt.subplots(figsize=(9, 3.2))
    ax1.plot(p_ref, label='Teacher Reference (ต้นแบบครู)', color='#2563EB', linewidth=2)
    ax1.plot(p_stu, label='Student Recitation (นักเรียน)', color='#EA580C', linewidth=1.8, linestyle='--')
    if path_p:
        for (i, j) in path_p[::max(1, len(path_p)//30)]:
            ax1.plot([i, j], [p_ref[i], p_stu[j]], color='#CBD5E1', linestyle=':')
    ax1.set_title("1. การเปรียบเทียบส่วนโค้งทำนอง (Pitch Contour Alignment)", fontsize=10, weight='bold')
    ax1.set_ylabel("ระดับเสียงสัมพัทธ์ (Relative Semitones)")
    ax1.set_xlabel("ลำดับเฟรมเวลา (Time Frames: ~23ms/frame)")
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.2)
    fig1.tight_layout()
    fig1.savefig(chart1_path, dpi=160)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(9, 3.2))
    ax2.plot(r_ref, label='Teacher Energy (จังหวะต้นแบบ)', color='#059669', linewidth=2)
    ax2.plot(r_stu, label='Student Energy (จังหวะนักเรียน)', color='#DC2626', linewidth=1.8, linestyle='--')
    if path_r:
        for (i, j) in path_r[::max(1, len(path_r)//30)]:
            ax2.plot([i, j], [r_ref[i], r_stu[j]], color='#CBD5E1', linestyle=':')
    ax2.set_title("2. การเปรียบเทียบจังหวะและพลังงานเสียง (Rhythm Envelope Alignment)", fontsize=10, weight='bold')
    ax2.set_ylabel("พลังงานความดัง (Normalized Energy: 0.0 - 1.0)")
    ax2.set_xlabel("ลำดับเฟรมเวลา (Time Frames: ~23ms/frame)")
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.2)
    fig2.tight_layout()
    fig2.savefig(chart2_path, dpi=160)
    plt.close(fig2)

    return f"/static/{chart1_filename}", f"/static/{chart2_filename}"

def generate_student_rubric_feedback(text_score, pitch_score, rhythm_score, aligned_tokens, p_ref, p_stu, path_p, rms_ref, rms_stu):
    """สร้างจุดแข็ง (Strengths) และจุดอ่อน/จุดที่ควรพัฒนา (Weaknesses) แยก 3 เกณฑ์สำหรับส่งให้นักเรียน"""
    rubric_feedback = {
        "pronunciation": {"strengths": [], "weaknesses": []},
        "pitch": {"strengths": [], "weaknesses": []},
        "rhythm": {"strengths": [], "weaknesses": []}
    }

    # 1. การออกเสียง
    mispronounced = [t['char'] for t in aligned_tokens if t['status'] == 'error_pronounce']
    omitted = [t['char'] for t in aligned_tokens if t['status'] == 'error_omission']

    if text_score >= 80:
        rubric_feedback["pronunciation"]["strengths"].append("ออกเสียงอักขระ พยัญชนะ สระ และวรรณยุกต์ได้ชัดเจน ถูกต้องตามฉันทลักษณ์ของบทประพันธ์ส่วนใหญ่")
    else:
        rubric_feedback["pronunciation"]["strengths"].append("มีความมั่นใจในการเปล่งเสียงคำศัพท์หลักในบทกลอน")

    if mispronounced:
        sample_w = ' '.join(mispronounced[:4])
        rubric_feedback["pronunciation"]["weaknesses"].append(f"มีคำที่ออกเสียงวรรณยุกต์หรือรูปพยัญชนะคลาดเคลื่อน เช่น: '{sample_w}'")
    if omitted:
        sample_o = ' '.join(omitted[:3])
        rubric_feedback["pronunciation"]["weaknesses"].append(f"มีพยางค์ที่อ่านตกหล่นหรือกลืนเสียงหายไป เช่น: '{sample_o}'")
    if not mispronounced and not omitted:
        rubric_feedback["pronunciation"]["weaknesses"].append("ไม่มีข้อบกพร่องเรื่องคำอ่าน ควรรักษาความชัดถ้อยชัดคำนี้ไว้")

    # 2. ทำนองและการเอื้อน
    bias = float(np.mean([p_stu[j] - p_ref[i] for (i, j) in path_p])) if len(path_p) > 0 else 0.0
    if pitch_score >= 80:
        rubric_feedback["pitch"]["strengths"].append("จับท่วงทำนองของบทกลอนได้ไพเราะ มีการทอดเสียงและยกเสียงสูง-ต่ำตรงตามมาตรฐานครูต้นแบบ")
    else:
        rubric_feedback["pitch"]["strengths"].append("สามารถรักษาระดับโทนเสียงพูดให้นิ่งและต่อเนื่องได้ดีตลอดบทกลอน")

    if pitch_score < 75:
        if bias < 0:
            rubric_feedback["pitch"]["weaknesses"].append("เสียงตกในท่อนเอื้อน มักกดระดับเสียงต่ำกว่าคีย์ทำนองเสนาะมาตรฐาน แนะนำให้เปิดช่องคอเพื่อยกเสียงขึ้น")
        else:
            rubric_feedback["pitch"]["weaknesses"].append("เอื้อนเสียงสูงและแหลมเกินไปในบางวรรค ทำให้หลุดออกจากบันไดเสียงต้นแบบ")
    else:
        rubric_feedback["pitch"]["weaknesses"].append("ท่วงทำนองอยู่ในเกณฑ์ดี หากเพิ่มความประณีตในการสั่นเสียง (Vibrato) ปลายวรรคจะไพเราะยิ่งขึ้น")

    # 3. จังหวะและการเว้นวรรค
    dur_ratio = len(rms_stu) / max(1, len(rms_ref))
    if rhythm_score >= 80:
        rubric_feedback["rhythm"]["strengths"].append("จังหวะการลงเสียงหนัก-เบา และการหยุดพักหายใจระหว่างวรรคทำได้ถูกต้องเป็นธรรมชาติ ไม่เร่งรีบ")
    else:
        rubric_feedback["rhythm"]["strengths"].append("สามารถท่องวรรคตอนตั้งแต่ต้นจนจบได้อย่างต่อเนื่อง")

    if dur_ratio > 1.20:
        rubric_feedback["rhythm"]["weaknesses"].append(f"ท่องช้ากว่ามาตรฐานประมาณ {int((dur_ratio-1)*100)}% แช่เสียงท้ายวรรคนานเกินไป ควรจัดสรรลมหายใจให้กระชับขึ้น")
    elif dur_ratio < 0.80:
        rubric_feedback["rhythm"]["weaknesses"].append(f"ท่องเร่งจังหวะเร็วกว่ามาตรฐานประมาณ {int((1-dur_ratio)*100)}% เว้นช่วงหยุดหายใจสั้นเกินไป ควรทอดหางเสียงให้ครบช่วงจังหวะ")
    else:
        rubric_feedback["rhythm"]["weaknesses"].append("จังหวะโดยรวมสม่ำเสมอ แต่อาจเน้นน้ำหนักคำหนัก-เบาในคำเอก-คำโทให้ชัดเจนกว่านี้")

    return rubric_feedback

# --- APIs ---
@app.get("/", response_class=HTMLResponse)
async def serve_index():
    with open(os.path.join(BASE_DIR, "index.html"), "r", encoding="utf-8") as f:
        return f.read()

@app.post("/api/login")
async def login(email: str = Form(...)):
    user = db["users"].get(email.strip().lower())
    if not user:
        raise HTTPException(status_code=401, detail="อีเมลไม่ถูกต้อง (ใช้ teacher@gmail.com หรือ student@gmail.com)")
    return {
        "email": email,
        "role": user["role"],
        "name": user["name"],
        "allowed_classes": user["classes"]
    }

@app.get("/api/classroom-data")
async def get_classroom_data(email: str):
    user = db["users"].get(email.strip().lower())
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    filtered_classes = [c for c in db["classes"] if c["id"] in user["classes"]]
    return {
        "classes": filtered_classes,
        "assignments": db["assignments"]
    }

@app.post("/api/create-assignment")
async def create_assignment(
    class_id: str = Form(...),
    title: str = Form(...),
    description: str = Form(""),
    poem_text: str = Form(...),
    w_text: int = Form(...),
    w_pitch: int = Form(...),
    w_rhythm: int = Form(...),
    teacher_audio: UploadFile = File(None)
):
    if w_text + w_pitch + w_rhythm != 100:
        raise HTTPException(status_code=400, detail="ผลรวมค่าน้ำหนักต้องเท่ากับ 100% พอดี")

    ref_audio_url = None
    if teacher_audio and teacher_audio.filename:
        ref_ext = os.path.splitext(teacher_audio.filename)[1]
        saved_ref_name = f"ref_{uuid.uuid4().hex[:8]}{ref_ext}"
        saved_ref_path = os.path.join(BASE_DIR, "uploads", saved_ref_name)
        with open(saved_ref_path, "wb") as f:
            shutil.copyfileobj(teacher_audio.file, f)
        ref_audio_url = f"/uploads/{saved_ref_name}"

    new_asg = {
        "id": f"asg-{uuid.uuid4().hex[:6]}",
        "class_id": class_id,
        "title": title,
        "description": description,
        "poem_text": poem_text,
        "weights": {"text": w_text, "pitch": w_pitch, "rhythm": w_rhythm},
        "tolerance": 35.0,
        "ref_audio_url": ref_audio_url,
        "submissions": []
    }
    db["assignments"].append(new_asg)
    return new_asg

@app.put("/api/update-assignment/{assignment_id}")
async def update_assignment(
    assignment_id: str,
    title: str = Form(...),
    description: str = Form(""),
    poem_text: str = Form(...),
    w_text: int = Form(...),
    w_pitch: int = Form(...),
    w_rhythm: int = Form(...),
    teacher_audio: UploadFile = File(None)
):
    asg = next((a for a in db["assignments"] if a["id"] == assignment_id), None)
    if not asg:
        raise HTTPException(status_code=404, detail="ไม่พบ Assignment นี้")

    if w_text + w_pitch + w_rhythm != 100:
        raise HTTPException(status_code=400, detail="ผลรวมค่าน้ำหนักต้องเท่ากับ 100% พอดี")

    asg["title"] = title
    asg["description"] = description
    asg["poem_text"] = poem_text
    asg["weights"] = {"text": w_text, "pitch": w_pitch, "rhythm": w_rhythm}

    if teacher_audio and teacher_audio.filename:
        ref_ext = os.path.splitext(teacher_audio.filename)[1]
        saved_ref_name = f"ref_{uuid.uuid4().hex[:8]}{ref_ext}"
        saved_ref_path = os.path.join(BASE_DIR, "uploads", saved_ref_name)
        with open(saved_ref_path, "wb") as f:
            shutil.copyfileobj(teacher_audio.file, f)
        asg["ref_audio_url"] = f"/uploads/{saved_ref_name}"

    return asg

@app.post("/api/submit-recitation")
async def submit_recitation(
    assignment_id: str = Form(...),
    student_email: str = Form(...),
    student_name: str = Form(...),
    student_video: UploadFile = File(...)
):
    asg = next((a for a in db["assignments"] if a["id"] == assignment_id), None)
    if not asg:
        raise HTTPException(status_code=404, detail="ไม่พบ Assignment นี้")

    stu_ext = os.path.splitext(student_video.filename)[1]
    raw_stu_name = f"stu_{uuid.uuid4().hex[:8]}{stu_ext}"
    raw_stu_path = os.path.join(BASE_DIR, "uploads", raw_stu_name)
    with open(raw_stu_path, "wb") as f:
        shutil.copyfileobj(student_video.file, f)

    sub_id = f"sub-{uuid.uuid4().hex[:6]}"
    submission_entry = {
        "id": sub_id,
        "student_email": student_email,
        "student_name": student_name,
        "video_url": f"/uploads/{raw_stu_name}",
        "raw_file_path": raw_stu_path,
        "status": "Pending",
        "evaluation": None,
        "teacher_note": ""
    }
    asg["submissions"].append(submission_entry)
    return {"message": "ส่งงานสำเร็จเรียบร้อยแล้ว", "submission_id": sub_id}

@app.post("/api/evaluate-submission")
async def evaluate_submission(
    assignment_id: str = Form(...),
    submission_id: str = Form(...)
):
    try:
        asg = next((a for a in db["assignments"] if a["id"] == assignment_id), None)
        if not asg:
            raise HTTPException(status_code=404, detail="Assignment not found")

        sub = next((s for s in asg["submissions"] if s["id"] == submission_id), None)
        if not sub:
            raise HTTPException(status_code=404, detail="Submission not found")

        raw_stu_path = sub["raw_file_path"]
        if asg.get("ref_audio_url"):
            raw_ref_path = os.path.join(BASE_DIR, asg["ref_audio_url"].lstrip("/"))
        else:
            raw_ref_path = raw_stu_path

        wav_stu_path = convert_to_wav(raw_stu_path)
        wav_ref_path = convert_to_wav(raw_ref_path)

        text_score, student_text, aligned_tokens = evaluate_text_accuracy(wav_stu_path, asg["poem_text"])
        pitch_score, p_ref, p_stu, path_p, pitch_analysis = evaluate_pitch_similarity(wav_ref_path, wav_stu_path, asg["tolerance"])
        rhythm_score, r_ref, r_stu, path_r, rhythm_analysis = evaluate_rhythm_similarity(wav_ref_path, wav_stu_path)

        w = asg["weights"]
        final_score = (text_score * (w["text"] / 100)) + (pitch_score * (w["pitch"] / 100)) + (rhythm_score * (w["rhythm"] / 100))

        chart_p_url, chart_r_url = generate_visual_charts(sub["id"], p_ref, p_stu, path_p, r_ref, r_stu, path_r)
        rubric_feedback = generate_student_rubric_feedback(
            text_score, pitch_score, rhythm_score, 
            aligned_tokens, p_ref, p_stu, path_p, 
            r_ref, r_stu
        )

        sub["status"] = "Evaluated"
        sub["evaluation"] = {
            "total_score": round(final_score, 1),
            "text_score": round(text_score, 1),
            "pitch_score": round(pitch_score, 1),
            "rhythm_score": round(rhythm_score, 1),
            "reference_text": asg["poem_text"],
            "transcribed_text": student_text,
            "aligned_tokens": aligned_tokens,
            "pitch_analysis": pitch_analysis,
            "rhythm_analysis": rhythm_analysis,
            "chart_pitch": chart_p_url,
            "chart_rhythm": chart_r_url,
            "ref_audio_url": asg.get("ref_audio_url"),
            "rubric_feedback": rubric_feedback
        }

        return JSONResponse(sub)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"detail": f"การประมวลผลล้มเหลว: {str(e)}"})

@app.post("/api/return-score")
async def return_score(
    assignment_id: str = Form(...),
    submission_id: str = Form(...),
    teacher_note: str = Form("")
):
    asg = next((a for a in db["assignments"] if a["id"] == assignment_id), None)
    if not asg:
        raise HTTPException(status_code=404, detail="Assignment not found")

    sub = next((s for s in asg["submissions"] if s["id"] == submission_id), None)
    if not sub:
        raise HTTPException(status_code=404, detail="Submission not found")

    sub["status"] = "Returned"
    sub["teacher_note"] = teacher_note
    return {"message": "ส่งคะแนนและข้อเสนอแนะคืนนักเรียนสำเร็จแล้ว", "submission": sub}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)