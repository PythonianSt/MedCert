import base64
import html
import json
import re
import uuid
from datetime import date, datetime, timedelta
from io import BytesIO, StringIO
from pathlib import Path
from urllib.parse import quote
from zoneinfo import ZoneInfo

import cv2
import numpy as np
import pandas as pd
import qrcode
import requests
import streamlit as st
from openai import OpenAI
from PIL import Image

WEASYPRINT_IMPORT_ERROR = ""
try:
    from weasyprint import HTML
    WEASYPRINT_AVAILABLE = True
except Exception as import_error:
    HTML = None
    WEASYPRINT_AVAILABLE = False
    WEASYPRINT_IMPORT_ERROR = f"{type(import_error).__name__}: {import_error}"


# =====================================================
# App configuration
# =====================================================
st.set_page_config(
    page_title="ใบรับรองแพทย์ KU KPS",
    page_icon="🩺",
    layout="wide",
)

BKK = ZoneInfo("Asia/Bangkok")

GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]
GITHUB_REPO = st.secrets["GITHUB_REPO"]
GITHUB_BRANCH = st.secrets.get("GITHUB_BRANCH", "main")
CSV_PATH = st.secrets.get("CSV_PATH", "medical_certificate.csv")

PASS_REG = st.secrets.get("PASS_REG", "KUKPS01")
PASS_LAB = st.secrets.get("PASS_LAB", "KUKPS02")
PASS_DOC = st.secrets.get("PASS_DOC", "KUKPS03")
PASS_PRINT = st.secrets.get("PASS_PRINT", "KUKPS04")

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
OPENAI_MODEL = st.secrets.get("OPENAI_MODEL", "gpt-4.1-mini")

DOCTORS = {
    "นายแพทย์กำธร ตันติวิทยาทันต์": "12082",
    "นายแพทย์สมชาย เจนลาภวัฒนกุล": "15771",
}

# เวลานัดหมายใหม่: 08.30–12.00 และ 15.00–16.30 น.
TIME_SLOTS = [
    "08:30", "09:00", "09:30", "10:00", "10:30", "11:00", "11:30", "12:00",
    "15:00", "15:30", "16:00", "16:30",
]

ACTIVE_STATUSES = {
    "booked", "registered", "lab_done", "doctor_approved", "printed"
}

STATUS_LABELS = {
    "booked": "นัดหมายแล้ว",
    "registered": "เวชระเบียนยืนยันแล้ว",
    "lab_done": "ลงผลตรวจปัสสาวะแล้ว",
    "doctor_approved": "แพทย์อนุมัติแล้ว",
    "printed": "พิมพ์ใบรับรองแล้ว",
    "cancelled": "ยกเลิก",
}

CONFIRMATION_OPTIONS = [
    "รอการยืนยัน",
    "ส่งอีเมลแล้ว",
    "ผู้รับบริการยืนยันแล้ว",
    "ติดต่อไม่ได้",
    "ยกเลิกนัด",
]


# =====================================================
# GitHub CSV storage
# =====================================================
def github_headers():
    return {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def read_csv():
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{CSV_PATH}"
    response = requests.get(
        url,
        headers=github_headers(),
        params={"ref": GITHUB_BRANCH},
        timeout=30,
    )

    if response.status_code == 404:
        return pd.DataFrame(), None

    response.raise_for_status()
    payload = response.json()
    content = base64.b64decode(payload["content"]).decode("utf-8-sig")
    return pd.read_csv(StringIO(content), dtype=str).fillna(""), payload["sha"]


def ensure_columns(df):
    columns = [
        "record_id", "created_at_bkk", "last_modified_at_bkk", "status",
        "appointment_date", "appointment_time",
        "email", "confirmation_status", "confirmation_note", "confirmation_at_bkk",
        "edit_count_date", "edit_count_today", "last_appointment_edit_at_bkk",
        "citizen_id", "prefix", "full_name", "sex", "address", "purpose",
        "chronic", "chronic_detail", "accident", "accident_detail",
        "hospital", "hospital_detail", "epilepsy", "epilepsy_detail",
        "other_history", "other_history_detail",
        "hn", "registration_note", "registered_at_bkk",
        "urine_meth_result", "urine_note", "urine_checked_at_bkk",
        "vital_bp", "vital_pulse", "vital_weight", "vital_height",
        "vital_ai_raw", "vital_checked_at_bkk",
        "doctor_name", "doctor_license", "weight", "height", "bp", "pulse",
        "general_status", "abnormal_detail", "other_exam",
        "doctor_opinion", "doctor_approved_at_bkk",
        "printed_at_bkk",
    ]

    for column in columns:
        if column not in df.columns:
            df[column] = ""

    # เติมค่าเริ่มต้นให้ข้อมูลเดิม
    df.loc[df["confirmation_status"].eq(""), "confirmation_status"] = "รอการยืนยัน"
    df.loc[df["edit_count_today"].eq(""), "edit_count_today"] = "0"
    return df


def save_csv(df, sha=None):
    csv_text = df.to_csv(index=False, encoding="utf-8-sig")
    encoded = base64.b64encode(csv_text.encode("utf-8-sig")).decode("utf-8")

    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{CSV_PATH}"
    payload = {
        "message": f"update medical certificate {now_bkk().isoformat()}",
        "content": encoded,
        "branch": GITHUB_BRANCH,
    }
    if sha:
        payload["sha"] = sha

    response = requests.put(
        url,
        headers=github_headers(),
        json=payload,
        timeout=30,
    )
    response.raise_for_status()


# =====================================================
# Helpers
# =====================================================
def now_bkk():
    return datetime.now(BKK)


def clean_id(value):
    return "".join(character for character in str(value) if character.isdigit())


def normalize_email(value):
    return str(value).strip().lower()


def valid_email(value):
    email_value = normalize_email(value)
    pattern = r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"
    return bool(re.fullmatch(pattern, email_value))


def is_workday(day_value):
    return day_value.weekday() < 5


def thai_date(day_value):
    if isinstance(day_value, datetime):
        day_value = day_value.date()

    months = [
        "", "มกราคม", "กุมภาพันธ์", "มีนาคม", "เมษายน", "พฤษภาคม", "มิถุนายน",
        "กรกฎาคม", "สิงหาคม", "กันยายน", "ตุลาคม", "พฤศจิกายน", "ธันวาคม",
    ]
    return f"{day_value.day} {months[day_value.month]} {day_value.year + 543}"


def display_date(value):
    try:
        return datetime.strptime(str(value), "%Y-%m-%d").strftime("%d/%m/%Y")
    except Exception:
        return str(value)


def parse_iso_datetime(value):
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=BKK)
        return parsed.astimezone(BKK)
    except Exception:
        return None


def get_issue_date(row):
    approved = parse_iso_datetime(row.get("doctor_approved_at_bkk", ""))
    return approved.date() if approved else now_bkk().date()


def start_of_week(day_value):
    return day_value - timedelta(days=day_value.weekday())


def weekly_booking_count(df, citizen_id, target_date, exclude_record_id=None):
    if df.empty:
        return 0

    week_start = start_of_week(target_date)
    week_end = week_start + timedelta(days=6)

    working = df.copy()
    working["_appt_date"] = pd.to_datetime(
        working["appointment_date"], errors="coerce"
    ).dt.date

    mask = (
        working["citizen_id"].astype(str).eq(str(citizen_id))
        & working["status"].astype(str).isin(ACTIVE_STATUSES)
        & working["_appt_date"].apply(
            lambda item: bool(item and week_start <= item <= week_end)
        )
    )

    if exclude_record_id:
        mask &= ~working["record_id"].astype(str).eq(str(exclude_record_id))

    return int(mask.sum())


def edits_used_today(df, citizen_id):
    if df.empty:
        return 0

    today_text = now_bkk().date().isoformat()
    same_person = df[df["citizen_id"].astype(str).eq(str(citizen_id))]
    same_day = same_person[same_person["edit_count_date"].astype(str).eq(today_text)]

    total = 0
    for value in same_day["edit_count_today"].tolist():
        try:
            total += int(float(value))
        except Exception:
            pass
    return total


def make_qr(record_id):
    image = qrcode.make(record_id)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def find_by_record(df, record_id):
    if df.empty or "record_id" not in df.columns:
        return None

    matches = df[df["record_id"].astype(str).str.upper() == str(record_id).strip().upper()]
    if matches.empty:
        return None
    return matches.index[0]


def password_gate(correct_password, key):
    password = st.sidebar.text_input("รหัสผ่าน", type="password", key=key)
    if password != correct_password:
        st.warning("กรุณาใส่รหัสผ่านให้ถูกต้อง")
        st.stop()


def read_qr_from_image(uploaded_file):
    try:
        image = Image.open(uploaded_file).convert("RGB")
        image_array = np.array(image)
        image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        detector = cv2.QRCodeDetector()
        data, _, _ = detector.detectAndDecode(image_bgr)
        return data.strip() if data else ""
    except Exception:
        return ""


def image_to_data_url(uploaded_file):
    image_bytes = uploaded_file.getvalue()
    mime_type = getattr(uploaded_file, "type", None) or "image/jpeg"
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def extract_vitals_with_ai(uploaded_file):
    if not OPENAI_API_KEY:
        raise RuntimeError("ยังไม่ได้กำหนด OPENAI_API_KEY ใน Streamlit Secrets")

    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "อ่านค่าจากกระดาษบันทึกสัญญาณชีพในภาพนี้ "
                            "ให้หาเฉพาะความดันโลหิตและชีพจร โดย BP ต้องอยู่ในรูป "
                            "systolic/diastolic เช่น 120/80 และ pulse เป็นจำนวนครั้งต่อนาที "
                            "ห้ามเดาค่าที่มองไม่ชัด ให้คืน JSON เท่านั้นในรูป "
                            '{"bp":"", "pulse":"", "note":""}'
                        ),
                    },
                    {
                        "type": "input_image",
                        "image_url": image_to_data_url(uploaded_file),
                    },
                ],
            }
        ],
    )

    raw = (response.output_text or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I | re.S).strip()
    try:
        data = json.loads(cleaned)
    except Exception as error:
        raise RuntimeError(f"AI ส่งผลลัพธ์ที่อ่านไม่ได้: {raw}") from error

    return {
        "bp": str(data.get("bp", "")).strip(),
        "pulse": str(data.get("pulse", "")).strip(),
        "note": str(data.get("note", "")).strip(),
        "raw": raw,
    }


def valid_bp(value):
    text = str(value).strip()
    match = re.fullmatch(r"(\d{2,3})\s*/\s*(\d{2,3})", text)
    if not match:
        return False
    systolic, diastolic = map(int, match.groups())
    return 50 <= systolic <= 300 and 30 <= diastolic <= 200


def valid_positive_number(value, minimum, maximum):
    try:
        number = float(str(value).strip())
        return minimum <= number <= maximum
    except Exception:
        return False


def scan_or_enter(key_prefix):
    camera_image = st.camera_input(
        "ถ่ายภาพ QR code",
        key=f"{key_prefix}_camera",
    )

    qr_text = ""
    if camera_image is not None:
        qr_text = read_qr_from_image(camera_image)
        if qr_text:
            st.success("อ่าน QR code สำเร็จ")
            st.code(qr_text)
        else:
            st.warning("ยังอ่าน QR ไม่ได้ กรุณาถ่ายใหม่ให้ QR ชัดและอยู่กลางภาพ")

    manual = st.text_input(
        "หรือกรอกรหัสจาก QR ด้วยตนเอง",
        key=f"{key_prefix}_manual",
    )

    return manual.strip() if manual.strip() else qr_text.strip()


def safe_int(value, default=0):
    try:
        return int(float(value))
    except Exception:
        return default


def h(value):
    return html.escape(str(value or ""), quote=True)


def checked(value, target):
    return "☑" if str(value) == target else "☐"


def citizen_id_display(citizen_id):
    cid = clean_id(citizen_id)
    if len(cid) == 13:
        return f"{cid[0]}-{cid[1:5]}-{cid[5:10]}-{cid[10:12]}-{cid[12]}"
    return h(citizen_id)


def embedded_font_css():
    candidates = [
        Path("THSarabunNew.ttf"),
        Path(__file__).resolve().parent / "THSarabunNew.ttf",
    ]

    for font_path in candidates:
        if font_path.exists():
            encoded = base64.b64encode(font_path.read_bytes()).decode("ascii")
            return f"""
            @font-face {{
                font-family: 'THSarabunEmbedded';
                src: url(data:font/ttf;base64,{encoded}) format('truetype');
                font-weight: normal;
                font-style: normal;
            }}
            """
    return ""


def build_certificate_html(row):
    issue_date = thai_date(get_issue_date(row))
    name_text = f"{row.get('prefix', '')}{row.get('full_name', '')}"
    cid_text = citizen_id_display(row.get("citizen_id", ""))
    font_css = embedded_font_css()

    return f"""
    <style>
        {font_css}
        @page {{ size: A4; margin: 7mm 9mm; }}
        .certificate-page {{
            box-sizing: border-box; width: 100%; max-width: 192mm; margin: 0 auto;
            font-family: 'THSarabunEmbedded', 'TH Sarabun New', 'Sarabun', sans-serif;
            font-size: 13.2pt; line-height: 1.08; color: #111; background: white;
        }}
        .certificate-page h1 {{ margin: 0 0 2mm 0; text-align: center; font-size: 19pt; line-height: 1; }}
        .certificate-page p {{ margin: 0.45mm 0; }}
        .certificate-page ol {{ margin: 0.5mm 0 0.5mm 7mm; padding-left: 5mm; }}
        .certificate-page li {{ margin: 0.15mm 0; }}
        .right {{ text-align: right; }} .center {{ text-align: center; }} .bold {{ font-weight: 700; }}
        .section-label {{ display:inline-block; border:1px solid #222; border-radius:1.5mm; padding:0.2mm 2mm; font-weight:700; }}
        .line {{ display:inline-block; min-width:38mm; padding:0 1mm; border-bottom:1px dotted #333; vertical-align:baseline; }}
        .line-short {{ min-width:23mm; }} .line-long {{ min-width:82mm; }} .line-full {{ min-width:135mm; }}
        .section-divider {{ border-top: 1px solid #555; margin: 1.5mm 0 1mm 0; }}
        .signature-row {{ display:flex; justify-content:flex-end; gap:4mm; margin-top:1mm; }}
        .small-note {{ font-size: 11.5pt; }}
        @media screen {{ .certificate-page {{ box-shadow:0 0 10px rgba(0,0,0,.12); padding:7mm 9mm; min-height:297mm; }} }}
        @media print {{ .certificate-page {{ box-shadow:none; padding:0; }} }}
    </style>

    <div class="certificate-page">
      <h1>ใบรับรองแพทย์ (สำหรับใบอนุญาตขับรถ)</h1>
      <p class="right">เลขที่ <span class="line line-short">&nbsp;</span></p>
      <p><span class="section-label">ส่วนที่ 1</span> <span class="bold">ของผู้ขอรับใบรับรองสุขภาพ</span></p>
      <p>ข้าพเจ้า <span class="line line-long">{h(name_text)}</span> เลขบัตรประชาชน <span class="line">{cid_text}</span></p>
      <p>สถานที่อยู่ที่ติดต่อได้ <span class="line line-full">{h(row.get('address', ''))}</span></p>
      <p>อีเมล <span class="line line-long">{h(row.get('email', ''))}</span></p>
      <p>ข้าพเจ้าขอใบรับรองสุขภาพ โดยมีประวัติสุขภาพดังนี้</p>
      <p>1. โรคประจำตัว {checked(row.get('chronic'), 'ไม่มี')} ไม่มี {checked(row.get('chronic'), 'มี')} มี ระบุ <span class="line">{h(row.get('chronic_detail', ''))}</span></p>
      <p>2. อุบัติเหตุและผ่าตัด {checked(row.get('accident'), 'ไม่มี')} ไม่มี {checked(row.get('accident'), 'มี')} มี ระบุ <span class="line">{h(row.get('accident_detail', ''))}</span></p>
      <p>3. เคยรักษาในโรงพยาบาล {checked(row.get('hospital'), 'ไม่มี')} ไม่มี {checked(row.get('hospital'), 'มี')} มี ระบุ <span class="line">{h(row.get('hospital_detail', ''))}</span></p>
      <p>4. โรคลมชัก * {checked(row.get('epilepsy'), 'ไม่มี')} ไม่มี {checked(row.get('epilepsy'), 'มี')} มี ระบุ <span class="line">{h(row.get('epilepsy_detail', ''))}</span></p>
      <p>5. ประวัติอื่นที่สำคัญ {checked(row.get('other_history'), 'ไม่มี')} ไม่มี {checked(row.get('other_history'), 'มี')} มี ระบุ <span class="line">{h(row.get('other_history_detail', ''))}</span></p>
      <div class="signature-row"><span>ลงชื่อ <span class="line">&nbsp;</span> ผู้ขอรับใบรับรองสุขภาพ</span><span>วันที่ <span class="line line-short">&nbsp;</span></span></div>

      <div class="section-divider"></div>
      <p><span class="section-label">ส่วนที่ 2</span> <span class="bold">ของแพทย์</span></p>
      <p>สถานที่ตรวจ สถานพยาบาลมหาวิทยาลัยเกษตรศาสตร์ วิทยาเขตกำแพงแสน วันที่ <span class="line line-short">{issue_date}</span></p>
      <p>ข้าพเจ้า <span class="line">{h(row.get('doctor_name', ''))}</span> ใบอนุญาตประกอบวิชาชีพเวชกรรมเลขที่ <span class="line line-short">{h(row.get('doctor_license', ''))}</span></p>
      <p>สถานพยาบาลมหาวิทยาลัยเกษตรศาสตร์ วิทยาเขตกำแพงแสน เลขที่ 1 หมู่ 6 ต.กำแพงแสน อ.กำแพงแสน จ.นครปฐม 73140</p>
      <p>ได้ตรวจร่างกาย <span class="line">{h(name_text)}</span> เมื่อวันที่ <span class="line line-short">{issue_date}</span></p>
      <p>น้ำหนัก <span class="line line-short">{h(row.get('weight', ''))}</span> กก. ความสูง <span class="line line-short">{h(row.get('height', ''))}</span> ซม. ความดันโลหิต <span class="line line-short">{h(row.get('bp', ''))}</span> มม.ปรอท ชีพจร <span class="line line-short">{h(row.get('pulse', ''))}</span> ครั้ง/นาที</p>
      <p>สุขภาพทั่วไป {checked(row.get('general_status'), 'ปกติ')} ปกติ {checked(row.get('general_status'), 'ผิดปกติ')} ผิดปกติ ระบุ <span class="line">{h(row.get('abnormal_detail', ''))}</span> ผล Methamphetamine <span class="line line-short">{h(row.get('urine_meth_result', ''))}</span></p>
      <p>ขอรับรองว่าบุคคลดังกล่าวไม่เป็นผู้มีร่างกายทุพพลภาพจนไม่สามารถปฏิบัติหน้าที่ได้ ไม่ปรากฏอาการของโรคจิตหรือจิตฟั่นเฟือนหรือปัญญาอ่อน ไม่ปรากฏอาการติดยาเสพติดให้โทษหรือโรคพิษสุราเรื้อรัง และไม่ปรากฏอาการของโรคต่อไปนี้</p>
      <ol><li>โรคเรื้อนในระยะติดต่อ หรือระยะที่ปรากฏอาการเป็นที่รังเกียจแก่สังคม</li><li>วัณโรคในระยะอันตราย</li><li>โรคเท้าช้างในระยะที่ปรากฏอาการเป็นที่รังเกียจแก่สังคม</li><li>อื่น ๆ ถ้ามี <span class="line line-long">{h(row.get('other_exam', ''))}</span></li></ol>
      <p>สรุปความคิดเห็นและข้อแนะนำของแพทย์ <span class="line line-full">{h(row.get('doctor_opinion', ''))}</span></p>
      <div class="signature-row"><span>ลงชื่อ <span class="line">&nbsp;</span> แพทย์ผู้ตรวจ ({h(row.get('doctor_name', ''))})</span></div>
      <p class="small-note">หมายเหตุ: ประทับตราสถานพยาบาลหลังพิมพ์เอกสาร</p>
    </div>
    """


def create_certificate_pdf(row):
    if not WEASYPRINT_AVAILABLE:
        raise RuntimeError(
            "ไม่สามารถโหลด WeasyPrint ได้: "
            + (WEASYPRINT_IMPORT_ERROR or "ไม่ทราบสาเหตุ")
        )

    document_html = build_certificate_html(row)
    pdf_bytes = HTML(string=document_html, base_url=str(Path.cwd())).write_pdf()
    return BytesIO(pdf_bytes)


def dataframe_for_dashboard(source_df):
    if source_df.empty:
        return pd.DataFrame()

    output = source_df.copy()
    output["วันนัดหมาย"] = output["appointment_date"].apply(display_date)
    output["เวลา"] = output["appointment_time"].astype(str)
    output["ชื่อ-นามสกุล"] = (
        output["prefix"].astype(str) + output["full_name"].astype(str)
    )
    output["อีเมล"] = output["email"].astype(str)
    output["สถานะการยืนยัน"] = output["confirmation_status"].replace("", "รอการยืนยัน")
    output["สถานะงาน"] = output["status"].map(STATUS_LABELS).fillna(output["status"])
    output["รหัสรายการ"] = output["record_id"].astype(str)

    return output[[
        "วันนัดหมาย", "เวลา", "ชื่อ-นามสกุล", "อีเมล",
        "สถานะการยืนยัน", "สถานะงาน", "รหัสรายการ",
    ]]


# =====================================================
# Global CSS
# =====================================================
st.markdown(
    """
    <style>
        .notice-box {
            border-left: 6px solid #2b6cb0;
            background: #eef6ff;
            padding: 12px 16px;
            border-radius: 6px;
            margin-bottom: 14px;
        }
        .notice-box p { margin: 4px 0; }
        .dashboard-note {
            background: #f7fafc;
            border: 1px solid #cbd5e0;
            border-radius: 8px;
            padding: 10px 14px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =====================================================
# Load data
# =====================================================
try:
    df, sha = read_csv()
    df = ensure_columns(df)
except Exception as error:
    st.error(f"ไม่สามารถอ่านข้อมูลจาก GitHub ได้: {error}")
    st.stop()


# =====================================================
# Sidebar menu
# =====================================================
st.sidebar.title("เมนู")
page = st.sidebar.radio(
    "เลือกหน้า",
    [
        "ผู้รับบริการ",
        "วัดสัญญาณชีพ",
        "เวชระเบียน",
        "ลงผลตรวจปัสสาวะ",
        "พยาบาล/แพทย์",
        "พิมพ์",
    ],
)


# =====================================================
# 1. ผู้รับบริการ
# =====================================================
if page == "ผู้รับบริการ":
    st.title("ระบบนัดหมายขอใบรับรองแพทย์ KU KPS Infirmary")

    st.markdown(
        """
        <div class="notice-box">
            <p><b>คำชี้แจงก่อนนัดหมาย</b></p>
            <p>• ท่านสามารถแก้ไขวันหรือเวลานัดหมายได้ไม่เกินวันละ 2 ครั้ง</p>
            <p>• ระบบอนุญาตให้มีนัดหมายได้มากที่สุดสัปดาห์ละ 2 ครั้ง</p>
            <p>• เวลานัดหมายคือ 08.30–12.00 น. และ 15.00–16.30 น.</p>
            <p>• หลังบันทึก โปรดเก็บ QR code หรือรหัสรายการไว้ใช้ติดต่อและแก้ไขนัดหมาย</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    booking_tab, edit_tab = st.tabs(["นัดหมายใหม่", "แก้ไขนัดหมาย"])

    today = now_bkk().date()
    max_day = today + timedelta(days=30)

    with booking_tab:
        with st.form("patient_booking_form"):
            st.subheader("ข้อมูลนัดหมาย")

            appt_date = st.date_input(
                "เลือกวันนัดหมาย",
                min_value=today,
                max_value=max_day,
                value=today,
                format="DD/MM/YYYY",
            )
            appt_time = st.selectbox("เลือกเวลา", TIME_SLOTS)

            st.subheader("ส่วนที่ 1 ของผู้ขอรับใบรับรองสุขภาพ")

            citizen_id = st.text_input("เลขบัตรประชาชน 13 หลัก")
            email = st.text_input("อีเมลสำหรับรับการยืนยันนัดหมาย")
            prefix = st.selectbox("คำนำหน้า", ["นาย", "นาง", "นางสาว", "อื่น ๆ"])
            other_prefix = st.text_input("ระบุคำนำหน้า") if prefix == "อื่น ๆ" else ""
            full_name = st.text_input("ชื่อ-นามสกุล")
            sex = st.selectbox("เพศ", ["ชาย", "หญิง", "อื่น ๆ"])
            address = st.text_area("ที่อยู่ที่ติดต่อได้")

            purpose = st.selectbox(
                "วัตถุประสงค์",
                ["ใบอนุญาตขับรถ", "สมัครงาน", "สมัครเรียน", "อื่น ๆ"],
            )
            purpose_other = (
                st.text_input("ระบุวัตถุประสงค์อื่น") if purpose == "อื่น ๆ" else ""
            )

            chronic = st.radio("1. โรคประจำตัว", ["ไม่มี", "มี"], horizontal=True)
            chronic_detail = st.text_input("ระบุโรคประจำตัว") if chronic == "มี" else ""

            accident = st.radio("2. อุบัติเหตุและผ่าตัด", ["ไม่มี", "มี"], horizontal=True)
            accident_detail = (
                st.text_input("ระบุอุบัติเหตุ/ผ่าตัด") if accident == "มี" else ""
            )

            hospital = st.radio(
                "3. เคยเข้ารับการรักษาในโรงพยาบาล",
                ["ไม่มี", "มี"],
                horizontal=True,
            )
            hospital_detail = (
                st.text_input("ระบุรายละเอียดการรักษาในโรงพยาบาล")
                if hospital == "มี" else ""
            )

            epilepsy = st.radio("4. โรคลมชัก", ["ไม่มี", "มี"], horizontal=True)
            epilepsy_detail = (
                st.text_input("ระบุรายละเอียดโรคลมชัก") if epilepsy == "มี" else ""
            )

            other_history = st.radio(
                "5. ประวัติอื่นที่สำคัญ",
                ["ไม่มี", "มี"],
                horizontal=True,
            )
            other_history_detail = (
                st.text_input("ระบุประวัติอื่น") if other_history == "มี" else ""
            )

            consent = st.checkbox(
                "ข้าพเจ้ารับรองว่าข้อมูลเป็นความจริง และจะลงนามที่หน้างาน"
            )

            booking_submitted = st.form_submit_button(
                "บันทึกนัดหมายและสร้าง QR",
                type="primary",
            )

        if booking_submitted:
            cid = clean_id(citizen_id)
            normalized_email = normalize_email(email)

            if len(cid) != 13:
                st.error("กรุณากรอกเลขบัตรประชาชน 13 หลัก")
                st.stop()

            if not valid_email(normalized_email):
                st.error("กรุณากรอกอีเมลให้ถูกต้อง")
                st.stop()

            if not is_workday(appt_date):
                st.error("กรุณาเลือกเฉพาะวันทำการ จันทร์–ศุกร์")
                st.stop()

            if not full_name.strip():
                st.error("กรุณากรอกชื่อ-นามสกุล")
                st.stop()

            if prefix == "อื่น ๆ" and not other_prefix.strip():
                st.error("กรุณาระบุคำนำหน้า")
                st.stop()

            if purpose == "อื่น ๆ" and not purpose_other.strip():
                st.error("กรุณาระบุวัตถุประสงค์")
                st.stop()

            if not consent:
                st.error("กรุณายืนยันว่าข้อมูลเป็นความจริง")
                st.stop()

            used_in_week = weekly_booking_count(df, cid, appt_date)
            if used_in_week >= 2:
                st.error("สัปดาห์นี้ท่านมีนัดหมายครบ 2 ครั้งแล้ว")
                st.stop()

            record_id = str(uuid.uuid4())[:8].upper()
            real_prefix = other_prefix.strip() if prefix == "อื่น ๆ" else prefix
            real_purpose = purpose_other.strip() if purpose == "อื่น ๆ" else purpose
            timestamp = now_bkk().isoformat()

            record = {
                "record_id": record_id,
                "created_at_bkk": timestamp,
                "last_modified_at_bkk": timestamp,
                "status": "booked",
                "appointment_date": appt_date.strftime("%Y-%m-%d"),
                "appointment_time": appt_time,
                "email": normalized_email,
                "confirmation_status": "รอการยืนยัน",
                "confirmation_note": "",
                "confirmation_at_bkk": "",
                "edit_count_date": "",
                "edit_count_today": "0",
                "last_appointment_edit_at_bkk": "",
                "citizen_id": cid,
                "prefix": real_prefix,
                "full_name": full_name.strip(),
                "sex": sex,
                "address": address.strip(),
                "purpose": real_purpose,
                "chronic": chronic,
                "chronic_detail": chronic_detail.strip(),
                "accident": accident,
                "accident_detail": accident_detail.strip(),
                "hospital": hospital,
                "hospital_detail": hospital_detail.strip(),
                "epilepsy": epilepsy,
                "epilepsy_detail": epilepsy_detail.strip(),
                "other_history": other_history,
                "other_history_detail": other_history_detail.strip(),
            }

            try:
                new_df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
                new_df = ensure_columns(new_df)
                save_csv(new_df, sha)
            except Exception as error:
                st.error(f"บันทึกข้อมูลไม่สำเร็จ: {error}")
                st.stop()

            st.success("บันทึกนัดหมายเรียบร้อยแล้ว")
            st.subheader("QR code สำหรับนำมาติดต่อหน้างาน")
            st.image(make_qr(record_id), width=260)
            st.code(record_id)
            st.write(
                f"วันนัดหมาย: {appt_date.strftime('%d/%m/%Y')} เวลา {appt_time} น."
            )
            st.write(f"อีเมลยืนยัน: {normalized_email}")

    with edit_tab:
        st.subheader("แก้ไขวันหรือเวลานัดหมาย")
        st.caption("ใช้รหัสรายการจากใต้ QR code พร้อมเลขบัตรประชาชนและอีเมลเดิม")

        with st.form("edit_appointment_form"):
            edit_record_id = st.text_input("รหัสรายการ")
            edit_citizen_id = st.text_input("เลขบัตรประชาชน 13 หลัก")
            edit_email = st.text_input("อีเมลที่ใช้จอง")
            new_appt_date = st.date_input(
                "วันนัดหมายใหม่",
                min_value=today,
                max_value=max_day,
                value=today,
                format="DD/MM/YYYY",
            )
            new_appt_time = st.selectbox(
                "เวลานัดหมายใหม่",
                TIME_SLOTS,
                key="new_appointment_time",
            )
            edit_submitted = st.form_submit_button(
                "ยืนยันการแก้ไขนัดหมาย",
                type="primary",
            )

        if edit_submitted:
            cid = clean_id(edit_citizen_id)
            normalized_email = normalize_email(edit_email)
            idx = find_by_record(df, edit_record_id)

            if idx is None:
                st.error("ไม่พบรหัสรายการ")
                st.stop()

            row = df.loc[idx]

            if clean_id(row.get("citizen_id", "")) != cid:
                st.error("เลขบัตรประชาชนไม่ตรงกับข้อมูลนัดหมาย")
                st.stop()

            if normalize_email(row.get("email", "")) != normalized_email:
                st.error("อีเมลไม่ตรงกับข้อมูลนัดหมาย")
                st.stop()

            if row.get("status", "") not in ACTIVE_STATUSES:
                st.error("รายการนี้ไม่อยู่ในสถานะที่แก้ไขนัดหมายได้")
                st.stop()

            if row.get("status", "") in {"doctor_approved", "printed"}:
                st.error("รายการนี้ผ่านการตรวจแพทย์แล้ว ไม่สามารถแก้ไขนัดหมายได้")
                st.stop()

            if not is_workday(new_appt_date):
                st.error("กรุณาเลือกเฉพาะวันทำการ จันทร์–ศุกร์")
                st.stop()

            if edits_used_today(df, cid) >= 2:
                st.error("วันนี้ท่านแก้ไขนัดหมายครบ 2 ครั้งแล้ว")
                st.stop()

            other_bookings = weekly_booking_count(
                df,
                cid,
                new_appt_date,
                exclude_record_id=row.get("record_id", ""),
            )
            if other_bookings >= 2:
                st.error("สัปดาห์ที่เลือกมีนัดหมายอื่นครบ 2 ครั้งแล้ว")
                st.stop()

            old_date = row.get("appointment_date", "")
            old_time = row.get("appointment_time", "")
            new_date_text = new_appt_date.strftime("%Y-%m-%d")

            if old_date == new_date_text and old_time == new_appt_time:
                st.info("วันและเวลานัดหมายยังเหมือนเดิม จึงไม่มีการแก้ไขข้อมูล")
                st.stop()

            today_text = now_bkk().date().isoformat()
            current_record_count = (
                safe_int(row.get("edit_count_today", ""), 0)
                if row.get("edit_count_date", "") == today_text
                else 0
            )
            timestamp = now_bkk().isoformat()

            df.loc[idx, "appointment_date"] = new_date_text
            df.loc[idx, "appointment_time"] = new_appt_time
            df.loc[idx, "edit_count_date"] = today_text
            df.loc[idx, "edit_count_today"] = str(current_record_count + 1)
            df.loc[idx, "last_appointment_edit_at_bkk"] = timestamp
            df.loc[idx, "last_modified_at_bkk"] = timestamp
            df.loc[idx, "confirmation_status"] = "รอการยืนยัน"
            df.loc[idx, "confirmation_note"] = "แก้ไขนัดหมายโดยผู้รับบริการ"
            df.loc[idx, "confirmation_at_bkk"] = ""

            try:
                save_csv(df, sha)
            except Exception as error:
                st.error(f"แก้ไขข้อมูลไม่สำเร็จ: {error}")
                st.stop()

            used_after = edits_used_today(df, cid)
            st.success("แก้ไขนัดหมายเรียบร้อยแล้ว")
            st.write(
                f"วันนัดหมายใหม่: {new_appt_date.strftime('%d/%m/%Y')} "
                f"เวลา {new_appt_time} น."
            )
            st.write(f"วันนี้ใช้สิทธิ์แก้ไขแล้ว {used_after}/2 ครั้ง")


# =====================================================
# 2. วัดสัญญาณชีพ
# =====================================================
elif page == "วัดสัญญาณชีพ":
    st.title("วัดสัญญาณชีพ")
    st.caption("สแกน QR code จากใบนัดหมายก่อนเริ่มวัด")

    record_id = scan_or_enter("vital")
    if not record_id:
        st.stop()

    idx = find_by_record(df, record_id)
    if idx is None:
        st.error("ไม่พบข้อมูลนัดหมาย")
        st.stop()

    row = df.loc[idx]
    if row.get("status", "") == "cancelled":
        st.error("รายการนัดหมายนี้ถูกยกเลิกแล้ว")
        st.stop()

    today_bkk = now_bkk().date().isoformat()
    if str(row.get("appointment_date", "")) != today_bkk:
        st.info("รอวัดสัญญาณชีพที่สถานพยาบาลในวันนัดหมาย")
        st.write(f"วันนัดหมาย: {display_date(row.get('appointment_date', ''))} เวลา {row.get('appointment_time', '')} น.")
        st.stop()

    st.success(f"{row.get('prefix', '')}{row.get('full_name', '')} — นัดเวลา {row.get('appointment_time', '')} น.")

    vital_image = st.camera_input("ถ่ายภาพแผ่นกระดาษที่บันทึกค่า BP และ P", key="vital_paper_camera")

    if "vital_ai_record_id" not in st.session_state or st.session_state.get("vital_ai_record_id") != record_id:
        st.session_state["vital_ai_record_id"] = record_id
        st.session_state["vital_ai_bp"] = row.get("vital_bp", "")
        st.session_state["vital_ai_pulse"] = row.get("vital_pulse", "")
        st.session_state["vital_ai_note"] = ""
        st.session_state["vital_ai_raw"] = row.get("vital_ai_raw", "")

    if vital_image is not None and st.button("ให้ AI อ่านค่าจากภาพ", type="primary"):
        try:
            with st.spinner("AI กำลังอ่านค่า BP และ P..."):
                result = extract_vitals_with_ai(vital_image)
            st.session_state["vital_ai_bp"] = result["bp"]
            st.session_state["vital_ai_pulse"] = result["pulse"]
            st.session_state["vital_ai_note"] = result["note"]
            st.session_state["vital_ai_raw"] = result["raw"]
            st.success("AI อ่านค่าแล้ว กรุณาตรวจสอบและแก้ไขก่อนบันทึก")
        except Exception as error:
            st.error(f"AI อ่านค่าไม่สำเร็จ: {error}")
            st.info("ยังสามารถกรอกค่าด้วยตนเองได้")

    if st.session_state.get("vital_ai_note"):
        st.caption(f"หมายเหตุจาก AI: {st.session_state['vital_ai_note']}")

    with st.form("vital_confirmation_form"):
        st.subheader("ตรวจสอบหรือกรอกค่าด้วยตนเอง")
        c1, c2 = st.columns(2)
        with c1:
            vital_bp = st.text_input("BP (mmHg) เช่น 120/80", value=st.session_state.get("vital_ai_bp", row.get("vital_bp", "")))
            vital_weight = st.text_input("BW (kg)", value=row.get("vital_weight", ""))
        with c2:
            vital_pulse = st.text_input("P (ครั้ง/นาที)", value=st.session_state.get("vital_ai_pulse", row.get("vital_pulse", "")))
            vital_height = st.text_input("Ht (cm)", value=row.get("vital_height", ""))

        confirmed = st.checkbox("ตรวจสอบค่าจากภาพหรือค่าที่กรอกแล้ว")
        save_vitals = st.form_submit_button("บันทึกสัญญาณชีพ", type="primary")

    if save_vitals:
        if not confirmed:
            st.error("กรุณายืนยันว่าได้ตรวจสอบค่าแล้ว")
            st.stop()
        if not valid_bp(vital_bp):
            st.error("กรุณากรอก BP ในรูป systolic/diastolic เช่น 120/80")
            st.stop()
        if not valid_positive_number(vital_pulse, 20, 250):
            st.error("กรุณาตรวจสอบค่า P")
            st.stop()
        if not valid_positive_number(vital_weight, 10, 400):
            st.error("กรุณาตรวจสอบค่า BW")
            st.stop()
        if not valid_positive_number(vital_height, 50, 250):
            st.error("กรุณาตรวจสอบค่า Ht")
            st.stop()

        timestamp = now_bkk().isoformat()
        df.loc[idx, "vital_bp"] = vital_bp.strip()
        df.loc[idx, "vital_pulse"] = vital_pulse.strip()
        df.loc[idx, "vital_weight"] = vital_weight.strip()
        df.loc[idx, "vital_height"] = vital_height.strip()
        df.loc[idx, "vital_ai_raw"] = st.session_state.get("vital_ai_raw", "")
        df.loc[idx, "vital_checked_at_bkk"] = timestamp
        df.loc[idx, "last_modified_at_bkk"] = timestamp

        try:
            save_csv(df, sha)
            st.success("บันทึกสัญญาณชีพเรียบร้อยแล้ว")
            st.write({"BP": vital_bp, "P": vital_pulse, "BW": vital_weight, "Ht": vital_height})
        except Exception as error:
            st.error(f"บันทึกข้อมูลไม่สำเร็จ: {error}")


# =====================================================
# 3. เวชระเบียน
# =====================================================
elif page == "เวชระเบียน":
    password_gate(PASS_REG, "password_registry")
    st.title("เวชระเบียนและการยืนยันนัดหมาย")

    selected_date = st.date_input(
        "แสดงรายชื่อวันที่",
        value=now_bkk().date(),
        format="DD/MM/YYYY",
    )

    selected_date_text = selected_date.strftime("%Y-%m-%d")
    appointments = df[
        df["appointment_date"].astype(str).eq(selected_date_text)
        & ~df["status"].astype(str).eq("cancelled")
    ].copy()

    if not appointments.empty:
        appointments = appointments.sort_values(
            ["appointment_time", "full_name"],
            kind="stable",
        )
        st.dataframe(
            dataframe_for_dashboard(appointments),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("ไม่พบรายชื่อนัดหมายในวันที่เลือก")

    st.markdown(
        """
        <div class="dashboard-note">
            เลือกรายชื่อจากรายการด้านล่าง หรือเปิดกล้องเพื่อสแกน QR code
        </div>
        """,
        unsafe_allow_html=True,
    )

    record_options = appointments["record_id"].astype(str).tolist()
    record_label_map = {
        str(row["record_id"]): (
            f"{row['appointment_time']} | {row['prefix']}{row['full_name']} | "
            f"{row['email']} | {row['confirmation_status']}"
        )
        for _, row in appointments.iterrows()
    }

    selected_record_id = ""
    if record_options:
        selected_record_id = st.selectbox(
            "เลือกรายการนัดหมาย",
            options=[""] + record_options,
            format_func=lambda value: "-- เลือกรายการ --" if value == "" else record_label_map[value],
        )

    with st.expander("สแกน QR หรือกรอกรหัสรายการ"):
        scanned_record_id = scan_or_enter("registry")
        if scanned_record_id:
            selected_record_id = scanned_record_id

    if selected_record_id:
        idx = find_by_record(df, selected_record_id)
        if idx is None:
            st.error("ไม่พบข้อมูล")
            st.stop()

        row = df.loc[idx]
        st.success("พบข้อมูลผู้รับบริการ")

        detail_col1, detail_col2 = st.columns(2)
        with detail_col1:
            st.write(f"**ชื่อ:** {row.get('prefix', '')}{row.get('full_name', '')}")
            st.write(f"**วันนัด:** {display_date(row.get('appointment_date', ''))}")
            st.write(f"**เวลา:** {row.get('appointment_time', '')} น.")
            st.write(f"**เลขบัตรประชาชน:** {citizen_id_display(row.get('citizen_id', ''))}")
        with detail_col2:
            email_value = row.get("email", "")
            st.write(f"**อีเมล:** {email_value}")
            st.write(f"**วัตถุประสงค์:** {row.get('purpose', '')}")
            st.write(
                f"**สถานะงาน:** {STATUS_LABELS.get(row.get('status', ''), row.get('status', ''))}"
            )
            st.write(
                f"**สถานะการยืนยัน:** {row.get('confirmation_status', '') or 'รอการยืนยัน'}"
            )

        if email_value:
            subject = quote("ยืนยันนัดหมายขอใบรับรองแพทย์")
            body = quote(
                f"เรียน คุณ{row.get('full_name', '')}\n\n"
                f"ขอยืนยันนัดหมายวันที่ {display_date(row.get('appointment_date', ''))} "
                f"เวลา {row.get('appointment_time', '')} น.\n"
                "สถานพยาบาลมหาวิทยาลัยเกษตรศาสตร์ วิทยาเขตกำแพงแสน\n\n"
                f"รหัสรายการ: {row.get('record_id', '')}"
            )
            st.markdown(
                f"[เปิดอีเมลเพื่อส่งข้อความยืนยัน](mailto:{email_value}?subject={subject}&body={body})"
            )

        with st.form("registry_confirmation_form"):
            hn = st.text_input("เลขเวชระเบียน / HN", value=row.get("hn", ""))
            registration_note = st.text_area(
                "หมายเหตุเวชระเบียน",
                value=row.get("registration_note", ""),
            )

            current_confirmation = row.get("confirmation_status", "") or "รอการยืนยัน"
            confirmation_index = (
                CONFIRMATION_OPTIONS.index(current_confirmation)
                if current_confirmation in CONFIRMATION_OPTIONS else 0
            )
            confirmation_status = st.selectbox(
                "สถานะการยืนยันกลับ",
                CONFIRMATION_OPTIONS,
                index=confirmation_index,
            )
            confirmation_note = st.text_area(
                "หมายเหตุการยืนยัน",
                value=row.get("confirmation_note", ""),
            )

            save_registry = st.form_submit_button(
                "บันทึกเวชระเบียนและสถานะการยืนยัน",
                type="primary",
            )

        if save_registry:
            timestamp = now_bkk().isoformat()
            df.loc[idx, "hn"] = hn.strip()
            df.loc[idx, "registration_note"] = registration_note.strip()
            df.loc[idx, "registered_at_bkk"] = timestamp
            df.loc[idx, "confirmation_status"] = confirmation_status
            df.loc[idx, "confirmation_note"] = confirmation_note.strip()
            df.loc[idx, "confirmation_at_bkk"] = timestamp
            df.loc[idx, "last_modified_at_bkk"] = timestamp

            if confirmation_status == "ยกเลิกนัด":
                df.loc[idx, "status"] = "cancelled"
            elif row.get("status", "") == "booked":
                df.loc[idx, "status"] = "registered"

            try:
                save_csv(df, sha)
                st.success("บันทึกเวชระเบียนและสถานะการยืนยันแล้ว")
            except Exception as error:
                st.error(f"บันทึกข้อมูลไม่สำเร็จ: {error}")


# =====================================================
# 3. ลงผลตรวจปัสสาวะ
# =====================================================
elif page == "ลงผลตรวจปัสสาวะ":
    password_gate(PASS_LAB, "password_lab")
    st.title("ลงผลตรวจปัสสาวะ Methamphetamine")

    record_id = scan_or_enter("lab")
    if not record_id:
        st.stop()

    idx = find_by_record(df, record_id)
    if idx is None:
        st.error("ไม่พบข้อมูล")
        st.stop()

    row = df.loc[idx]
    st.write(
        {
            "ชื่อ": f"{row.get('prefix', '')}{row.get('full_name', '')}",
            "เลขบัตรประชาชน": citizen_id_display(row.get("citizen_id", "")),
            "วันนัด": display_date(row.get("appointment_date", "")),
            "เวลา": row.get("appointment_time", ""),
        }
    )

    result_options = ["Negative", "Positive", "Invalid / ต้องตรวจซ้ำ"]
    current_result = row.get("urine_meth_result", "")
    result_index = result_options.index(current_result) if current_result in result_options else 0

    urine_result = st.radio(
        "ผลตรวจ Methamphetamine",
        result_options,
        index=result_index,
        horizontal=True,
    )
    urine_note = st.text_area("หมายเหตุผลตรวจ", value=row.get("urine_note", ""))

    if st.button("บันทึกผลตรวจ", type="primary"):
        timestamp = now_bkk().isoformat()
        df.loc[idx, "urine_meth_result"] = urine_result
        df.loc[idx, "urine_note"] = urine_note.strip()
        df.loc[idx, "urine_checked_at_bkk"] = timestamp
        df.loc[idx, "last_modified_at_bkk"] = timestamp
        df.loc[idx, "status"] = "lab_done"

        try:
            save_csv(df, sha)
            st.success("บันทึกผลตรวจแล้ว")
        except Exception as error:
            st.error(f"บันทึกข้อมูลไม่สำเร็จ: {error}")


# =====================================================
# 4. พยาบาล/แพทย์
# =====================================================
elif page == "พยาบาล/แพทย์":
    password_gate(PASS_DOC, "password_doctor")
    st.title("แพทย์ตรวจและอนุมัติ")

    record_id = scan_or_enter("doctor")
    if not record_id:
        st.stop()

    idx = find_by_record(df, record_id)
    if idx is None:
        st.error("ไม่พบข้อมูล")
        st.stop()

    row = df.loc[idx]
    st.write(
        {
            "ชื่อ": f"{row.get('prefix', '')}{row.get('full_name', '')}",
            "เลขบัตรประชาชน": citizen_id_display(row.get("citizen_id", "")),
            "วัตถุประสงค์": row.get("purpose", ""),
            "ผล Methamphetamine": row.get("urine_meth_result", "") or "ยังไม่มีผล",
        }
    )

    if not row.get("urine_meth_result", ""):
        st.warning("ยังไม่มีผลตรวจปัสสาวะ Methamphetamine")

    if row.get("vital_checked_at_bkk", ""):
        st.subheader("ข้อมูลจากสถานีวัดสัญญาณชีพ")
        st.write({
            "BP": row.get("vital_bp", ""),
            "P": row.get("vital_pulse", ""),
            "BW": row.get("vital_weight", ""),
            "Ht": row.get("vital_height", ""),
            "บันทึกเวลา": row.get("vital_checked_at_bkk", ""),
        })
        st.caption("แพทย์/พยาบาลกรุณาตรวจสอบ และแก้ไขในช่องด้านล่างได้")
    else:
        st.warning("ยังไม่มีข้อมูลจากสถานีวัดสัญญาณชีพ")

    doctor_names = list(DOCTORS.keys())
    current_doctor = row.get("doctor_name", "")
    doctor_index = doctor_names.index(current_doctor) if current_doctor in doctor_names else 0
    doctor_name = st.selectbox("แพทย์ผู้ตรวจ", doctor_names, index=doctor_index)
    license_no = DOCTORS[doctor_name]
    st.write(f"เลขที่ใบอนุญาต: {license_no}")

    col1, col2 = st.columns(2)
    with col1:
        weight = st.text_input("น้ำหนัก กก.", value=row.get("weight", "") or row.get("vital_weight", ""))
        bp = st.text_input("ความดันโลหิต มม.ปรอท", value=row.get("bp", "") or row.get("vital_bp", ""))
    with col2:
        height = st.text_input("ส่วนสูง ซม.", value=row.get("height", "") or row.get("vital_height", ""))
        pulse = st.text_input("ชีพจร ครั้ง/นาที", value=row.get("pulse", "") or row.get("vital_pulse", ""))

    status_options = ["ปกติ", "ผิดปกติ"]
    current_general = row.get("general_status", "")
    status_index = status_options.index(current_general) if current_general in status_options else 0
    general_status = st.radio(
        "สุขภาพทั่วไป",
        status_options,
        index=status_index,
        horizontal=True,
    )

    abnormal_detail = st.text_input(
        "ระบุความผิดปกติ",
        value=row.get("abnormal_detail", ""),
        disabled=general_status != "ผิดปกติ",
    )
    other_exam = st.text_input("อื่น ๆ ถ้ามี", value=row.get("other_exam", ""))

    default_opinion = (
        row.get("doctor_opinion", "")
        or f"คุณ{row.get('full_name', '')} มีสุขภาพแข็งแรงดี"
    )
    opinion = st.text_area("สรุปความคิดเห็นแพทย์", value=default_opinion)

    approve = st.checkbox("แพทย์ตรวจแล้วและอนุมัติให้ออกใบรับรองแพทย์")

    if st.button("Approve", type="primary"):
        if not approve:
            st.error("กรุณาติ๊กยืนยันการอนุมัติ")
            st.stop()

        timestamp = now_bkk().isoformat()
        df.loc[idx, "doctor_name"] = doctor_name
        df.loc[idx, "doctor_license"] = license_no
        df.loc[idx, "weight"] = weight.strip()
        df.loc[idx, "height"] = height.strip()
        df.loc[idx, "bp"] = bp.strip()
        df.loc[idx, "pulse"] = pulse.strip()
        df.loc[idx, "general_status"] = general_status
        df.loc[idx, "abnormal_detail"] = (
            abnormal_detail.strip() if general_status == "ผิดปกติ" else ""
        )
        df.loc[idx, "other_exam"] = other_exam.strip()
        df.loc[idx, "doctor_opinion"] = opinion.strip()
        df.loc[idx, "doctor_approved_at_bkk"] = timestamp
        df.loc[idx, "last_modified_at_bkk"] = timestamp
        df.loc[idx, "status"] = "doctor_approved"

        try:
            save_csv(df, sha)
            st.success("แพทย์อนุมัติเรียบร้อยแล้ว")
        except Exception as error:
            st.error(f"บันทึกข้อมูลไม่สำเร็จ: {error}")


# =====================================================
# 5. พิมพ์
# =====================================================
elif page == "พิมพ์":
    password_gate(PASS_PRINT, "password_print")
    st.title("พิมพ์ใบรับรองแพทย์")

    record_id = scan_or_enter("print")
    if not record_id:
        st.stop()

    idx = find_by_record(df, record_id)
    if idx is None:
        st.error("ไม่พบข้อมูล")
        st.stop()

    row = df.loc[idx].to_dict()

    if row.get("status") not in {"doctor_approved", "printed"}:
        st.warning("รายการนี้ยังไม่ผ่านการอนุมัติจากแพทย์")
        st.stop()

    # หน้าจอและ PDF ใช้ HTML template เดียวกัน
    certificate_html = build_certificate_html(row)
    st.markdown(certificate_html, unsafe_allow_html=True)

    if WEASYPRINT_AVAILABLE:
        try:
            pdf_buffer = create_certificate_pdf(row)
            st.download_button(
                label="ดาวน์โหลด PDF เพื่อพิมพ์",
                data=pdf_buffer.getvalue(),
                file_name=f"medical_certificate_{row.get('record_id', '')}.pdf",
                mime="application/pdf",
                type="primary",
            )
        except Exception as error:
            st.error(f"สร้าง PDF ไม่สำเร็จ: {error}")
    else:
        st.error("ไม่สามารถโหลด WeasyPrint จึงยังสร้าง PDF ไม่ได้")
        if WEASYPRINT_IMPORT_ERROR:
            st.code(WEASYPRINT_IMPORT_ERROR)
        st.info(
            "ตรวจสอบว่า repository มีทั้ง requirements.txt และ packages.txt "
            "จากนั้น Reboot หรือ Redeploy แอป"
        )

    if st.button("บันทึกว่าพิมพ์แล้ว"):
        timestamp = now_bkk().isoformat()
        df.loc[idx, "printed_at_bkk"] = timestamp
        df.loc[idx, "last_modified_at_bkk"] = timestamp
        df.loc[idx, "status"] = "printed"

        try:
            save_csv(df, sha)
            st.success("บันทึกสถานะพิมพ์แล้ว")
        except Exception as error:
            st.error(f"บันทึกข้อมูลไม่สำเร็จ: {error}")


