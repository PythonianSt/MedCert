# app.py
import streamlit as st
import pandas as pd
import requests, base64, uuid, qrcode
from io import StringIO, BytesIO
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import cv2
import numpy as np
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

st.set_page_config(page_title="ใบรับรองแพทย์ KU KPS", layout="wide")

BKK = ZoneInfo("Asia/Bangkok")

GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]
GITHUB_REPO = st.secrets["GITHUB_REPO"]
GITHUB_BRANCH = st.secrets.get("GITHUB_BRANCH", "main")
CSV_PATH = st.secrets.get("CSV_PATH", "medical_certificate.csv")

PASS_REG = "KUKPS01"
PASS_LAB = "KUKPS02"
PASS_DOC = "KUKPS03"
PASS_PRINT = "KUKPS04"

DOCTORS = {
    "นายแพทย์กำธร ตันติวิทยาทันต์": "12082",
    "นายแพทย์สมชาย เจนลาภวัฒนกุล": "15771",
}

TIME_SLOTS = [
    "09:00", "09:30", "10:00", "10:30",
    "13:00", "13:30", "14:00", "14:30"
]

# ---------- GitHub ----------
def headers():
    return {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json"
    }

def read_csv():
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{CSV_PATH}"
    r = requests.get(url, headers=headers(), params={"ref": GITHUB_BRANCH})
    if r.status_code == 404:
        return pd.DataFrame(), None
    r.raise_for_status()
    data = r.json()
    content = base64.b64decode(data["content"]).decode("utf-8-sig")
    return pd.read_csv(StringIO(content), dtype=str).fillna(""), data["sha"]

def ensure_columns(df):
    cols = [
        "record_id", "created_at_bkk", "status",
        "appointment_date", "appointment_time",
        "citizen_id", "prefix", "full_name", "sex", "address", "purpose",
        "chronic", "chronic_detail", "accident", "accident_detail",
        "hospital", "hospital_detail", "epilepsy", "epilepsy_detail",
        "other_history", "other_history_detail",
        "hn", "registration_note", "registered_at_bkk",
        "urine_meth_result", "urine_note", "urine_checked_at_bkk",
        "doctor_name", "doctor_license", "weight", "height", "bp", "pulse",
        "general_status", "abnormal_detail", "other_exam",
        "doctor_opinion", "doctor_approved_at_bkk",
        "printed_at_bkk"
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    return df

def save_csv(df, sha=None):
    csv_text = df.to_csv(index=False, encoding="utf-8-sig")
    encoded = base64.b64encode(csv_text.encode("utf-8-sig")).decode("utf-8")

    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{CSV_PATH}"
    payload = {
        "message": f"update certificate {datetime.now(BKK).isoformat()}",
        "content": encoded,
        "branch": GITHUB_BRANCH
    }
    if sha:
        payload["sha"] = sha

    r = requests.put(url, headers=headers(), json=payload)
    r.raise_for_status()

# ---------- Helper ----------
def now_bkk():
    return datetime.now(BKK)

def clean_id(x):
    return "".join(c for c in str(x) if c.isdigit())

def is_workday(d):
    return d.weekday() < 5

def thai_date(dt):
    months = [
        "", "มกราคม", "กุมภาพันธ์", "มีนาคม", "เมษายน", "พฤษภาคม", "มิถุนายน",
        "กรกฎาคม", "สิงหาคม", "กันยายน", "ตุลาคม", "พฤศจิกายน", "ธันวาคม"
    ]
    return f"{dt.day} {months[dt.month]} {dt.year + 543}"

def make_qr(record_id):
    img = qrcode.make(record_id)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def find_by_record(df, record_id):
    if df.empty or "record_id" not in df.columns:
        return None
    hit = df[df["record_id"].astype(str) == str(record_id)]
    if hit.empty:
        return None
    return hit.index[0]

def password_gate(correct_password):
    pw = st.sidebar.text_input("รหัสผ่าน", type="password")
    if pw != correct_password:
        st.warning("กรุณาใส่รหัสผ่านให้ถูกต้อง")
        st.stop()

def read_qr_from_image(uploaded_file):
    try:
        image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        detector = cv2.QRCodeDetector()
        data, points, _ = detector.detectAndDecode(img_bgr)

        if data:
            return data.strip()
        return ""
    except Exception:
        return ""


def scan_or_enter():
    st.subheader("สแกน QR หรือกรอกรหัสประจำตัว")

    cam = st.camera_input("ถ่ายภาพ QR code")

    qr_text = ""
    if cam is not None:
        qr_text = read_qr_from_image(cam)

        if qr_text:
            st.success("อ่าน QR code สำเร็จ")
            st.code(qr_text)
        else:
            st.warning("ยังอ่าน QR ไม่ได้ กรุณาถ่ายใหม่ให้ QR ชัด อยู่กลางภาพ และมีแสงเพียงพอ")

    manual = st.text_input("หรือกรอกรหัสจาก QR ด้วยตนเอง")

    if manual.strip():
        return manual.strip()

    return qr_text.strip()
def create_certificate_pdf(row):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    try:
        pdfmetrics.registerFont(TTFont("THSarabun", "THSarabunNew.ttf"))
        font = "THSarabun"
    except:
        font = "Helvetica"

    c.setFont(font, 18)

    y = height - 50
    line_gap = 24

    def draw(text, x=50):
        nonlocal y
        c.drawString(x, y, str(text))
        y -= line_gap

    c.setFont(font, 22)
    c.drawCentredString(width / 2, y, "ใบรับรองแพทย์")
    y -= 40

    c.setFont(font, 18)

    name_text = f"{row.get('prefix','')}{row.get('full_name','')}"
    print_date = thai_date(now_bkk())

    draw(f"สถานที่ตรวจ สถานพยาบาลมหาวิทยาลัยเกษตรศาสตร์ วิทยาเขตกำแพงแสน")
    draw(f"วันที่ {print_date}")
    draw(f"ข้าพเจ้า {row.get('doctor_name','')} ใบอนุญาตเลขที่ {row.get('doctor_license','')}")
    draw(f"ได้ตรวจร่างกาย {name_text}")
    draw(f"เลขบัตรประชาชน {row.get('citizen_id','')}")
    draw(f"ที่อยู่ {row.get('address','')}")
    draw("")
    draw(f"น้ำหนัก {row.get('weight','')} กก.  ส่วนสูง {row.get('height','')} ซม.")
    draw(f"ความดันโลหิต {row.get('bp','')} มม.ปรอท  ชีพจร {row.get('pulse','')} ครั้ง/นาที")
    draw(f"ผลตรวจ Methamphetamine: {row.get('urine_meth_result','')}")
    draw(f"สุขภาพทั่วไป: {row.get('general_status','')}")
    draw(f"รายละเอียดผิดปกติ: {row.get('abnormal_detail','')}")
    draw("")
    draw(f"สรุปความคิดเห็นแพทย์: {row.get('doctor_opinion','')}")
    y -= 50
    draw("ลงชื่อ ........................................ แพทย์ผู้ตรวจ", x=300)
    draw(f"({row.get('doctor_name','')})", x=340)

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer
# ---------- CSS ----------
st.markdown("""
<style>
@media print {
    header, footer, [data-testid="stSidebar"], .no-print {
        display: none !important;
    }
    .cert {
        font-family: "TH Sarabun New", "Sarabun", sans-serif;
        font-size: 20px;
        line-height: 1.55;
    }
}
.cert {
    font-size: 20px;
    line-height: 1.55;
}
.cert h2 {
    text-align: center;
}
.form-box {
    border: 1px solid #333;
    border-radius: 8px;
    padding: 4px 12px;
    font-weight: bold;
}
.line {
    border-bottom: 1px dotted #555;
    display: inline-block;
    min-width: 220px;
}
.longline {
    border-bottom: 1px dotted #555;
    display: inline-block;
    min-width: 520px;
}
</style>
""", unsafe_allow_html=True)

df, sha = read_csv()
df = ensure_columns(df)

# ---------- Menu ----------
st.sidebar.title("เมนู")
page = st.sidebar.radio(
    "เลือกหน้า",
    [
        "ผู้รับบริการ",
        "เวชระเบียน",
        "ลงผลตรวจปัสสาวะ",
        "พยาบาล/แพทย์",
        "พิมพ์"
    ]
)

# =====================================================
# 1. ผู้รับบริการ
# =====================================================
if page == "ผู้รับบริการ":
    st.title("ระบบนัดหมายขอใบรับรองแพทย์")
    st.info("กรอกข้อมูล นัดหมาย แล้วระบบจะสร้าง QR code สำหรับนำมาติดต่อหน้างาน")

    today = now_bkk().date()
    max_day = today + timedelta(days=30)

    with st.form("patient_form"):
        st.subheader("ข้อมูลนัดหมาย")

        appt_date = st.date_input(
            "เลือกวันนัดหมาย",
            min_value=today,
            max_value=max_day,
            value=today,
            format="DD/MM/YYYY"
        )
        appt_time = st.selectbox("เลือกเวลา", TIME_SLOTS)

        st.subheader("ส่วนที่ 1 ของผู้ขอรับใบรับรองสุขภาพ")

        citizen_id = st.text_input("เลขบัตรประชาชน 13 หลัก")
        prefix = st.selectbox("คำนำหน้า", ["นาย", "นาง", "นางสาว", "อื่น ๆ"])
        other_prefix = st.text_input("ระบุคำนำหน้า") if prefix == "อื่น ๆ" else ""
        full_name = st.text_input("ชื่อ-นามสกุล")
        sex = st.selectbox("เพศ", ["ชาย", "หญิง", "อื่น ๆ"])
        address = st.text_area("ที่อยู่ที่ติดต่อได้")

        purpose = st.selectbox("วัตถุประสงค์", ["ใบอนุญาตขับรถ", "สมัครงาน", "สมัครเรียน", "อื่น ๆ"])
        purpose_other = st.text_input("ระบุวัตถุประสงค์อื่น") if purpose == "อื่น ๆ" else ""

        chronic = st.radio("1. โรคประจำตัว", ["ไม่มี", "มี"], horizontal=True)
        chronic_detail = st.text_input("ระบุโรคประจำตัว") if chronic == "มี" else ""

        accident = st.radio("2. อุบัติเหตุและผ่าตัด", ["ไม่มี", "มี"], horizontal=True)
        accident_detail = st.text_input("ระบุอุบัติเหตุ/ผ่าตัด") if accident == "มี" else ""

        hospital = st.radio("3. เคยเข้ารับการรักษาในโรงพยาบาล", ["ไม่มี", "มี"], horizontal=True)
        hospital_detail = st.text_input("ระบุรายละเอียด") if hospital == "มี" else ""

        epilepsy = st.radio("4. โรคลมชัก", ["ไม่มี", "มี"], horizontal=True)
        epilepsy_detail = st.text_input("ระบุรายละเอียดโรคลมชัก") if epilepsy == "มี" else ""

        other_history = st.radio("5. ประวัติอื่นที่สำคัญ", ["ไม่มี", "มี"], horizontal=True)
        other_history_detail = st.text_input("ระบุประวัติอื่น") if other_history == "มี" else ""

        consent = st.checkbox("ข้าพเจ้ารับรองว่าข้อมูลเป็นความจริง และจะลงนามที่หน้างาน")

        ok = st.form_submit_button("บันทึกนัดหมายและสร้าง QR")

    if ok:
        cid = clean_id(citizen_id)

        if len(cid) != 13:
            st.error("กรุณากรอกเลขบัตรประชาชน 13 หลัก")
            st.stop()

        if not is_workday(appt_date):
            st.error("กรุณาเลือกเฉพาะวันทำการ จันทร์-ศุกร์")
            st.stop()

        if not full_name.strip():
            st.error("กรุณากรอกชื่อ-นามสกุล")
            st.stop()

        if not consent:
            st.error("กรุณายืนยันข้อมูล")
            st.stop()

        if not df.empty and "citizen_id" in df.columns:
            existing = df[
                (df["citizen_id"].astype(str) == cid) &
                (df["status"].astype(str).isin(["booked", "registered", "lab_done", "doctor_approved"]))
            ]
            if not existing.empty:
                st.error("พบว่าท่านมีนัดหมายอยู่แล้ว ไม่สามารถเลือกมากกว่า 1 วันได้")
                st.stop()

        record_id = str(uuid.uuid4())[:8].upper()
        real_prefix = other_prefix if prefix == "อื่น ๆ" else prefix
        real_purpose = purpose_other if purpose == "อื่น ๆ" else purpose

        record = {
            "record_id": record_id,
            "created_at_bkk": now_bkk().isoformat(),
            "status": "booked",
            "appointment_date": appt_date.strftime("%Y-%m-%d"),
            "appointment_time": appt_time,

            "citizen_id": cid,
            "prefix": real_prefix,
            "full_name": full_name.strip(),
            "sex": sex,
            "address": address,
            "purpose": real_purpose,

            "chronic": chronic,
            "chronic_detail": chronic_detail,
            "accident": accident,
            "accident_detail": accident_detail,
            "hospital": hospital,
            "hospital_detail": hospital_detail,
            "epilepsy": epilepsy,
            "epilepsy_detail": epilepsy_detail,
            "other_history": other_history,
            "other_history_detail": other_history_detail,
        }

        new_df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
        save_csv(new_df, sha)

        st.success("บันทึกนัดหมายเรียบร้อยแล้ว")
        st.subheader("QR code สำหรับนำมาติดต่อหน้างาน")
        st.image(make_qr(record_id), width=260)
        st.code(record_id)
        st.write(f"วันนัดหมาย: {appt_date.strftime('%d/%m/%Y')} เวลา {appt_time} น.")

# =====================================================
# 2. เวชระเบียน
# =====================================================
elif page == "เวชระเบียน":
    password_gate(PASS_REG)
    st.title("เวชระเบียน")

    record_id = scan_or_enter()
    if not record_id:
        st.stop()

    idx = find_by_record(df, record_id)
    if idx is None:
        st.error("ไม่พบข้อมูล")
        st.stop()

    row = df.loc[idx]
    st.success("พบข้อมูลผู้รับบริการ")
    st.write(row[["appointment_date", "appointment_time", "full_name", "citizen_id", "purpose"]])

    hn = st.text_input("เลขเวชระเบียน / HN")
    reg_note = st.text_area("หมายเหตุเวชระเบียน")

    if st.button("ยืนยันเวชระเบียน"):
        df.loc[idx, "hn"] = hn
        df.loc[idx, "registration_note"] = reg_note
        df.loc[idx, "registered_at_bkk"] = now_bkk().isoformat()
        df.loc[idx, "status"] = "registered"
        save_csv(df, sha)
        st.success("บันทึกเวชระเบียนแล้ว")

# =====================================================
# 3. Lab urine
# =====================================================
elif page == "ลงผลตรวจปัสสาวะ":
    password_gate(PASS_LAB)
    st.title("ลงผลตรวจปัสสาวะ Methamphetamine")

    record_id = scan_or_enter()
    if not record_id:
        st.stop()

    idx = find_by_record(df, record_id)
    if idx is None:
        st.error("ไม่พบข้อมูล")
        st.stop()

    row = df.loc[idx]
    st.write(row[["full_name", "citizen_id", "appointment_date", "appointment_time"]])

    urine_result = st.radio(
        "ผลตรวจ Methamphetamine",
        ["Negative", "Positive", "Invalid / ต้องตรวจซ้ำ"],
        horizontal=True
    )
    urine_note = st.text_area("หมายเหตุผลตรวจ")

    if st.button("บันทึกผลตรวจ"):
        df.loc[idx, "urine_meth_result"] = urine_result
        df.loc[idx, "urine_note"] = urine_note
        df.loc[idx, "urine_checked_at_bkk"] = now_bkk().isoformat()
        df.loc[idx, "status"] = "lab_done"
        save_csv(df, sha)
        st.success("บันทึกผลตรวจแล้ว")

# =====================================================
# 4. Doctor approve
# =====================================================
elif page == "พยาบาล/แพทย์":
    password_gate(PASS_DOC)
    st.title("แพทย์ตรวจและ Approve")

    record_id = scan_or_enter()
    if not record_id:
        st.stop()

    idx = find_by_record(df, record_id)
    if idx is None:
        st.error("ไม่พบข้อมูล")
        st.stop()

    row = df.loc[idx]
    show_cols = ["full_name", "citizen_id", "purpose", "urine_meth_result"]
    available_cols = [c for c in show_cols if c in row.index]
    
    st.write(row[available_cols])
    
    if "urine_meth_result" not in row.index or row.get("urine_meth_result", "") == "":
        st.warning("ยังไม่มีผลตรวจปัสสาวะ Methamphetamine")

    doctor_name = st.selectbox("แพทย์ผู้ตรวจ", list(DOCTORS.keys()))
    license_no = DOCTORS[doctor_name]
    st.write(f"เลขที่ใบอนุญาต: {license_no}")

    weight = st.text_input("น้ำหนัก กก.")
    height = st.text_input("ส่วนสูง ซม.")
    bp = st.text_input("ความดันโลหิต มม.ปรอท")
    pulse = st.text_input("ชีพจร ครั้ง/นาที")

    general_status = st.radio("สุขภาพทั่วไป", ["ปกติ", "ผิดปกติ"], horizontal=True)
    abnormal_detail = st.text_input("ระบุความผิดปกติ") if general_status == "ผิดปกติ" else ""

    other_exam = st.text_input("อื่น ๆ ถ้ามี")
    opinion = st.text_area(
        "สรุปความคิดเห็นแพทย์",
        value=f"คุณ{row.get('full_name','')} มีสุขภาพแข็งแรงดี"
    )

    approve = st.checkbox("แพทย์ตรวจแล้วและอนุมัติให้ออกใบรับรองแพทย์")

    if st.button("Approve"):
        if not approve:
            st.error("กรุณาติ๊กยืนยันการอนุมัติ")
            st.stop()

        df.loc[idx, "doctor_name"] = doctor_name
        df.loc[idx, "doctor_license"] = license_no
        df.loc[idx, "weight"] = weight
        df.loc[idx, "height"] = height
        df.loc[idx, "bp"] = bp
        df.loc[idx, "pulse"] = pulse
        df.loc[idx, "general_status"] = general_status
        df.loc[idx, "abnormal_detail"] = abnormal_detail
        df.loc[idx, "other_exam"] = other_exam
        df.loc[idx, "doctor_opinion"] = opinion
        df.loc[idx, "doctor_approved_at_bkk"] = now_bkk().isoformat()
        df.loc[idx, "status"] = "doctor_approved"

        save_csv(df, sha)
        st.success("แพทย์ Approve เรียบร้อยแล้ว")

# =====================================================
# 5. Print
# =====================================================
elif page == "พิมพ์":
    password_gate(PASS_PRINT)
    st.title("พิมพ์ใบรับรองแพทย์")

    record_id = scan_or_enter()
    if not record_id:
        st.stop()

    idx = find_by_record(df, record_id)
    if idx is None:
        st.error("ไม่พบข้อมูล")
        st.stop()

    row = df.loc[idx].to_dict()

    if row.get("status") != "doctor_approved":
        st.warning("รายการนี้ยังไม่ผ่านการ Approve จากแพทย์")
        st.stop()

    print_dt = now_bkk()
    print_date = thai_date(print_dt)

    name_text = f"{row.get('prefix','')}{row.get('full_name','')}"
    cid = row.get("citizen_id", "")
    cid_display = f"{cid[0:1]}-{cid[1:5]}-{cid[5:10]}-{cid[10:12]}-{cid[12:13]}" if len(cid) == 13 else cid

    def checked(value, target):
        return "☑" if value == target else "☐"

    html = f"""
    <div class="cert">
    <h2>ใบรับรองแพทย์ (สำหรับใบอนุญาตขับรถ)</h2>

    <p style="text-align:right;">เลขที่ <span class="line">&nbsp;</span></p>

    <p><span class="form-box">ส่วนที่ 1</span> <b>ของผู้ขอรับใบรับรองสุขภาพ</b></p>
    <p>ข้าพเจ้า {name_text}</p>
    <p>สถานที่อยู่ (ที่สามารถติดต่อได้) <span class="longline">{row.get('address','')}</span></p>
    <p>หมายเลขบัตรประชาชน <span class="line">{cid_display}</span></p>

    <p>ข้าพเจ้าขอใบรับรองสุขภาพ โดยมีประวัติสุขภาพดังนี้</p>

    <p>1. โรคประจำตัว {checked(row.get('chronic'), 'ไม่มี')} ไม่มี {checked(row.get('chronic'), 'มี')} มี (ระบุ)
    <span class="line">{row.get('chronic_detail','')}</span></p>

    <p>2. อุบัติเหตุ และผ่าตัด {checked(row.get('accident'), 'ไม่มี')} ไม่มี {checked(row.get('accident'), 'มี')} มี (ระบุ)
    <span class="line">{row.get('accident_detail','')}</span></p>

    <p>3. เคยเข้ารับการรักษาในโรงพยาบาล {checked(row.get('hospital'), 'ไม่มี')} ไม่มี {checked(row.get('hospital'), 'มี')} มี (ระบุ)
    <span class="line">{row.get('hospital_detail','')}</span></p>

    <p>4. โรคลมชัก * {checked(row.get('epilepsy'), 'ไม่มี')} ไม่มี {checked(row.get('epilepsy'), 'มี')} มี (ระบุ)
    <span class="line">{row.get('epilepsy_detail','')}</span></p>

    <p>5. ประวัติอื่นที่สำคัญ {checked(row.get('other_history'), 'ไม่มี')} ไม่มี {checked(row.get('other_history'), 'มี')} มี (ระบุ)
    <span class="line">{row.get('other_history_detail','')}</span></p>

    <p>ลงชื่อ <span class="line">&nbsp;</span> วันที่ <span class="line">&nbsp;</span></p>

    <br>

    <p><span class="form-box">ส่วนที่ 2</span> <b>ของแพทย์</b></p>

    <p>สถานที่ตรวจ สถานพยาบาลมหาวิทยาลัยเกษตรศาสตร์ วิทยาเขตกำแพงแสน วันที่ {print_date}</p>

    <p>
    ข้าพเจ้า {row.get('doctor_name','')} ใบอนุญาตประกอบวิชาชีพเวชกรรมเลขที่ {row.get('doctor_license','')}
    สถานพยาบาลชื่อ สถานพยาบาลมหาวิทยาลัยเกษตรศาสตร์ วิทยาเขตกำแพงแสน
    ที่อยู่ เลขที่ 1 หมู่ที่ 6 ตำบลกำแพงแสน อำเภอกำแพงแสน จังหวัดนครปฐม 73140
    </p>

    <p>ได้ตรวจร่างกาย {name_text} แล้วเมื่อวันที่ {print_date} มีรายละเอียดดังนี้</p>

    <p>
    น้ำหนักตัว <span class="line">{row.get('weight','')}</span> กก.
    ความสูง <span class="line">{row.get('height','')}</span> ซม.
    ความดันโลหิต <span class="line">{row.get('bp','')}</span> มม.ปรอท
    ชีพจร <span class="line">{row.get('pulse','')}</span> ครั้ง/นาที
    </p>

    <p>
    สุขภาพร่างกายทั่วไปอยู่ในเกณฑ์
    {checked(row.get('general_status'), 'ปกติ')} ปกติ
    {checked(row.get('general_status'), 'ผิดปกติ')} ผิดปกติ (ระบุ)
    <span class="longline">{row.get('abnormal_detail','')}</span>
    </p>

    <p>
    ขอรับรองว่า บุคคลดังกล่าว ไม่เป็นผู้มีร่างกายทุพพลภาพจนไม่สามารถปฏิบัติหน้าที่ได้
    ไม่ปรากฏอาการของโรคจิต หรือจิตฟั่นเฟือน หรือปัญญาอ่อน
    ไม่ปรากฏอาการของการติดยาเสพติดให้โทษ และอาการของโรคพิษสุราเรื้อรัง
    และไม่ปรากฏอาการและอาการแสดงของโรคต่อไปนี้
    </p>

    <ol>
    <li>โรคเรื้อนในระยะติดต่อ หรือในระยะที่ปรากฏอาการเป็นที่รังเกียจแก่สังคม</li>
    <li>วัณโรคในระยะอันตราย</li>
    <li>โรคเท้าช้างในระยะที่ปรากฏอาการเป็นที่รังเกียจแก่สังคม</li>
    <li>อื่น ๆ ถ้ามี <span class="longline">{row.get('other_exam','')}</span></li>
    </ol>

    <p>สรุปความคิดเห็นและข้อแนะนำของแพทย์ <span class="longline">{row.get('doctor_opinion','')}</span></p>

    <br>

    <p style="text-align:right;">ลงชื่อ <span class="line">&nbsp;</span> แพทย์ผู้ตรวจ</p>
    <p style="text-align:right;">({row.get('doctor_name','')})</p>

    <p>หมายเหตุ: ประทับตราสถานพยาบาลหลังพิมพ์เอกสาร</p>
    </div>
    """

    st.markdown(html, unsafe_allow_html=True)

   pdf_buffer = create_certificate_pdf(row)

    st.download_button(
        label="ดาวน์โหลด PDF เพื่อพิมพ์",
        data=pdf_buffer,
        file_name=f"medical_certificate_{row.get('record_id','')}.pdf",
        mime="application/pdf"
    )

    if st.button("บันทึกว่าพิมพ์แล้ว"):
        df.loc[idx, "printed_at_bkk"] = now_bkk().isoformat()
        df.loc[idx, "status"] = "printed"
        save_csv(df, sha)
        st.success("บันทึกสถานะพิมพ์แล้ว")
