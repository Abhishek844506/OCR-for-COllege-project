import streamlit as st
import google.generativeai as genai
import json
import re
import time
import io
import cv2
import numpy as np
from PIL import Image

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="RxScan — Prescription Reader",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Background */
.stApp {
    background: #2f85f5;
}

/* Hide streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }

/* Top banner */
.rx-banner {
    background: #1a2332;
    border-radius: 18px;
    padding: 32px 40px;
    margin-bottom: 28px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.rx-banner h1 {
    font-family: 'DM Serif Display', serif;
    color: #f7f4ef;
    font-size: 2.4rem;
    margin: 0;
    letter-spacing: -0.5px;
}
.rx-banner p {
    color: #8ca0b8;
    margin: 6px 0 0 0;
    font-size: 0.95rem;
}
.rx-pill {
    background: #2ecc87;
    color: #0a1a10;
    font-weight: 600;
    font-size: 0.8rem;
    padding: 5px 14px;
    border-radius: 20px;
    letter-spacing: 0.5px;
}

/* Upload zone */
.upload-zone {
    background: white;
    border: 2.5px dashed #c8bfae;
    border-radius: 16px;
    padding: 40px;
    text-align: center;
    transition: border-color 0.2s;
}
.upload-zone:hover { border-color: #1a2332; }

/* Card */
.rx-card {
    background: white;
    border-radius: 16px;
    padding: 24px 28px;
    margin-bottom: 16px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.rx-card h3 {
    font-family: 'DM Serif Display', serif;
    color: #1a2332;
    margin: 0 0 16px 0;
    font-size: 1.25rem;
}

/* Medicine card */
.med-card {
    background: #f7f4ef;
    border-radius: 12px;
    padding: 16px 20px;
    margin: 10px 0;
    border-left: 4px solid #2ecc87;
    position: relative;
}
.med-card.low { border-left-color: #e74c3c; }
.med-card.medium { border-left-color: #f39c12; }
.med-card.high { border-left-color: #2ecc87; }

.med-name {
    font-family: 'DM Serif Display', serif;
    font-size: 1.1rem;
    color: #1a2332;
    margin: 0 0 8px 0;
}
.med-details {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 8px;
}
.med-tag {
    background: white;
    border: 1px solid #e0dbd1;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.82rem;
    color: #4a5568;
}
.conf-badge {
    position: absolute;
    top: 14px;
    right: 16px;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 12px;
}
.conf-high   { background: #d4f5e5; color: #1a7a4a; }
.conf-medium { background: #fef3cd; color: #856404; }
.conf-low    { background: #fde8e8; color: #9b1c1c; }

/* Cart item */
.cart-item {
    background: white;
    border-radius: 12px;
    padding: 14px 18px;
    margin: 8px 0;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.cart-item-name { font-weight: 600; color: #1a2332; font-size: 0.95rem; }
.cart-item-sub  { color: #718096; font-size: 0.82rem; margin-top: 2px; }

/* Steps */
.step-row {
    display: flex;
    gap: 12px;
    margin-bottom: 28px;
}
.step-box {
    flex: 1;
    background: white;
    border-radius: 14px;
    padding: 18px 20px;
    text-align: center;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
.step-num {
    width: 36px; height: 36px;
    background: #1a2332;
    color: white;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 0.9rem;
    margin: 0 auto 10px auto;
}
.step-box p { color: #718096; font-size: 0.83rem; margin: 4px 0 0 0; }

/* Overall confidence bar */
.conf-bar-wrap { background: #e8e4dc; border-radius: 8px; height: 10px; margin: 10px 0; }
.conf-bar { height: 10px; border-radius: 8px; transition: width 0.6s ease; }

/* Divider */
.rx-divider { border: none; border-top: 1.5px solid #e8e4dc; margin: 20px 0; }

/* Instruction item */
.instr-item {
    padding: 8px 14px;
    background: #fff8ed;
    border-radius: 8px;
    margin: 6px 0;
    color: #5a3e1b;
    font-size: 0.9rem;
    border-left: 3px solid #f39c12;
}

/* Buttons */
.stButton > button {
    background: #1a2332 !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 28px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    transition: background 0.2s !important;
}
.stButton > button:hover {
    background: #2d3f56 !important;
}

/* Success button */
.success-btn > button {
    background: #2ecc87 !important;
    color: #0a1a10 !important;
}

/* Warning box */
.warn-box {
    background: #fff8ed;
    border: 1.5px solid #f39c12;
    border-radius: 12px;
    padding: 14px 18px;
    color: #856404;
    font-size: 0.9rem;
    margin: 12px 0;
}

/* Info box */
.info-box {
    background: #eff6ff;
    border: 1.5px solid #3b82f6;
    border-radius: 12px;
    padding: 14px 18px;
    color: #1e40af;
    font-size: 0.9rem;
    margin: 12px 0;
}
</style>
""", unsafe_allow_html=True)


# ── Gemini setup ───────────────────────────────────────────────
@st.cache_resource
def load_model():
    api_key = st.secrets.get("GEMINI_API_KEY", "")
    if not api_key:
        return None
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.5-flash")


# ── Helpers ────────────────────────────────────────────────────
def preprocess_image(pil_image: Image.Image) -> Image.Image:
    """Enhance contrast and denoise for better OCR."""
    img_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # Resize if too large
    h, w = img_cv.shape[:2]
    if max(h, w) > 1600:
        scale  = 1600 / max(h, w)
        img_cv = cv2.resize(img_cv, (int(w * scale), int(h * scale)))

    # CLAHE contrast on L channel
    lab        = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
    l, a, b    = cv2.split(lab)
    clahe      = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab        = cv2.merge((clahe.apply(l), a, b))
    enhanced   = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Mild denoise
    enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 5, 5, 7, 21)

    return Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))


PROMPT = """
You are a medical prescription reader. Analyze this handwritten prescription image carefully.

Return ONLY a valid JSON object — no markdown, no explanation, no ```json fences.

{
  "doctor_info": {
    "name": "doctor name or empty string",
    "qualification": "MBBS/MD etc or empty string",
    "clinic": "clinic/hospital name or empty string"
  },
  "patient_info": {
    "name": "patient name or empty string",
    "age": "age or empty string",
    "date": "prescription date or empty string"
  },
  "medicines": [
    {
      "name": "medicine name",
      "type": "Tab/Cap/Syp/Inj/Drops/Cream/other",
      "dosage": "e.g. 500mg",
      "frequency": "e.g. BD, TDS, twice daily, 1-0-1",
      "duration": "e.g. 5 days",
      "instructions": "e.g. after food",
      "confidence": "high/medium/low"
    }
  ],
  "general_instructions": ["instruction 1"],
  "diagnosis": "diagnosis if mentioned or empty string",
  "overall_confidence": "high/medium/low",
  "illegible_parts": "describe unclear parts or empty string"
}

Rules:
- ONLY return JSON, nothing else.
- If a field is unclear, best-guess it and mark confidence as low.
- OD=once daily, BD=twice daily, TDS=3x daily, QID=4x daily.
"""


def extract_from_gemini(model, image: Image.Image) -> dict | None:
    """Call Gemini API and parse JSON response."""
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=92)
    image_part = {"mime_type": "image/jpeg", "data": buf.getvalue()}

    for attempt in range(3):
        try:
            response = model.generate_content([PROMPT, image_part])
            text     = response.text.strip()
            text     = re.sub(r'^```json\s*', '', text)
            text     = re.sub(r'^```\s*',     '', text)
            text     = re.sub(r'\s*```$',     '', text).strip()
            return json.loads(text)
        except json.JSONDecodeError:
            if attempt < 2:
                time.sleep(2)
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                wait = 15 * (attempt + 1)
                st.warning(f"Rate limit hit — waiting {wait}s...")
                time.sleep(wait)
            else:
                st.error(f"Gemini error: {e}")
                return None
    return None


def confidence_color(conf: str):
    return {"high": "#2ecc87", "medium": "#f39c12", "low": "#e74c3c"}.get(conf.lower(), "#f39c12")

def confidence_pct(conf: str):
    return {"high": 92, "medium": 65, "low": 35}.get(conf.lower(), 60)

def confidence_label(conf: str):
    return {"high": "🟢 High", "medium": "🟡 Medium", "low": "🔴 Low"}.get(conf.lower(), "🟡 Medium")


# ── Session state ──────────────────────────────────────────────
if "parsed"     not in st.session_state: st.session_state.parsed     = None
if "cart"       not in st.session_state: st.session_state.cart       = []
if "cart_open"  not in st.session_state: st.session_state.cart_open  = False
if "analyzed"   not in st.session_state: st.session_state.analyzed   = False


# ── Banner ─────────────────────────────────────────────────────
st.markdown("""
<div class="rx-banner">
    <div>
        <h1>💊 RxScan</h1>
        <p>Upload a handwritten prescription — medicines are added to cart automatically</p>
    </div>
    <span class="rx-pill"></span>
</div>
""", unsafe_allow_html=True)

# ── Steps row ─────────────────────────────────────────────────
st.markdown("""
<div class="step-row">
    <div class="step-box">
        <div class="step-num">1</div>
        <strong>Upload</strong>
        <p>Photo or scan of your prescription</p>
    </div>
    <div class="step-box">
        <div class="step-num">2</div>
        <strong>Review</strong>
        <p>Verify detected medicines & dosage</p>
    </div>
    <div class="step-box">
        <div class="step-num">3</div>
        <strong>Cart</strong>
        <p>Confirm and proceed to order</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ── API key check ──────────────────────────────────────────────
model = load_model()
if model is None:
    st.markdown("""
    <div class="warn-box">
        ⚠️ <b>Gemini API key not configured.</b><br>
        Add <code>GEMINI_API_KEY = "your_key"</code> to your Streamlit secrets
        (<code>.streamlit/secrets.toml</code> locally or the Secrets tab on Streamlit Cloud).
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Main layout ────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

# ════════════════ LEFT COLUMN — Upload ════════════════════════
with col_left:
    st.markdown('<div class="rx-card"><h3>📷 Upload Prescription</h3>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed"
    )

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Your prescription", use_container_width=True)

        st.markdown("<hr class='rx-divider'>", unsafe_allow_html=True)

        tips_expander = st.expander("📌 Tips for best results")
        with tips_expander:
            st.markdown("""
            - 📸 Take photo in **good lighting** (no shadows)
            - 🔲 Keep prescription **flat**, not curved
            - 🔍 Make sure text is **in focus**
            - ↕️ Capture the **full page** including header
            """)

        if st.button("🔍 Analyze Prescription", use_container_width=True):
            with st.spinner("Enhancing image & reading prescription..."):
                enhanced = preprocess_image(image)
                result   = extract_from_gemini(model, enhanced)

            if result:
                st.session_state.parsed    = result
                st.session_state.analyzed  = True
                # Pre-populate cart with high/medium confidence medicines
                st.session_state.cart = [
                    {
                        "name":         f"{m.get('type','')} {m.get('name','')}".strip(),
                        "dosage":       m.get("dosage", ""),
                        "frequency":    m.get("frequency", ""),
                        "duration":     m.get("duration", ""),
                        "instructions": m.get("instructions", ""),
                        "confidence":   m.get("confidence", "medium"),
                        "selected":     m.get("confidence", "medium") != "low",
                    }
                    for m in result.get("medicines", [])
                ]
                st.success("✅ Prescription analyzed!")
                st.rerun()
            else:
                st.error("❌ Could not extract data. Try a clearer image.")

    else:
        st.markdown("""
        <div class="upload-zone">
            <div style="font-size:2.5rem;">📋</div>
            <p style="color:#6b7280; margin:8px 0 0 0;">
                Drag & drop or click <b>Browse files</b> above
            </p>
            <p style="color:#9ca3af; font-size:0.82rem; margin-top:4px;">
                JPG, PNG, WEBP supported
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ════════════════ RIGHT COLUMN — Results ══════════════════════
with col_right:
    if not st.session_state.analyzed or not st.session_state.parsed:
        st.markdown("""
        <div class="rx-card" style="min-height:300px; display:flex;
             flex-direction:column; align-items:center; justify-content:center; text-align:center;">
            <div style="font-size:3rem;">🔬</div>
            <h3 style="color:#9ca3af; font-weight:400; margin-top:12px;">
                Upload a prescription to see results
            </h3>
            <p style="color:#d1d5db; font-size:0.88rem;">
                Medicines, dosage, and instructions will appear here
            </p>
        </div>
        """, unsafe_allow_html=True)

    else:
        data = st.session_state.parsed

        # ── Confidence bar ─────────────────────────────────────
        overall = data.get("overall_confidence", "medium")
        pct     = confidence_pct(overall)
        color   = confidence_color(overall)
        label   = confidence_label(overall)

        st.markdown(f"""
        <div class="rx-card">
            <h3>📊 Overall Confidence</h3>
            <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                <span style="color:#4a5568; font-size:0.9rem;">{label}</span>
                <span style="font-weight:700; color:{color};">{pct}%</span>
            </div>
            <div class="conf-bar-wrap">
                <div class="conf-bar" style="width:{pct}%; background:{color};"></div>
            </div>
            {'<div class="warn-box">⚠️ Unclear parts: ' + data.get("illegible_parts","") + '</div>'
             if data.get("illegible_parts") else ''}
        </div>
        """, unsafe_allow_html=True)

        # ── Doctor / Patient row ───────────────────────────────
        doc = data.get("doctor_info",  {})
        pat = data.get("patient_info", {})

        dc1, dc2 = st.columns(2)
        with dc1:
            st.markdown(f"""
            <div class="rx-card" style="margin-bottom:0;">
                <h3>👨‍⚕️ Doctor</h3>
                <p style="margin:4px 0; color:#374151;"><b>{doc.get('name','—')}</b></p>
                <p style="margin:2px 0; color:#6b7280; font-size:0.85rem;">{doc.get('qualification','')}</p>
                <p style="margin:2px 0; color:#6b7280; font-size:0.85rem;">{doc.get('clinic','')}</p>
            </div>""", unsafe_allow_html=True)
        with dc2:
            st.markdown(f"""
            <div class="rx-card" style="margin-bottom:0;">
                <h3>🧑 Patient</h3>
                <p style="margin:4px 0; color:#374151;"><b>{pat.get('name','—')}</b></p>
                <p style="margin:2px 0; color:#6b7280; font-size:0.85rem;">Age: {pat.get('age','—')}</p>
                <p style="margin:2px 0; color:#6b7280; font-size:0.85rem;">Date: {pat.get('date','—')}</p>
            </div>""", unsafe_allow_html=True)

        if data.get("diagnosis"):
            st.markdown(f"""
            <div class="info-box" style="margin-top:12px;">
                🩺 <b>Diagnosis:</b> {data['diagnosis']}
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Medicines ──────────────────────────────────────────
        medicines = data.get("medicines", [])
        st.markdown('<div class="rx-card"><h3>💊 Detected Medicines</h3>', unsafe_allow_html=True)

        if medicines:
            if overall == "low":
                st.markdown("""
                <div class="warn-box">
                ⚠️ Low confidence — please carefully verify each medicine name and dosage
                before confirming your cart.
                </div>""", unsafe_allow_html=True)

            for i, med in enumerate(medicines):
                conf       = med.get("confidence", "medium").lower()
                conf_color = confidence_color(conf)
                conf_lbl   = confidence_label(conf)

                tags = ""
                if med.get("dosage"):       tags += f'<span class="med-tag">📏 {med["dosage"]}</span>'
                if med.get("frequency"):    tags += f'<span class="med-tag">🕐 {med["frequency"]}</span>'
                if med.get("duration"):     tags += f'<span class="med-tag">📅 {med["duration"]}</span>'
                if med.get("instructions"): tags += f'<span class="med-tag">ℹ️ {med["instructions"]}</span>'

                st.markdown(f"""
                <div class="med-card {conf}">
                    <span class="conf-badge conf-{conf}">{conf_lbl}</span>
                    <p class="med-name">{i+1}. {med.get('type','')} {med.get('name','Unknown')}</p>
                    <div class="med-details">{tags}</div>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="warn-box">No medicines detected. Try uploading a clearer image.</div>
            """, unsafe_allow_html=True)

        # ── Instructions ───────────────────────────────────────
        instructions = data.get("general_instructions", [])
        if instructions:
            st.markdown("<hr class='rx-divider'>", unsafe_allow_html=True)
            st.markdown("<b style='color:#1a2332;'>📋 Doctor's Instructions</b>", unsafe_allow_html=True)
            for instr in instructions:
                st.markdown(f'<div class="instr-item">• {instr}</div>', unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)


# ════════════════ CART SECTION (full width) ══════════════════
if st.session_state.analyzed and st.session_state.cart:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")

    cart_col, _ = st.columns([2, 1])
    with cart_col:
        selected_count = sum(1 for i in st.session_state.cart if i.get("selected", True))
        st.markdown(f"""
        <div class="rx-card">
            <h3>🛒 Cart Review &nbsp;
                <span style="background:#1a2332; color:white; border-radius:50%;
                             padding:2px 9px; font-size:0.85rem;">{selected_count}</span>
            </h3>
            <p style="color:#6b7280; font-size:0.88rem; margin-top:-8px;">
                Uncheck any medicine you want to remove before ordering.
            </p>
        </div>
        """, unsafe_allow_html=True)

        low_conf_items = [i for i in st.session_state.cart if i.get("confidence") == "low"]
        if low_conf_items:
            st.markdown("""
            <div class="warn-box">
                🔴 <b>Low confidence items</b> are unchecked by default.
                Please verify them against your prescription before adding.
            </div>""", unsafe_allow_html=True)

        for idx, item in enumerate(st.session_state.cart):
            col_chk, col_info = st.columns([0.07, 0.93])
            with col_chk:
                checked = st.checkbox(
                    "", value=item.get("selected", True),
                    key=f"cart_item_{idx}",
                    label_visibility="collapsed"
                )
                st.session_state.cart[idx]["selected"] = checked

            with col_info:
                conf       = item.get("confidence", "medium")
                conf_color = confidence_color(conf)
                sub_parts  = [p for p in [item.get("dosage"), item.get("frequency"), item.get("duration")] if p]
                sub_text   = " · ".join(sub_parts) if sub_parts else "Details unclear"

                st.markdown(f"""
                <div class="cart-item" style="opacity:{'1' if checked else '0.45'};">
                    <div>
                        <div class="cart-item-name">{item['name']}</div>
                        <div class="cart-item-sub">{sub_text}</div>
                    </div>
                    <span style="font-size:0.78rem; font-weight:600; color:{conf_color};">
                        {confidence_label(conf)}
                    </span>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        final_items = [i for i in st.session_state.cart if i.get("selected")]

        if final_items:
            bcol1, bcol2 = st.columns(2)
            with bcol1:
                st.markdown('<div class="success-btn">', unsafe_allow_html=True)
                if st.button(f"✅ Confirm {len(final_items)} item(s) → Cart", use_container_width=True):
                    st.session_state.cart_open = True
                    st.balloons()
                    st.success(f"🎉 {len(final_items)} medicine(s) added to cart! Connect your backend here.")
                    st.markdown("**Cart JSON (for your backend API):**")
                    st.json([{k: v for k, v in i.items() if k != "selected"} for i in final_items])
                st.markdown("</div>", unsafe_allow_html=True)
            with bcol2:
                if st.button("🔄 Scan New Prescription", use_container_width=True):
                    st.session_state.parsed    = None
                    st.session_state.cart      = []
                    st.session_state.analyzed  = False
                    st.session_state.cart_open = False
                    st.rerun()
        else:
            st.markdown("""
            <div class="warn-box">No medicines selected. Check at least one item above.</div>
            """, unsafe_allow_html=True)
