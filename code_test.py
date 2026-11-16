# -*- coding: utf-8 -*-
"""
code_test.py  -  ค้นหาโปรแกรมออกกำลังกายด้วย Hybrid Search (TF-IDF + Embedding)

ขั้นตอน:
    คำค้น -> (0) ตรวจจับเงื่อนไข เพศ/อายุ/ระดับ/อุปกรณ์/สถานที่/จำนวนวัน จากคำค้น
          -> (1) TF-IDF cosine similarity   (2) Embedding cosine similarity
          -> (3) Min-max normalization ให้อยู่ในช่วง 0-1
          -> (4) Hybrid score = 0.4 x TF-IDF + 0.6 x Embedding  แล้วบวกโบนัสตามเงื่อนไขที่ตรง (อายุ 0.2, เพศ 0.15, ...)
          -> (5) Top-3 โปรแกรม พร้อมตารางรายวัน

Input : hybrid_model.sav (สร้างจาก code_train.py) + คำค้นของผู้ใช้
Output: โปรแกรมที่ตรงที่สุด 3 อันดับ พร้อมตารางรายวัน
Run   : ต้องรัน code_train.py ก่อน แล้วรันไฟล์นี้ พิมพ์คำค้นได้เรื่อย ๆ  พิมพ์ q เพื่อจบ
        (ใช้ได้ทั้งใน Colab cell และ python3 code_test.py)
"""
import os
import pickle
import re
from collections import Counter

import numpy as np

MODEL_FILE = "hybrid_model.sav"
W_TFIDF, W_EMB = 0.4, 0.6
# โบนัสเมื่อโปรแกรมตรงกับเงื่อนไขในคำค้น (อายุ/เพศสำคัญที่สุดเพราะเกี่ยวกับความเหมาะสมและความปลอดภัย)
BONUS = {"age_group": 0.20, "gender": 0.15, "level": 0.10, "equipment": 0.10, "location": 0.05, "days_per_week": 0.05}
TOP_K = 3
EXAMPLE_QUERIES = [
    "ลดน้ำหนักที่บ้าน ไม่มีอุปกรณ์",
    "build muscle with dumbbells 4 days a week",
    "อยากมีหุ่นนางแบบ ผู้หญิง",
    "ปวดหลังจากนั่งทำงานนาน",
    "kettlebell swing for fat loss beginner",
    "ผู้สูงอายุ ฝึกการทรงตัว ป้องกันหกล้ม",
]

# ---------------------------------------------------------------------------
# Text preprocessing (ต้องเหมือนกับใน code_train.py ทุกอย่าง)
# ---------------------------------------------------------------------------
EN_STOPWORDS = {
    "a", "an", "the", "and", "or", "for", "to", "of", "in", "on", "at", "with", "by", "from", "is", "are",
    "be", "was", "were", "it", "its", "this", "that", "these", "those", "who", "what", "which", "want",
    "wants", "i", "me", "my", "we", "you", "your", "can", "will", "would", "should", "about", "per",
    "each", "every", "into", "than", "then", "so", "as", "do", "does", "need", "like", "get", "some",
    "program", "programs", "workout", "workouts", "exercise", "exercises", "please", "help", "min", "x",
}
TH_STOPWORDS = ["ออกกำลังกาย", "โปรแกรม", "ต้องการ", "สำหรับ", "อยาก", "และ", "ที่", "ของ", "ให้", "ได้", "กับ", "ใน"]
TOKEN_RE = re.compile(r"[a-z0-9]+|[฀-๿]+")


def stem(word):
    """ตัด suffix ภาษาอังกฤษให้เหลือรากศัพท์ เช่น running -> run, squats -> squat"""
    if len(word) <= 3 or word.isdigit():
        return word
    for suffix, repl in (("ies", "y"), ("ing", ""), ("ed", ""), ("es", ""), ("ly", ""), ("s", "")):
        if word.endswith(suffix) and not word.endswith("ss") and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)] + repl
            break
    if word.endswith("e") and len(word) > 4:
        word = word[:-1]
    if len(word) > 3 and word[-1] == word[-2] and word[-1] not in "lsz":
        word = word[:-1]
    return word


def tokenize(text):
    """อังกฤษ: lowercase -> แยกคำ -> ตัด stopword -> stemming / ไทย: ตัด stopword -> character bigram"""
    text = str(text).lower()
    tokens = []
    for tok in TOKEN_RE.findall(text):
        if "฀" <= tok[0] <= "๿":
            for sw in TH_STOPWORDS:
                tok = tok.replace(sw, " ")
            for part in tok.split():
                if len(part) == 1:
                    tokens.append(part)
                tokens += [part[i:i + 2] for i in range(len(part) - 1)]
        elif tok not in EN_STOPWORDS:
            tokens.append(stem(tok))
    return tokens


def tfidf_vector(tokens, vocab, idf):
    """TF x IDF ของคำค้น แล้ว normalize ความยาวเป็น 1"""
    vec = np.zeros(len(vocab), dtype=np.float32)
    if not tokens:
        return vec
    for term, c in Counter(tokens).items():
        j = vocab.get(term)
        if j is not None:
            vec[j] = (c / len(tokens)) * idf[j]
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def min_max(scores):
    """ปรับคะแนนให้อยู่ในช่วง 0-1:  (x - min) / (max - min)"""
    lo, hi = scores.min(), scores.max()
    return (scores - lo) / (hi - lo) if hi > lo else np.zeros_like(scores)


# ---------------------------------------------------------------------------
# โหลดโมเดล + ค้นหา
# ---------------------------------------------------------------------------
def load():
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"ไม่พบ {MODEL_FILE} -> ให้รัน code_train.py ก่อน")
    m = pickle.load(open(MODEL_FILE, "rb"))
    m["encoder"] = None
    if m["emb_matrix"] is not None:
        try:
            from sentence_transformers import SentenceTransformer
            m["encoder"] = SentenceTransformer(m["embed_model_name"], device="cpu")
        except ImportError:
            print("[!] ไม่พบ sentence-transformers -> ค้นหาด้วย TF-IDF อย่างเดียว")
    return m


# ---------------------------------------------------------------------------
# Query Understanding: ตรวจจับเงื่อนไข (เพศ อายุ ระดับ อุปกรณ์ สถานที่ จำนวนวัน) จากคำค้น
# ---------------------------------------------------------------------------
GENDER_WORDS = {
    "female": ["ผู้หญิง", "หญิง", "สาว", "female", "woman", "women", "girl", "lady"],
    "male":   ["ผู้ชาย", "ชาย", "หนุ่ม", "male", "man", "men", "boy", "guy"],
}
AGE_WORDS = {
    "teen":       ["วัยรุ่น", "นักเรียน", "teen", "teenager"],
    "adult":      ["วัยทำงาน", "คนทำงาน"],
    "middle_age": ["วัยกลางคน", "middle age", "middle-aged"],
    "senior":     ["ผู้สูงอายุ", "สูงอายุ", "สูงวัย", "คนแก่", "senior", "elderly"],
}
LEVEL_WORDS = {
    "beginner":     ["มือใหม่", "เริ่มต้น", "เพิ่งเริ่ม", "ไม่เคยเล่น", "beginner", "newbie", "novice"],
    "intermediate": ["ระดับกลาง", "intermediate"],
    "advanced":     ["ระดับสูง", "ขั้นสูง", "มือโปร", "advanced", "expert"],
}
EQUIP_WORDS = {   # คำค้น -> อุปกรณ์ในข้อมูลที่ใช้ได้
    ("bodyweight", "outdoor"): ["ไม่มีอุปกรณ์", "ไม่ใช้อุปกรณ์", "บอดี้เวท", "bodyweight", "no equipment"],
    ("dumbbells", "home_gym", "full_gym"): ["ดัมเบล", "dumbbell"],
    ("resistance_bands", "home_gym", "full_gym"): ["ยางยืด", "resistance band", "band"],
    ("kettlebell", "full_gym"): ["เคตเทิลเบล", "เคทเทิลเบล", "kettlebell"],
}
PLACE_WORDS = {
    "gym":     ["ฟิตเนส", "ยิม", "gym"],
    "home":    ["ที่บ้าน", "บ้าน", "คอนโด", "หอพัก", "home"],
    "outdoor": ["กลางแจ้ง", "สวนสาธารณะ", "ลู่วิ่ง", "outdoor", "park"],
}


def has_word(q, words):
    for w in words:
        if re.search(r"[a-z]", w):                       # อังกฤษ: ต้องเป็นคำ (men ไม่ใช่ women)
            if re.search(r"\b" + re.escape(w) + r"s?\b", q):
                return True
        elif w in q:                                      # ไทย: หา substring
            return True
    return False


def age_to_group(age):
    return ("teen" if age < 18 else "young_adult" if age < 30 else "adult" if age < 45
            else "middle_age" if age < 60 else "senior")


def parse_query(query):
    """คำค้น -> dict เงื่อนไข เช่น {'age_group': 'young_adult', 'gender': 'female'}"""
    q = query.lower()
    f = {}
    genders = [g for g, ws in GENDER_WORDS.items() if has_word(q, ws)]
    if len(genders) == 1:
        f["gender"] = genders[0]
    m = re.search(r"(?:อายุ|age|aged)\s*(\d{1,2})", q) or re.search(r"(?<!\d)(\d{1,2})\s*(?:ปี|years?|yrs?|y/?o)", q)
    if m:
        f["age_group"] = age_to_group(int(m.group(1)))
    else:
        for g, ws in AGE_WORDS.items():
            if has_word(q, ws):
                f["age_group"] = g
                break
    for lv, ws in LEVEL_WORDS.items():
        if has_word(q, ws):
            f["level"] = lv
            break
    for eq, ws in EQUIP_WORDS.items():
        if has_word(q, ws):
            f["equipment"] = eq
            break
    for pl, ws in PLACE_WORDS.items():
        if has_word(q, ws):
            f["location"] = pl
            break
    m = re.search(r"(?<!\d)([2-6])\s*(?:วัน|days?)", q)
    if m:
        f["days_per_week"] = int(m.group(1))
    return f


def condition_bonus(d, f):
    """ให้โบนัสโปรแกรมที่ตรงกับเงื่อนไขในคำค้น ตามน้ำหนักใน BONUS (เพศ 'ทุกเพศ' ได้ครึ่งหนึ่ง)"""
    bonus = np.zeros(len(d), dtype=np.float32)
    for key, val in f.items():
        if key == "gender":
            match = (d["gender"] == val).to_numpy() + 0.5 * (d["gender"] == "unisex").to_numpy()
        elif key == "equipment":
            match = d["equipment"].isin(val).to_numpy()
        else:
            match = (d[key] == val).to_numpy()
        bonus += BONUS[key] * match
    return bonus


def search(m, query, top_k=TOP_K):
    d = m["data"]

    # (0) Query Understanding: หาเงื่อนไข เพศ/อายุ/ระดับ/อุปกรณ์/สถานที่/จำนวนวัน จากคำค้น
    filters = parse_query(query)

    # (1) TF-IDF: คำค้น -> token -> เวกเตอร์ -> cosine กับทุกโปรแกรม
    q_tokens = tokenize(query)
    q_vec = tfidf_vector(q_tokens, m["vocab"], m["idf"])
    tfidf_scores = m["tfidf_matrix"] @ q_vec

    # (2) Embedding: คำค้น -> เวกเตอร์ความหมาย -> cosine กับทุกโปรแกรม
    if m["encoder"] is not None:
        q_emb = m["encoder"].encode(["query: " + query], normalize_embeddings=True)[0]
        emb_scores = m["emb_matrix"] @ q_emb
        w_tfidf, w_emb = W_TFIDF, W_EMB
    else:
        emb_scores = np.zeros(len(tfidf_scores), dtype=np.float32)
        w_tfidf, w_emb = 1.0, 0.0

    # (3) Min-max normalization  (4) Hybrid score + โบนัสเงื่อนไข
    tfidf_norm, emb_norm = min_max(tfidf_scores), min_max(emb_scores)
    hybrid = w_tfidf * tfidf_norm + w_emb * emb_norm
    bonus = condition_bonus(d, filters)
    final = hybrid + bonus

    print(f"\nคำค้น: {query}")
    print(f"Token หลังตัดคำ/ตัด stopword/stemming: {q_tokens}")
    print(f"เงื่อนไขที่ตรวจพบ: {filters if filters else 'ไม่มี'}")
    for label, s in (("TF-IDF อย่างเดียว", tfidf_scores), ("Embedding อย่างเดียว", emb_scores)):
        if label.startswith("Embedding") and m["encoder"] is None:
            continue
        top = np.argsort(-s)[:top_k]
        print(f"  {label:<20}: " + " | ".join(f"{d.loc[i, 'program_id']} ({s[i]:.3f})" for i in top))

    # (5) เรียงคะแนนรวมจากมากไปน้อย เลือก Top-K
    top = np.argsort(-final)[:top_k]
    return [(int(i), float(final[i]), float(hybrid[i]), float(tfidf_norm[i]), float(emb_norm[i]), float(bonus[i]))
            for i in top]


def show(m, results):
    d = m["data"]
    for rank, (i, f, h, t, e, b) in enumerate(results, 1):
        r = d.loc[i]
        print("\n" + "=" * 90)
        print(f"อันดับ {rank}: [{r['program_id']}] {r['program_name']}")
        print(f"         {r['program_name_th']}")
        print(f"คะแนนรวม = {f:.3f}  = Hybrid {h:.3f} (TF-IDF {t:.3f}, Embedding {e:.3f}) + โบนัสเงื่อนไข {b:.2f}")
        print(f"เป้าหมาย: {r['goal_th']} | เพศ: {r['gender']} | อายุ: {r['age_range']} | ระดับ: {r['level']} | "
              f"{r['days_per_week']} วัน/สัปดาห์ ครั้งละ {r['minutes_per_session']} นาที | {r['program_weeks']} สัปดาห์")
        print(f"อุปกรณ์: {r['equipment_detail']} | สถานที่: {r['location']}")
        print(f"รายละเอียด: {r['description_th']}")
        print("ตารางรายวัน:")
        for day in r["weekly_schedule"].split(" | "):
            print("   " + day)


load_model = load()
print(f"โหลดโมเดลแล้ว: {len(load_model['data'])} โปรแกรม, TF-IDF {len(load_model['vocab'])} คำ, "
      f"Embedding = {load_model['embed_model_name'] if load_model['encoder'] else 'ไม่ใช้'}")
print("ตัวอย่างคำค้น:", " / ".join(EXAMPLE_QUERIES))

try:                                            # ใน Colab/Jupyter: ลบผลเก่าก่อนแสดงผลใหม่
    from IPython.display import clear_output
except ImportError:
    def clear_output():
        pass

while True:
    try:
        q = input("\nพิมพ์สิ่งที่ต้องการ (Enter = รันตัวอย่างทั้งหมด, q = จบ): ").strip()
    except EOFError:
        break
    if q.lower() == "q":
        print("จบการค้นหา")
        break
    clear_output()
    for query in ([q] if q else EXAMPLE_QUERIES):
        print("\n" + "#" * 90)
        show(load_model, search(load_model, query))
