# -*- coding: utf-8 -*-
"""
code_train.py  -  สร้างดัชนีค้นหา (index) สำหรับ Hybrid Search
    1) TF-IDF  (เขียนเองทั้งหมด: ตัดคำ -> ตัด stopword -> stemming -> TF x IDF -> เวกเตอร์)
    2) Embedding  (Sentence Transformer หลายภาษา: intfloat/multilingual-e5-small)
แล้วบันทึกทุกอย่างลงไฟล์ hybrid_model.sav  (ใช้ต่อใน code_test.py)

Input : workouts.csv  (776 โปรแกรมออกกำลังกาย)
Output: hybrid_model.sav
Run   : python3 code_train.py
"""
import math
import pickle
import re
from collections import Counter

import numpy as np
import pandas as pd

DATA_FILE = "workouts.csv"
MODEL_FILE = "hybrid_model.sav"
EMBED_MODEL = "intfloat/multilingual-e5-small"

# ---------------------------------------------------------------------------
# 1) Text preprocessing (ใช้ร่วมกันทั้งตอน train และตอนค้นหา)
# ---------------------------------------------------------------------------
EN_STOPWORDS = {
    "a", "an", "the", "and", "or", "for", "to", "of", "in", "on", "at", "with", "by", "from", "is", "are",
    "be", "was", "were", "it", "its", "this", "that", "these", "those", "who", "what", "which", "want",
    "wants", "i", "me", "my", "we", "you", "your", "can", "will", "would", "should", "about", "per",
    "each", "every", "into", "than", "then", "so", "as", "do", "does", "need", "like", "get", "some",
    "program", "programs", "workout", "workouts", "exercise", "exercises", "please", "help", "min", "x",
}
# คำไทยที่พบบ่อยแต่ไม่ช่วยแยกโปรแกรม (ลบออกก่อนตัดคำ เรียงจากยาวไปสั้น)
TH_STOPWORDS = ["ออกกำลังกาย", "โปรแกรม", "ต้องการ", "สำหรับ", "อยาก", "และ", "ที่", "ของ", "ให้", "ได้", "กับ", "ใน"]

TOKEN_RE = re.compile(r"[a-z0-9]+|[฀-๿]+")


def stem(word):
    """Stemmer อย่างง่าย (แนว Porter): ตัด suffix ภาษาอังกฤษให้เหลือรากศัพท์
    เช่น exercises/exercise -> exercis, running -> run, squats -> squat, toning/tone -> ton"""
    if len(word) <= 3 or word.isdigit():
        return word
    for suffix, repl in (("ies", "y"), ("ing", ""), ("ed", ""), ("es", ""), ("ly", ""), ("s", "")):
        if word.endswith(suffix) and not word.endswith("ss") and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)] + repl
            break
    if word.endswith("e") and len(word) > 4:                   # tone -> ton, exercise -> exercis
        word = word[:-1]
    if len(word) > 3 and word[-1] == word[-2] and word[-1] not in "lsz":   # runn -> run
        word = word[:-1]
    return word


def tokenize(text):
    """ข้อความ -> รายการ token
    - ภาษาอังกฤษ: ตัวพิมพ์เล็ก -> แยกคำ -> ตัด stopword -> stemming
    - ภาษาไทย (ไม่มีช่องว่างระหว่างคำ): ลบ stopword แล้วตัดเป็น character bigram เช่น 'ลดน้ำหนัก' -> ลด ดน น้ ...
    """
    text = str(text).lower()
    tokens = []
    for tok in TOKEN_RE.findall(text):
        if "฀" <= tok[0] <= "๿":                      # Thai run
            for sw in TH_STOPWORDS:
                tok = tok.replace(sw, " ")
            for part in tok.split():
                if len(part) == 1:
                    tokens.append(part)
                tokens += [part[i:i + 2] for i in range(len(part) - 1)]
        elif tok not in EN_STOPWORDS:
            tokens.append(stem(tok))
    return tokens


def build_document(row):
    """รวมฟิลด์ข้อความของโปรแกรมเป็นเอกสารเดียว และเพิ่มน้ำหนักฟิลด์สำคัญด้วยการทำซ้ำ"""
    return " ".join(
        [row["program_name"]] * 3 + [row["goal_en"]] * 3 + [row["goal_th"]] * 3 +
        [row["focus_areas"]] * 2 + [row["focus_areas_th"]] * 2 + [row["program_name_th"]] * 2 +
        [row["equipment_detail"]] * 2 + [row["tags"].replace("_", " ")] * 2 +
        [row["level"], row["description"], row["description_th"], row["weekly_schedule"]]
    )


def build_embedding_text(row):
    """ข้อความสำหรับ Embedding: ใช้ส่วนที่บอกความหมายของโปรแกรม (ไม่ใส่ตารางยาว ๆ)"""
    return (f"{row['program_name']}. {row['program_name_th']}. Goal: {row['goal_en']} ({row['goal_th']}). "
            f"Focus: {row['focus_areas']} ({row['focus_areas_th']}). {row['description']} {row['description_th']}")


# ---------------------------------------------------------------------------
# 2) TF-IDF (เขียนเอง)
# ---------------------------------------------------------------------------
def tfidf_vector(tokens, vocab, idf):
    """TF = จำนวนครั้งของคำ / จำนวนคำในเอกสาร,  weight = TF x IDF, แล้ว normalize ความยาวเป็น 1 (L2)"""
    vec = np.zeros(len(vocab), dtype=np.float32)
    if not tokens:
        return vec
    counts = Counter(tokens)
    for term, c in counts.items():
        j = vocab.get(term)
        if j is not None:
            vec[j] = (c / len(tokens)) * idf[j]
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def train_tfidf(docs_tokens):
    n_docs = len(docs_tokens)
    df = Counter()
    for toks in docs_tokens:
        df.update(set(toks))                                   # document frequency
    vocab = {term: i for i, term in enumerate(sorted(df))}
    # IDF = log((N + 1) / (df + 1)) + 1   คำที่หายาก -> น้ำหนักสูง
    idf = np.array([math.log((n_docs + 1) / (df[t] + 1)) + 1 for t in sorted(df)], dtype=np.float32)
    matrix = np.vstack([tfidf_vector(toks, vocab, idf) for toks in docs_tokens])
    return vocab, idf, matrix


# ---------------------------------------------------------------------------
# 3) Embedding
# ---------------------------------------------------------------------------
def train_embedding(texts):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("[!] ไม่พบ sentence-transformers  ->  ติดตั้งด้วย: pip install sentence-transformers")
        print("    ตอนนี้จะบันทึกเฉพาะ TF-IDF (code_test.py จะค้นหาด้วย TF-IDF อย่างเดียว)")
        return None
    model = SentenceTransformer(EMBED_MODEL, device="cpu")   # CPU: รันได้ทุกเครื่อง
    # e5 ต้องเติม 'passage: ' หน้าเอกสาร และ 'query: ' หน้าคำค้น
    emb = model.encode(["passage: " + t for t in texts], batch_size=32,
                       normalize_embeddings=True, show_progress_bar=True)
    return np.asarray(emb, dtype=np.float32)


if __name__ == "__main__":
    d = pd.read_csv(DATA_FILE)
    d = d.drop_duplicates(subset="program_id").dropna(subset=["program_name", "description"]).reset_index(drop=True)
    print(f"โหลดข้อมูล {len(d)} โปรแกรม จาก {DATA_FILE}")

    docs_tokens = [tokenize(build_document(r)) for _, r in d.iterrows()]
    vocab, idf, tfidf_matrix = train_tfidf(docs_tokens)
    print(f"TF-IDF: คำศัพท์ทั้งหมด {len(vocab)} คำ, เมทริกซ์ {tfidf_matrix.shape}")
    print("ตัวอย่าง token ของโปรแกรมแรก:", docs_tokens[0][:15])

    emb_matrix = train_embedding([build_embedding_text(r) for _, r in d.iterrows()])
    if emb_matrix is not None:
        print(f"Embedding: {EMBED_MODEL}, เมทริกซ์ {emb_matrix.shape}")

    save_model = {
        "data": d,
        "vocab": vocab,
        "idf": idf,
        "tfidf_matrix": tfidf_matrix,
        "embed_model_name": EMBED_MODEL,
        "emb_matrix": emb_matrix,
    }
    pickle.dump(save_model, open(MODEL_FILE, "wb"))
    print(f"บันทึกโมเดลแล้ว -> {MODEL_FILE}")
