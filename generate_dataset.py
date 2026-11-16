"""
Generate a synthetic workout-program dataset (~780 programs, every goal x gender x age group covered) for the
Hybrid Search (TF-IDF + Embedding) workout recommender project.

Variables: gender, age group, goal, level, days/week, minutes, equipment, location.
Output:
  workouts.csv   - one row per program (weekly schedule as text)
  workouts.json  - same data with a structured weekly schedule

Run:  python3 generate_dataset.py
"""
import csv
import json
import random
from collections import OrderedDict
from pathlib import Path

SEED = 42
PER_GROUP = 4      # จำนวนโปรแกรมต่อกลุ่ม (เป้าหมาย x เพศ x ช่วงอายุ)
OUT_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Lookup tables
# ---------------------------------------------------------------------------
GENDERS = OrderedDict([
    ("male",   ("Men", "ผู้ชาย")),
    ("female", ("Women", "ผู้หญิง")),
    ("unisex", ("Everyone", "ทุกเพศ")),
])

AGES = OrderedDict([
    ("teen",        ("13-17", "Teens", "วัยรุ่น")),
    ("young_adult", ("18-29", "Young Adults", "วัยหนุ่มสาว")),
    ("adult",       ("30-44", "Adults", "วัยทำงาน")),
    ("middle_age",  ("45-59", "Middle-Aged Adults", "วัยกลางคน")),
    ("senior",      ("60+",   "Seniors", "ผู้สูงอายุ")),
])

LEVELS = OrderedDict([
    ("beginner",     ("Beginner", "ผู้เริ่มต้น")),
    ("intermediate", ("Intermediate", "กลาง")),
    ("advanced",     ("Advanced", "สูง")),
])

# equipment key -> (label, thai, available tags, location, location_th)
EQUIPMENT = OrderedDict([
    ("bodyweight",       ("No equipment (bodyweight)", "ไม่ใช้อุปกรณ์ (บอดี้เวท)", {"bw"}, "home", "บ้าน")),
    ("dumbbells",        ("Dumbbells", "ดัมเบล", {"bw", "db"}, "home", "บ้าน")),
    ("resistance_bands", ("Resistance bands", "ยางยืดออกกำลังกาย", {"bw", "band"}, "home", "บ้าน")),
    ("kettlebell",       ("Kettlebell", "เคตเทิลเบล", {"bw", "kb"}, "home", "บ้าน")),
    ("home_gym",         ("Home gym (dumbbells, bands, pull-up bar)", "โฮมยิม (ดัมเบล ยางยืด บาร์โหน)",
                          {"bw", "db", "band", "pullbar"}, "home", "บ้าน")),
    ("full_gym",         ("Full gym (barbell, machines, cables)", "ฟิตเนส/ยิมเต็มรูปแบบ (บาร์เบล เครื่อง เคเบิล)",
                          {"bw", "db", "bb", "machine", "cable", "pullbar", "kb", "cardio_machine", "band"}, "gym", "ฟิตเนส")),
    ("outdoor",          ("No equipment, outdoor (park, track)", "กลางแจ้ง ไม่ใช้อุปกรณ์ (สวนสาธารณะ ลู่วิ่ง)",
                          {"bw", "outdoor"}, "outdoor", "กลางแจ้ง")),
])

# ---------------------------------------------------------------------------
# Exercise library: (name, muscle group, equipment tag, difficulty 1-3, high_impact)
# ---------------------------------------------------------------------------
EX = [
    # chest
    ("Push-up", "chest", "bw", 2, False), ("Incline Push-up", "chest", "bw", 1, False),
    ("Knee Push-up", "chest", "bw", 1, False), ("Wall Push-up", "chest", "bw", 1, False),
    ("Decline Push-up", "chest", "bw", 3, False), ("Dumbbell Bench Press", "chest", "db", 1, False),
    ("Dumbbell Floor Press", "chest", "db", 1, False), ("Incline Dumbbell Press", "chest", "db", 2, False),
    ("Dumbbell Fly", "chest", "db", 2, False), ("Band Chest Press", "chest", "band", 1, False),
    ("Band Chest Fly", "chest", "band", 1, False), ("Barbell Bench Press", "chest", "bb", 3, False),
    ("Incline Barbell Press", "chest", "bb", 3, False), ("Chest Press Machine", "chest", "machine", 1, False),
    ("Cable Crossover", "chest", "cable", 2, False), ("Pec Deck", "chest", "machine", 1, False),
    ("Parallel Bar Dip", "chest", "pullbar", 3, False),
    # back
    ("Superman Hold", "back", "bw", 1, False), ("Reverse Snow Angel", "back", "bw", 1, False),
    ("Inverted Row (table)", "back", "bw", 2, False), ("Prone Y-T-W Raise", "back", "bw", 1, False),
    ("Pull-up", "back", "pullbar", 3, False), ("Chin-up", "back", "pullbar", 3, False),
    ("Negative Pull-up", "back", "pullbar", 2, False), ("Dead Hang", "back", "pullbar", 1, False),
    ("One-Arm Dumbbell Row", "back", "db", 1, False), ("Dumbbell Pullover", "back", "db", 2, False),
    ("Chest-Supported Dumbbell Row", "back", "db", 2, False), ("Renegade Row", "back", "db", 3, False),
    ("Band Seated Row", "back", "band", 1, False), ("Band Pull-Apart", "back", "band", 1, False),
    ("Band Lat Pulldown", "back", "band", 1, False), ("Kettlebell Bent-Over Row", "back", "kb", 1, False),
    ("Kettlebell Gorilla Row", "back", "kb", 2, False), ("Barbell Bent-Over Row", "back", "bb", 3, False),
    ("Lat Pulldown", "back", "machine", 1, False), ("Seated Cable Row", "back", "cable", 1, False),
    ("Face Pull", "back", "cable", 1, False), ("T-Bar Row", "back", "machine", 2, False),
    ("Assisted Pull-up Machine", "back", "machine", 1, False),
    # shoulders
    ("Pike Push-up", "shoulders", "bw", 2, False), ("Wall Slide", "shoulders", "bw", 1, False),
    ("Plank Shoulder Tap", "shoulders", "bw", 1, False), ("Handstand Push-up (wall)", "shoulders", "bw", 3, False),
    ("Dumbbell Shoulder Press", "shoulders", "db", 1, False), ("Dumbbell Lateral Raise", "shoulders", "db", 1, False),
    ("Arnold Press", "shoulders", "db", 2, False), ("Dumbbell Rear Delt Fly", "shoulders", "db", 1, False),
    ("Band Overhead Press", "shoulders", "band", 1, False), ("Band Lateral Raise", "shoulders", "band", 1, False),
    ("Band Face Pull", "shoulders", "band", 1, False), ("Kettlebell Halo", "shoulders", "kb", 1, False),
    ("Kettlebell Overhead Press", "shoulders", "kb", 2, False), ("Barbell Overhead Press", "shoulders", "bb", 3, False),
    ("Machine Shoulder Press", "shoulders", "machine", 1, False), ("Cable Lateral Raise", "shoulders", "cable", 2, False),
    # biceps
    ("Dumbbell Biceps Curl", "biceps", "db", 1, False), ("Hammer Curl", "biceps", "db", 1, False),
    ("Incline Dumbbell Curl", "biceps", "db", 2, False), ("Band Biceps Curl", "biceps", "band", 1, False),
    ("Barbell Curl", "biceps", "bb", 2, False), ("Cable Curl", "biceps", "cable", 1, False),
    ("Towel Isometric Curl", "biceps", "bw", 1, False), ("Kettlebell Curl", "biceps", "kb", 1, False),
    # triceps
    ("Bench Dip", "triceps", "bw", 1, False), ("Diamond Push-up", "triceps", "bw", 2, False),
    ("Dumbbell Overhead Triceps Extension", "triceps", "db", 1, False), ("Dumbbell Kickback", "triceps", "db", 1, False),
    ("Band Triceps Pushdown", "triceps", "band", 1, False), ("Cable Triceps Pushdown", "triceps", "cable", 1, False),
    ("Barbell Skull Crusher", "triceps", "bb", 2, False), ("Close-Grip Bench Press", "triceps", "bb", 3, False),
    ("Kettlebell Overhead Extension", "triceps", "kb", 1, False),
    # quads
    ("Bodyweight Squat", "quads", "bw", 1, False), ("Sit-to-Stand (chair)", "quads", "bw", 1, False),
    ("Reverse Lunge", "quads", "bw", 1, False), ("Walking Lunge", "quads", "bw", 2, False),
    ("Bulgarian Split Squat", "quads", "bw", 2, False), ("Jump Squat", "quads", "bw", 2, True),
    ("Pistol Squat Progression", "quads", "bw", 3, False), ("Wall Sit", "quads", "bw", 1, False),
    ("Step-up", "quads", "bw", 1, False), ("Goblet Squat", "quads", "db", 1, False),
    ("Dumbbell Walking Lunge", "quads", "db", 2, False), ("Dumbbell Bulgarian Split Squat", "quads", "db", 2, False),
    ("Banded Squat", "quads", "band", 1, False), ("Kettlebell Goblet Squat", "quads", "kb", 1, False),
    ("Kettlebell Front Rack Lunge", "quads", "kb", 2, False), ("Barbell Back Squat", "quads", "bb", 3, False),
    ("Barbell Front Squat", "quads", "bb", 3, False), ("Leg Press", "quads", "machine", 1, False),
    ("Leg Extension", "quads", "machine", 1, False), ("Hack Squat", "quads", "machine", 2, False),
    # hamstrings
    ("Single-Leg Romanian Deadlift", "hamstrings", "bw", 2, False), ("Nordic Hamstring Curl", "hamstrings", "bw", 3, False),
    ("Sliding Leg Curl", "hamstrings", "bw", 2, False), ("Good Morning (bodyweight)", "hamstrings", "bw", 1, False),
    ("Dumbbell Romanian Deadlift", "hamstrings", "db", 1, False), ("Dumbbell Single-Leg RDL", "hamstrings", "db", 2, False),
    ("Band Good Morning", "hamstrings", "band", 1, False), ("Band Lying Leg Curl", "hamstrings", "band", 1, False),
    ("Kettlebell Swing", "hamstrings", "kb", 2, False), ("Kettlebell Deadlift", "hamstrings", "kb", 1, False),
    ("Barbell Deadlift", "hamstrings", "bb", 3, False), ("Barbell Romanian Deadlift", "hamstrings", "bb", 2, False),
    ("Lying Leg Curl Machine", "hamstrings", "machine", 1, False), ("Seated Leg Curl Machine", "hamstrings", "machine", 1, False),
    # glutes
    ("Glute Bridge", "glutes", "bw", 1, False), ("Single-Leg Glute Bridge", "glutes", "bw", 2, False),
    ("Donkey Kick", "glutes", "bw", 1, False), ("Fire Hydrant", "glutes", "bw", 1, False),
    ("Clamshell", "glutes", "bw", 1, False), ("Curtsy Lunge", "glutes", "bw", 2, False),
    ("Frog Pump", "glutes", "bw", 1, False), ("Side-Lying Leg Raise", "glutes", "bw", 1, False),
    ("Dumbbell Hip Thrust", "glutes", "db", 2, False), ("Dumbbell Sumo Squat", "glutes", "db", 1, False),
    ("Banded Lateral Walk", "glutes", "band", 1, False), ("Band Glute Kickback", "glutes", "band", 1, False),
    ("Banded Clamshell", "glutes", "band", 1, False), ("Banded Hip Thrust", "glutes", "band", 1, False),
    ("Kettlebell Sumo Deadlift", "glutes", "kb", 1, False), ("Barbell Hip Thrust", "glutes", "bb", 2, False),
    ("Cable Glute Kickback", "glutes", "cable", 1, False), ("Hip Abduction Machine", "glutes", "machine", 1, False),
    # calves
    ("Calf Raise", "calves", "bw", 1, False), ("Single-Leg Calf Raise", "calves", "bw", 2, False),
    ("Dumbbell Calf Raise", "calves", "db", 1, False), ("Standing Calf Raise Machine", "calves", "machine", 1, False),
    ("Seated Calf Raise Machine", "calves", "machine", 1, False),
    # core
    ("Plank", "core", "bw", 1, False), ("Side Plank", "core", "bw", 1, False), ("Dead Bug", "core", "bw", 1, False),
    ("Bird Dog", "core", "bw", 1, False), ("Crunch", "core", "bw", 1, False), ("Bicycle Crunch", "core", "bw", 1, False),
    ("Mountain Climber", "core", "bw", 2, True), ("Hollow Body Hold", "core", "bw", 2, False),
    ("Lying Leg Raise", "core", "bw", 2, False), ("V-up", "core", "bw", 3, False), ("Russian Twist", "core", "bw", 2, False),
    ("Heel Tap", "core", "bw", 1, False), ("Reverse Crunch", "core", "bw", 2, False),
    ("Hanging Knee Raise", "core", "pullbar", 2, False), ("Hanging Leg Raise", "core", "pullbar", 3, False),
    ("Pallof Press", "core", "band", 1, False), ("Band Woodchop", "core", "band", 1, False),
    ("Kettlebell Windmill", "core", "kb", 2, False), ("Kettlebell Suitcase Carry", "core", "kb", 1, False),
    ("Dumbbell Farmer's Carry", "core", "db", 1, False), ("Cable Crunch", "core", "cable", 1, False),
    ("Cable Woodchop", "core", "cable", 1, False), ("Ab Wheel Rollout", "core", "machine", 3, False),
    # power / athletic
    ("Box Jump", "power", "bw", 2, True), ("Broad Jump", "power", "bw", 2, True), ("Burpee", "power", "bw", 2, True),
    ("Skater Jump", "power", "bw", 2, True), ("Tuck Jump", "power", "bw", 3, True),
    ("Lateral Shuffle", "power", "bw", 1, False), ("Plyo Push-up", "power", "bw", 3, True),
    ("Kettlebell Swing (explosive)", "power", "kb", 2, False), ("Dumbbell Snatch", "power", "db", 3, True),
    ("Dumbbell Thruster", "power", "db", 2, False), ("Barbell Power Clean", "power", "bb", 3, True),
    ("Medicine Ball Slam", "power", "machine", 2, True), ("Sprint 20 m", "power", "outdoor", 2, True),
    ("Agility Ladder Drill", "power", "outdoor", 2, False),
]

LOW_IMPACT_HIIT = ["Step Jack", "Squat to Calf Raise", "Standing Knee Drive", "Shadow Boxing",
                   "Standing Oblique Crunch", "Low-Impact Skater", "Marching High Knees", "Inchworm Walkout"]
HIGH_IMPACT_HIIT = ["Burpee", "Jump Squat", "Mountain Climber", "Skater Jump", "Jumping Jack",
                    "High Knees", "Split Lunge Jump", "Tuck Jump", "Plank Jack", "Speed Skater"]

# cardio: (name, tag, difficulty, high_impact)
CARDIO = [
    ("Brisk Walking", "bw", 1, False), ("Jogging", "bw", 2, True), ("Stair Climbing", "bw", 2, False),
    ("Dance Cardio", "bw", 1, False), ("Shadow Boxing", "bw", 1, False), ("Jump Rope", "bw", 2, True),
    ("Low-Impact Cardio (marching, step-touch)", "bw", 1, False),
    ("Outdoor Running", "outdoor", 2, True), ("Outdoor Cycling", "outdoor", 1, False),
    ("Swimming", "outdoor", 2, False), ("Hill Walking", "outdoor", 1, False),
    ("Treadmill Incline Walk", "cardio_machine", 1, False), ("Rowing Machine", "cardio_machine", 2, False),
    ("Elliptical Trainer", "cardio_machine", 1, False), ("Stationary Bike", "cardio_machine", 1, False),
    ("Stair Climber Machine", "cardio_machine", 2, False),
]

MOBILITY = ["Cat-Cow", "Hip Flexor Stretch", "Hamstring Stretch", "Thoracic Spine Rotation",
            "World's Greatest Stretch", "90/90 Hip Switch", "Ankle Mobility Rock", "Child's Pose",
            "Doorway Chest Stretch", "Chin Tuck", "Neck Side Stretch", "Figure-4 Glute Stretch",
            "Shoulder CARs", "Hip CARs", "Foam Rolling (full body)", "Deep Squat Hold", "Open Book Stretch"]
YOGA = ["Sun Salutation A", "Downward Dog", "Warrior I", "Warrior II", "Triangle Pose", "Tree Pose",
        "Cobra Pose", "Pigeon Pose", "Bridge Pose", "Seated Forward Fold", "Low Lunge", "Boat Pose",
        "Chair Pose", "Extended Side Angle", "Supine Twist", "Happy Baby", "Legs-Up-the-Wall", "Savasana"]
BALANCE = ["Single-Leg Stand", "Heel-to-Toe Walk", "Tandem Stance", "Clock Reach", "Side Leg Raise (holding chair)",
           "Sit-to-Stand (chair)", "Marching in Place", "Weight Shift Side-to-Side", "Standing Hip Circle",
           "Back Leg Raise (holding chair)", "Toe Raise", "Heel Raise (holding chair)"]
POSTNATAL = ["Diaphragmatic Breathing", "Pelvic Floor Activation (Kegel)", "Pelvic Tilt", "Heel Slide",
             "Dead Bug (modified)", "Bird Dog", "Glute Bridge", "Side-Lying Clamshell", "Wall Push-up",
             "Band Seated Row", "Sit-to-Stand (chair)", "Cat-Cow"]
BREATHING = ["Box Breathing", "4-7-8 Breathing", "Body Scan Relaxation", "Guided Meditation"]

# ---------------------------------------------------------------------------
# Session (block) types: label, thai label, composition
# ---------------------------------------------------------------------------
BLOCKS = {
    "full_body":    ("Full Body", "ทั้งตัว", [("quads", 1), ("hamstrings", 1), ("chest", 1), ("back", 1), ("shoulders", 1), ("core", 1)]),
    "full_circuit": ("Full-Body Circuit", "เซอร์กิตทั้งตัว", [("quads", 1), ("back", 1), ("chest", 1), ("glutes", 1), ("power", 1), ("core", 1)]),
    "upper":        ("Upper Body", "ร่างกายส่วนบน", [("chest", 2), ("back", 2), ("shoulders", 1), ("biceps", 1), ("triceps", 1)]),
    "lower":        ("Lower Body", "ร่างกายส่วนล่าง", [("quads", 2), ("hamstrings", 1), ("glutes", 1), ("calves", 1), ("core", 1)]),
    "push":         ("Push (Chest/Shoulders/Triceps)", "วันดัน อก ไหล่ หลังแขน", [("chest", 2), ("shoulders", 2), ("triceps", 2)]),
    "pull":         ("Pull (Back/Biceps)", "วันดึง หลัง หน้าแขน", [("back", 3), ("biceps", 2), ("core", 1)]),
    "legs":         ("Legs", "ขา", [("quads", 2), ("hamstrings", 2), ("glutes", 1), ("calves", 1)]),
    "glutes":       ("Glutes & Hamstrings", "สะโพกและต้นขาหลัง", [("glutes", 3), ("hamstrings", 1), ("quads", 1), ("core", 1)]),
    "lower_tone":   ("Lower Body Toning", "กระชับขาและสะโพก", [("glutes", 2), ("quads", 1), ("hamstrings", 1), ("calves", 1), ("core", 1)]),
    "upper_tone":   ("Upper Body Toning", "กระชับแขนและหลัง", [("back", 2), ("shoulders", 2), ("triceps", 1), ("chest", 1)]),
    "core":         ("Core & Abs", "แกนกลางลำตัวและหน้าท้อง", [("core", 5)]),
    "power":        ("Power & Agility", "พลังและความคล่องตัว", [("power", 3), ("quads", 1), ("hamstrings", 1), ("core", 1)]),
    "posture":      ("Posture & Back Care", "ปรับท่าทางและดูแลหลัง", [("back", 2), ("glutes", 1), ("core", 2)]),
    "hiit":         ("HIIT", "คาร์ดิโอเข้มข้นแบบอินเทอร์วัล", None),
    "cardio":       ("Steady Cardio (Zone 2)", "คาร์ดิโอเบา-ปานกลาง", None),
    "long_cardio":  ("Long Endurance Cardio", "คาร์ดิโอระยะยาว", None),
    "intervals":    ("Interval Training", "อินเทอร์วัล", None),
    "mobility":     ("Mobility & Stretching", "ยืดเหยียดและเพิ่มการเคลื่อนไหวข้อต่อ", None),
    "yoga":         ("Yoga Flow", "โยคะ", None),
    "balance":      ("Balance & Fall Prevention", "การทรงตัวและป้องกันการล้ม", None),
    "postnatal":    ("Postnatal Core & Pelvic Floor", "ฟื้นฟูแกนกลางและอุ้งเชิงกราน", None),
    "relax":        ("Breathing & Relaxation", "ฝึกหายใจและผ่อนคลาย", None),
    "active_recovery": ("Active Recovery", "พักแบบเคลื่อนไหว", None),
}

# ---------------------------------------------------------------------------
# Goals
# ---------------------------------------------------------------------------
# key: en, th, verb_en, verb_th, templates{days:[blocks]}, rep scheme, allowed (genders, ages)
ALL_G = set(GENDERS)
ALL_A = set(AGES)
GOALS = OrderedDict([
    ("muscle_gain", dict(
        en="Muscle Building", th="สร้างกล้ามเนื้อ",
        verb="build muscle size and shape with progressive hypertrophy training",
        verb_th="เพิ่มขนาดกล้ามเนื้อด้วยการฝึกแบบไฮเปอร์โทรฟี่และเพิ่มน้ำหนักทีละขั้น",
        templates={3: ["full_body"] * 3, 4: ["upper", "lower", "upper", "lower"],
                   5: ["push", "pull", "legs", "upper", "lower"], 6: ["push", "pull", "legs", "push", "pull", "legs"]},
        reps="8-12", genders=ALL_G, ages=ALL_A - {"senior"})),
    ("strength", dict(
        en="Strength", th="เพิ่มความแข็งแรง",
        verb="get stronger on the main compound lifts",
        verb_th="เพิ่มความแข็งแรงของกล้ามเนื้อในท่าหลักแบบคอมพาวด์",
        templates={2: ["full_body", "full_body"], 3: ["full_body"] * 3, 4: ["upper", "lower", "upper", "lower"],
                   5: ["lower", "upper", "power", "lower", "upper"]},
        reps="4-6", genders=ALL_G, ages=ALL_A)),
    ("fat_loss", dict(
        en="Fat Loss", th="ลดไขมัน ลดน้ำหนัก",
        verb="burn fat and lose weight while keeping muscle",
        verb_th="เผาผลาญไขมันและลดน้ำหนักโดยยังรักษามวลกล้ามเนื้อ",
        templates={3: ["full_circuit", "hiit", "full_circuit"], 4: ["full_circuit", "cardio", "full_circuit", "hiit"],
                   5: ["upper", "hiit", "lower", "cardio", "full_circuit"],
                   6: ["upper", "hiit", "lower", "cardio", "full_circuit", "long_cardio"]},
        reps="12-15", genders=ALL_G, ages=ALL_A)),
    ("general_health", dict(
        en="General Health & Fitness", th="สุขภาพทั่วไป",
        verb="stay healthy, active and feel better day to day",
        verb_th="มีสุขภาพดี ร่างกายแข็งแรง และกระฉับกระเฉงในชีวิตประจำวัน",
        templates={2: ["full_body", "cardio"], 3: ["full_body", "cardio", "full_body"],
                   4: ["full_body", "cardio", "full_body", "mobility"], 5: ["full_body", "cardio", "full_body", "yoga", "cardio"]},
        reps="10-12", genders=ALL_G, ages=ALL_A)),
    ("model_physique", dict(
        en="Lean Model Physique", th="หุ่นนางแบบ/นายแบบ",
        verb="get a lean, slim and toned model-like physique",
        verb_th="มีหุ่นเพรียว สมส่วน กระชับ แบบนางแบบหรือนายแบบ",
        templates={3: ["lower_tone", "upper_tone", "hiit"], 4: ["lower_tone", "upper_tone", "cardio", "glutes"],
                   5: ["lower_tone", "upper_tone", "hiit", "glutes", "core"],
                   6: ["lower_tone", "upper_tone", "cardio", "glutes", "core", "yoga"]},
        templates_male={3: ["upper", "lower", "hiit"], 4: ["push", "pull", "legs", "hiit"],
                        5: ["push", "pull", "legs", "upper_tone", "hiit"], 6: ["push", "pull", "legs", "upper", "core", "cardio"]},
        reps="12-15", genders=ALL_G, ages={"young_adult", "adult"})),
    ("toning", dict(
        en="Body Toning", th="กระชับสัดส่วน",
        verb="tone and firm the whole body and reduce flabby areas",
        verb_th="กระชับสัดส่วนทั้งตัว ลดส่วนที่หย่อนคล้อย",
        templates={2: ["full_circuit", "full_circuit"], 3: ["lower_tone", "upper_tone", "hiit"],
                   4: ["lower_tone", "upper_tone", "hiit", "core"], 5: ["lower_tone", "upper_tone", "cardio", "glutes", "core"]},
        reps="12-15", genders=ALL_G, ages=ALL_A - {"teen"})),
    ("glute_shaping", dict(
        en="Glute & Leg Shaping", th="สร้างสะโพกและขาสวย",
        verb="build rounder glutes and shapely legs",
        verb_th="สร้างก้นกลม สะโพกสวย และขาเรียวได้รูป",
        templates={3: ["glutes", "upper_tone", "lower_tone"], 4: ["glutes", "upper_tone", "lower_tone", "cardio"],
                   5: ["glutes", "upper_tone", "lower_tone", "core", "glutes"]},
        reps="10-15", genders={"female", "unisex"}, ages=ALL_A - {"teen", "senior"})),
    ("core_abs", dict(
        en="Core & Six-Pack Abs", th="หน้าท้องแบนราบ ซิกแพ็ก",
        verb="strengthen the core and reveal flat abs or a six-pack",
        verb_th="เสริมแกนกลางลำตัวให้แข็งแรงและมีหน้าท้องแบนหรือซิกแพ็ก",
        templates={3: ["core", "full_circuit", "core"], 4: ["core", "full_body", "hiit", "core"],
                   5: ["core", "upper", "hiit", "lower", "core"]},
        reps="12-20", genders=ALL_G, ages=ALL_A - {"senior"})),
    ("endurance", dict(
        en="Cardio Endurance & Running", th="เพิ่มความอึด วิ่งได้ไกลขึ้น",
        verb="build stamina and endurance for running or long cardio",
        verb_th="เพิ่มความอึด ความทนทาน และวิ่งได้นานขึ้น",
        templates={3: ["intervals", "full_body", "long_cardio"], 4: ["intervals", "full_body", "cardio", "long_cardio"],
                   5: ["intervals", "lower", "cardio", "core", "long_cardio"],
                   6: ["intervals", "lower", "cardio", "upper", "intervals", "long_cardio"]},
        reps="15-20", genders=ALL_G, ages=ALL_A)),
    ("heart_health", dict(
        en="Heart Health", th="สุขภาพหัวใจและหลอดเลือด",
        verb="improve heart health, blood pressure and cardiovascular fitness",
        verb_th="บำรุงหัวใจ ควบคุมความดัน และเพิ่มสมรรถภาพหัวใจและปอด",
        templates={3: ["cardio", "full_body", "cardio"], 4: ["cardio", "full_body", "intervals", "cardio"],
                   5: ["cardio", "full_body", "intervals", "cardio", "mobility"]},
        reps="10-15", genders=ALL_G, ages=ALL_A - {"teen"})),
    ("flexibility", dict(
        en="Flexibility & Yoga", th="ความยืดหยุ่น โยคะ",
        verb="improve flexibility, joint mobility and range of motion",
        verb_th="เพิ่มความยืดหยุ่น ความคล่องตัวของข้อต่อ และช่วงการเคลื่อนไหว",
        templates={2: ["yoga", "mobility"], 3: ["yoga", "mobility", "yoga"], 4: ["yoga", "mobility", "yoga", "core"],
                   5: ["yoga", "mobility", "yoga", "core", "mobility"]},
        reps="hold", genders=ALL_G, ages=ALL_A)),
    ("posture_back", dict(
        en="Posture & Back Pain Relief", th="ปรับบุคลิกภาพ แก้ปวดหลัง (ออฟฟิศซินโดรม)",
        verb="fix posture and relieve neck, shoulder and back pain from sitting",
        verb_th="ปรับท่าทาง แก้ไหล่ห่อ และบรรเทาอาการปวดคอ บ่า ไหล่ หลังจากการนั่งทำงาน",
        templates={2: ["posture", "mobility"], 3: ["posture", "mobility", "posture"],
                   4: ["posture", "mobility", "posture", "cardio"], 5: ["posture", "mobility", "posture", "yoga", "cardio"]},
        reps="10-15", genders=ALL_G, ages=ALL_A)),
    ("athletic", dict(
        en="Athletic Performance", th="สมรรถภาพนักกีฬา",
        verb="improve speed, power, agility and sports performance",
        verb_th="เพิ่มความเร็ว พลังระเบิด ความคล่องแคล่ว สำหรับการเล่นกีฬา",
        templates={3: ["power", "full_body", "intervals"], 4: ["power", "upper", "lower", "intervals"],
                   5: ["power", "upper", "lower", "intervals", "hiit"], 6: ["power", "upper", "lower", "intervals", "power", "mobility"]},
        reps="5-8", genders=ALL_G, ages={"teen", "young_adult", "adult"})),
    ("weight_gain", dict(
        en="Healthy Weight Gain (Skinny to Fit)", th="เพิ่มน้ำหนัก สำหรับคนผอม",
        verb="gain healthy weight and muscle mass for naturally skinny people (hardgainers)",
        verb_th="เพิ่มน้ำหนักและมวลกล้ามเนื้ออย่างมีสุขภาพดี สำหรับคนผอมที่กินเท่าไหร่ก็ไม่อ้วน",
        templates={3: ["full_body"] * 3, 4: ["upper", "lower", "upper", "lower"], 5: ["push", "pull", "legs", "upper", "lower"]},
        reps="6-10", genders=ALL_G, ages={"teen", "young_adult", "adult", "middle_age"})),
    ("healthy_aging", dict(
        en="Healthy Aging, Balance & Mobility", th="การทรงตัวและสุขภาพวัยสูงอายุ",
        verb="improve balance, prevent falls and stay independent with strong muscles and joints",
        verb_th="เพิ่มการทรงตัว ป้องกันการหกล้ม และรักษากล้ามเนื้อกับข้อต่อให้แข็งแรง",
        templates={2: ["balance", "full_body"], 3: ["balance", "full_body", "cardio"],
                   4: ["balance", "full_body", "cardio", "mobility"], 5: ["balance", "full_body", "cardio", "mobility", "yoga"]},
        reps="8-12", genders=ALL_G, ages={"middle_age", "senior"})),
    ("stress_relief", dict(
        en="Stress Relief & Better Sleep", th="ผ่อนคลายความเครียด นอนหลับดีขึ้น",
        verb="reduce stress, relax the mind and sleep better",
        verb_th="ลดความเครียด ผ่อนคลายจิตใจ และนอนหลับได้ดีขึ้น",
        templates={2: ["yoga", "cardio"], 3: ["yoga", "cardio", "relax"], 4: ["yoga", "cardio", "mobility", "relax"],
                   5: ["yoga", "cardio", "mobility", "yoga", "relax"]},
        reps="hold", genders=ALL_G, ages=ALL_A)),
    ("postnatal", dict(
        en="Postnatal Recovery", th="ฟื้นฟูร่างกายหลังคลอด",
        verb="safely rebuild core and pelvic floor strength after giving birth (after doctor clearance)",
        verb_th="ฟื้นฟูกล้ามเนื้อแกนกลางและอุ้งเชิงกรานหลังคลอดอย่างปลอดภัย (หลังได้รับอนุญาตจากแพทย์)",
        templates={2: ["postnatal", "cardio"], 3: ["postnatal", "cardio", "postnatal"],
                   4: ["postnatal", "cardio", "postnatal", "mobility"]},
        reps="8-12", genders={"female"}, ages={"young_adult", "adult"})),
])

STRENGTH_GOALS = {"muscle_gain", "strength", "fat_loss", "toning", "model_physique", "general_health",
                  "weight_gain", "glute_shaping", "core_abs"}

DAY_SLOTS = {2: [0, 3], 3: [0, 2, 4], 4: [0, 1, 3, 4], 5: [0, 1, 2, 4, 5], 6: [0, 1, 2, 3, 4, 5]}
DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

TIME_BASED = ("Plank 3", "Side Plank", "Hold", "Wall Sit", "Dead Hang", "Carry", "Superman")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def max_difficulty(level, age):
    if age == "senior":
        return 1 if level == "beginner" else 2
    return {"beginner": 1, "intermediate": 2, "advanced": 3}[level]


def allow_high_impact(level, age):
    return age not in ("senior", "middle_age") and level != "beginner" or (age == "middle_age" and level == "advanced")


def sets_for(level, age):
    s = {"beginner": 2, "intermediate": 3, "advanced": 4}[level]
    return max(2, s - 1) if age == "senior" else s


def reps_for(goal, level, age):
    reps = GOALS[goal]["reps"]
    if goal == "strength":
        if age in ("teen", "senior") or level == "beginner":
            return "8-10" if age == "senior" else "6-8"
        return "3-5" if level == "advanced" else "4-6"
    if age == "senior" and reps not in ("hold",):
        return "8-12"
    return reps


def pick_exercises(rng, muscle, n, tags, maxd, hi_ok, used):
    mind = 2 if maxd == 3 else 1
    if maxd >= 2:  # easiest regressions only for beginners / seniors
        used = used | {"Wall Push-up", "Sit-to-Stand (chair)", "Knee Push-up"}
    pool = [e for e in EX if e[1] == muscle and e[2] in tags and mind <= e[3] <= maxd and (hi_ok or not e[4])
            and e[0] not in used]
    # prefer harder / equipment-specific moves when available (more realistic progression)
    rng.shuffle(pool)
    pool.sort(key=lambda e: (-(e[2] != "bw"), -e[3]))
    top = pool[: max(n + 2, 3)]
    rng.shuffle(top)
    chosen = top[:n]
    return chosen


def prescribe(name, sets, reps):
    if name in ("Plank", "Side Plank") or any(k in name for k in TIME_BASED[2:]):
        sec = {"2": "30s", "3": "40s", "4": "45s"}.get(str(sets), "30s")
        return f"{name} {sets}x{sec}"
    return f"{name} {sets}x{reps}"


SHORT_ONLY = {"Jump Rope", "Shadow Boxing", "Dance Cardio", "Stair Climbing", "Low-Impact Cardio (marching, step-touch)", "Stair Climber Machine"}
INTERVAL_OK = {"Brisk Walking", "Jogging", "Outdoor Running", "Outdoor Cycling", "Hill Walking", "Treadmill Incline Walk",
               "Rowing Machine", "Elliptical Trainer", "Stationary Bike", "Swimming", "Stair Climbing", "Stair Climber Machine"}


def cardio_choice(rng, tags, level, age, hi_ok, kind="steady", dur=30):
    pool = [c for c in CARDIO if c[1] in tags and (kind != "interval" or c[0] in INTERVAL_OK)
            and (dur <= 30 or c[0] not in SHORT_ONLY) and (hi_ok or not c[3]) and c[2] <= max_difficulty(level, age) + 1]
    if not pool:
        pool = [c for c in CARDIO if c[1] == "bw" and not c[3]]
    return rng.choice(pool)[0]


def build_block(rng, block, goal, gender, age, level, equip, minutes):
    tags = EQUIPMENT[equip][2]
    maxd = max_difficulty(level, age)
    hi_ok = allow_high_impact(level, age)
    sets, reps = sets_for(level, age), reps_for(goal, level, age)
    if goal == "strength" and equip in ("bodyweight", "outdoor", "resistance_bands"):
        reps = "6-10 (slow 3s lowering)"
    label, label_th, comp = BLOCKS[block]

    if comp is not None:
        comp = list(comp)
        # gender emphasis
        if gender == "female" and block in ("full_body", "lower", "legs", "full_circuit"):
            comp.append(("glutes", 1))
        if gender == "male" and block in ("upper", "full_body"):
            comp.append(("back" if rng.random() < 0.5 else "chest", 1))
        # session length scaling
        total = sum(n for _, n in comp)
        if minutes <= 30 and total > 5:
            comp = comp[:5] if len(comp) >= 5 else comp
            comp = [(m, 1) for m, _ in comp]
        elif minutes >= 75:
            comp.append(("core", 1))
        used, items = set(), []
        for muscle, n in comp:
            for ex in pick_exercises(rng, muscle, n, tags, maxd, hi_ok, used):
                used.add(ex[0])
                items.append(prescribe(ex[0], sets, reps if muscle not in ("power",) else "5-8"))
        if block == "full_circuit":
            rounds = {"beginner": 2, "intermediate": 3, "advanced": 4}[level]
            return label, label_th, [f"Circuit {rounds} rounds (45s work / 15s rest): " + ", ".join(i.rsplit(" ", 1)[0] for i in items)]
        return label, label_th, items

    work = max(15, minutes - 10)
    if block == "hiit":
        moves = rng.sample(HIGH_IMPACT_HIIT if hi_ok else LOW_IMPACT_HIIT, 5)
        on, off = {"beginner": (20, 40), "intermediate": (30, 30), "advanced": (40, 20)}[level]
        rounds = {"beginner": 3, "intermediate": 4, "advanced": 5}[level]
        return label, label_th, ["Warm-up 5 min", f"{rounds} rounds ({on}s work / {off}s rest): " + ", ".join(moves), "Cool-down stretch 5 min"]
    if block in ("cardio", "long_cardio"):
        dur = min(work, 60) if block == "cardio" else min(work + 20, 90)
        c = cardio_choice(rng, tags, level, age, hi_ok, "steady", dur)
        zone = "Zone 2, conversational pace" if block == "cardio" else "easy steady pace, build 5 min each week"
        return label, label_th, [f"{c} {dur} min ({zone})"]
    if block == "intervals":
        c = cardio_choice(rng, tags, level, age, hi_ok, "interval")
        reps_i = {"beginner": 5, "intermediate": 8, "advanced": 10}[level]
        fast = "brisk" if age == "senior" else "hard"
        return label, label_th, ["Warm-up 5 min easy", f"{c}: {reps_i} x (1 min {fast} / 2 min easy)", "Cool-down 5 min"]
    if block == "mobility":
        return label, label_th, [f"{m} 2x45s" for m in rng.sample(MOBILITY, 7)]
    if block == "yoga":
        poses = rng.sample(YOGA[:-1], 8) + ["Savasana"]
        return label, label_th, [f"{p} hold 5 breaths" for p in poses]
    if block == "balance":
        return label, label_th, [f"{b} 2x30s" if "Stand" in b or "Stance" in b else f"{b} 2x10"
                                 for b in rng.sample(BALANCE, 6)]
    if block == "postnatal":
        return label, label_th, [f"{p} 2x10" for p in rng.sample(POSTNATAL, 6)] + ["Gentle walk 10 min"]
    if block == "relax":
        return label, label_th, [f"{b} 5 min" for b in rng.sample(BREATHING, 3)] + [f"{m} 1x60s" for m in rng.sample(MOBILITY, 3)]
    if block == "active_recovery":
        return label, label_th, ["Light walk 20-30 min", "Foam Rolling (full body) 10 min"]
    raise ValueError(block)


def focus_of(blocks):
    m = []
    for b in blocks:
        comp = BLOCKS[b][2]
        if comp:
            m += [x for x, _ in comp]
        else:
            m.append({"hiit": "cardio", "cardio": "cardio", "long_cardio": "cardio", "intervals": "cardio",
                      "mobility": "mobility", "yoga": "flexibility", "balance": "balance",
                      "postnatal": "pelvic floor", "relax": "mind", "active_recovery": "recovery"}[b])
    rename = {"quads": "legs", "hamstrings": "legs", "calves": "legs", "power": "power"}
    out = []
    for x in m:
        x = rename.get(x, x)
        if x not in out:
            out.append(x)
    return out


FOCUS_TH = {"legs": "ขา", "glutes": "สะโพก/ก้น", "chest": "อก", "back": "หลัง", "shoulders": "ไหล่",
            "biceps": "ต้นแขนหน้า", "triceps": "ต้นแขนหลัง", "core": "หน้าท้อง/แกนกลาง", "cardio": "คาร์ดิโอ",
            "mobility": "การเคลื่อนไหวข้อต่อ", "flexibility": "ความยืดหยุ่น", "balance": "การทรงตัว",
            "pelvic floor": "อุ้งเชิงกราน", "mind": "จิตใจ", "recovery": "การฟื้นฟู", "power": "พลังระเบิด"}


def minutes_for(goal, level, equip, age, rng):
    base = {"beginner": [20, 30, 30, 40], "intermediate": [30, 40, 45, 50], "advanced": [45, 60, 60, 75]}[level]
    if equip == "full_gym" and goal in ("muscle_gain", "strength", "weight_gain"):
        base = [m + 15 for m in base]
    if goal in ("flexibility", "stress_relief", "postnatal", "posture_back"):
        base = [20, 25, 30, 40]
    if age == "senior":
        base = [min(m, 45) for m in base]
    return rng.choice(base)


INTENSITY = {"beginner": "low", "intermediate": "moderate", "advanced": "high"}
INTENSITY_TH = {"low": "เบา", "moderate": "ปานกลาง", "high": "หนัก"}


def make_program(pid, goal, gender, age, level, equip, days, rng):
    g = GOALS[goal]
    templates = g.get("templates_male") if (goal == "model_physique" and gender == "male") else g["templates"]
    blocks = list(templates[days])
    if age == "senior":
        blocks = [{"hiit": "cardio", "power": "balance", "intervals": "cardio"}.get(b, b) for b in blocks]
    minutes = minutes_for(goal, level, equip, age, rng)
    weeks = rng.choice([4, 6, 8, 12] if level != "beginner" else [4, 6, 8])

    schedule, slots = [], DAY_SLOTS[days]
    bi = 0
    for d in range(7):
        if d in slots:
            label, label_th, items = build_block(rng, blocks[bi], goal, gender, age, level, equip, minutes)
            schedule.append({"day": DAY_NAMES[d], "session": label, "session_th": label_th, "exercises": items})
            bi += 1
        else:
            rest = "Active Recovery" if (d == 6 or days <= 3) and rng.random() < 0.4 else "Rest"
            items = ["Light walk 20-30 min", "Stretching 10 min"] if rest == "Active Recovery" else []
            schedule.append({"day": DAY_NAMES[d], "session": rest,
                             "session_th": "พักแบบเคลื่อนไหว" if items else "พัก", "exercises": items})

    focus = focus_of(blocks)
    focus_str = ", ".join(focus)
    focus_th = ", ".join(FOCUS_TH.get(f, f) for f in focus)

    gen_en, gen_th = GENDERS[gender]
    age_rng, age_en, age_th = AGES[age]
    lvl_en, lvl_th = LEVELS[level]
    eq_en, eq_th, _, loc, loc_th = EQUIPMENT[equip]
    goal_en, goal_th = g["en"], g["th"]
    if goal == "model_physique":
        goal_en, goal_th = ("Lean Model Physique (fitness model)", "หุ่นนายแบบ") if gender == "male" else \
                           ("Slim Model Physique", "หุ่นนางแบบ") if gender == "female" else (goal_en, goal_th)

    audience = f"{gen_en.lower() if gender != 'unisex' else 'anyone'} aged {age_rng}"
    eq_short = {"bodyweight": "No Equipment", "dumbbells": "Dumbbell", "resistance_bands": "Band",
                "kettlebell": "Kettlebell", "home_gym": "Home Gym", "full_gym": "Gym", "outdoor": "Outdoor"}[equip]
    name = f"{days}-Day {lvl_en} {g['en']} - {gen_en} {age_rng} ({eq_short})"
    if goal == "model_physique":
        name = f"{days}-Day {lvl_en} {goal_en.split(' (')[0]} - {gen_en} {age_rng} ({eq_short})"
    name_th = f"โปรแกรม{goal_th} {days} วัน/สัปดาห์ ระดับ{lvl_th} สำหรับ{gen_th if gender != 'unisex' else 'ทุกเพศ'} อายุ {age_rng} ({eq_th.split(' (')[0]})"

    age_note = {
        "teen": "Focuses on good technique and bodyweight control suitable for growing bodies.",
        "young_adult": "Uses progressive overload to build a strong foundation.",
        "adult": "Designed to fit a busy work schedule.",
        "middle_age": "Joint-friendly exercise selection with extra warm-up and recovery.",
        "senior": "Low-impact, joint-friendly movements with a focus on safety, balance and daily function.",
    }[age]
    age_note_th = {
        "teen": "เน้นท่าทางที่ถูกต้องและการควบคุมน้ำหนักตัว เหมาะกับร่างกายที่กำลังเจริญเติบโต",
        "young_adult": "เพิ่มความหนักทีละขั้น (progressive overload) เพื่อสร้างพื้นฐานที่แข็งแรง",
        "adult": "ออกแบบให้เหมาะกับคนวัยทำงานที่มีเวลาจำกัด",
        "middle_age": "เลือกท่าที่ถนอมข้อต่อ มีการวอร์มอัพและพักฟื้นมากขึ้น",
        "senior": "ท่าแรงกระแทกต่ำ ถนอมข้อต่อ เน้นความปลอดภัย การทรงตัว และการใช้ชีวิตประจำวัน",
    }[age]
    gender_note = {
        "female": "Extra emphasis on glutes, legs and core." if goal in STRENGTH_GOALS else "",
        "male": "Extra emphasis on chest, back and shoulders." if goal in ("muscle_gain", "strength", "weight_gain", "model_physique") else "",
        "unisex": "",
    }[gender]
    gender_note_th = {
        "female": "เน้นสะโพก ขา และแกนกลางลำตัวเป็นพิเศษ" if goal in STRENGTH_GOALS else "",
        "male": "เน้นอก หลัง และไหล่เป็นพิเศษ" if goal in ("muscle_gain", "strength", "weight_gain", "model_physique") else "",
        "unisex": "",
    }[gender]
    loc_phrase = {"home": "at home", "gym": "in the gym", "outdoor": "outdoors (park or track)"}[loc]
    eq_phrase = {"bodyweight": "no equipment (bodyweight only)", "outdoor": "no equipment"}.get(equip, eq_en.lower())
    eq_phrase_th = {"bodyweight": "โดยไม่ใช้อุปกรณ์ (บอดี้เวท)", "outdoor": "โดยไม่ใช้อุปกรณ์"}.get(equip, "โดยใช้" + eq_th)

    description = (f"A {weeks}-week, {days}-day-per-week {lvl_en.lower()} program for {audience} who want to {g['verb']}. "
                   f"Train {loc_phrase} with {eq_phrase}, about {minutes} minutes per session, focusing on {focus_str}. "
                   f"{age_note} {gender_note}").strip()
    description_th = (f"โปรแกรม {weeks} สัปดาห์ ฝึก {days} วันต่อสัปดาห์ ระดับ{lvl_th} สำหรับ{gen_th}อายุ {age_rng} ปี "
                      f"ที่ต้องการ{g['verb_th']} ฝึกที่{loc_th}{eq_phrase_th} ครั้งละประมาณ {minutes} นาที "
                      f"เน้น{focus_th} {age_note_th} {gender_note_th}").strip()

    tags = {goal, gender, age, level, equip, loc, INTENSITY[level]}
    if not allow_high_impact(level, age):
        tags.add("low_impact")
    tags = sorted(tags)

    schedule_text = " | ".join(
        f"{s['day']}: {s['session']} ({s['session_th']})" + (f" - {'; '.join(s['exercises'])}" if s["exercises"] else "")
        for s in schedule)

    return OrderedDict([
        ("program_id", f"WP{pid:04d}"),
        ("program_name", name),
        ("program_name_th", name_th),
        ("goal", goal), ("goal_en", goal_en), ("goal_th", goal_th),
        ("gender", gender), ("age_group", age), ("age_range", age_rng),
        ("level", level), ("intensity", INTENSITY[level]),
        ("days_per_week", days), ("minutes_per_session", minutes), ("program_weeks", weeks),
        ("equipment", equip), ("equipment_detail", eq_en), ("location", loc),
        ("focus_areas", focus_str), ("focus_areas_th", focus_th),
        ("description", description), ("description_th", description_th),
        ("tags", ", ".join(tags)),
        ("weekly_schedule", schedule_text),
        ("_schedule", schedule),
    ])


def valid(goal, gender, age, level, equip, days):
    g = GOALS[goal]
    if gender not in g["genders"] or age not in g["ages"]:
        return False
    tmpl = g.get("templates_male") if (goal == "model_physique" and gender == "male") else g["templates"]
    if days not in tmpl:
        return False
    if age in ("senior", "teen") and level == "advanced":
        return False
    if age == "senior" and days > 5:
        return False
    if level == "beginner" and days > 4:
        return False
    if level == "advanced" and days < 3:
        return False
    if goal in ("strength", "muscle_gain", "weight_gain") and equip == "outdoor":
        return False
    if goal in ("flexibility", "stress_relief", "posture_back", "postnatal") and equip in ("kettlebell", "full_gym") and level == "advanced":
        return False
    if goal == "postnatal" and (level == "advanced" or equip in ("kettlebell", "full_gym")):
        return False
    if goal in ("flexibility", "stress_relief") and equip not in ("bodyweight", "resistance_bands", "outdoor"):
        return False
    if goal == "glute_shaping" and equip == "outdoor":
        return False
    return True


def main():
    rng = random.Random(SEED)
    # Stratified sampling: ทุกกลุ่ม (เป้าหมาย x เพศ x ช่วงอายุ) ได้ PER_GROUP โปรแกรม
    # บังคับให้มีแบบฟิตเนส + แบบไม่ใช้อุปกรณ์ (ถ้าทำได้) และกระจายระดับความยากให้ครบ
    picked = []
    for goal in GOALS:
        for gender in GENDERS:
            for age in AGES:
                opts = [(goal, gender, age, lv, eq, days) for lv in LEVELS for eq in EQUIPMENT for days in range(2, 7)
                        if valid(goal, gender, age, lv, eq, days)]
                if not opts:
                    continue
                rng.shuffle(opts)
                chosen, pairs = [], set()

                def add(pred):
                    for o in opts:
                        if len(chosen) < PER_GROUP and (o[3], o[4]) not in pairs and pred(o):
                            chosen.append(o)
                            pairs.add((o[3], o[4]))
                            return

                add(lambda o: o[4] == "full_gym")
                add(lambda o: o[4] == "bodyweight")
                for lv in LEVELS:
                    if lv not in {c[3] for c in chosen}:
                        add(lambda o, lv=lv: o[3] == lv and o[4] not in {c[4] for c in chosen})
                for lv in LEVELS:
                    if lv not in {c[3] for c in chosen}:
                        add(lambda o, lv=lv: o[3] == lv)
                while len(chosen) < PER_GROUP:
                    before = len(chosen)
                    add(lambda o: o[4] not in {c[4] for c in chosen})
                    if len(chosen) == before:
                        add(lambda o: True)
                    if len(chosen) == before:
                        break
                picked += chosen

    picked.sort(key=lambda c: (list(GOALS).index(c[0]), list(AGES).index(c[2]), c[1], list(LEVELS).index(c[3]), c[5]))
    programs = [make_program(i + 1, *c, rng=rng) for i, c in enumerate(picked)]

    fields = [k for k in programs[0] if not k.startswith("_")]
    with open(OUT_DIR / "workouts.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for p in programs:
            w.writerow({k: p[k] for k in fields})

    with open(OUT_DIR / "workouts.json", "w", encoding="utf-8") as f:
        json.dump([{**{k: p[k] for k in fields if k != "weekly_schedule"}, "weekly_schedule": p["_schedule"]}
                   for p in programs], f, ensure_ascii=False, indent=2)

    print(f"Generated {len(programs)} programs -> {OUT_DIR / 'workouts.csv'}")


if __name__ == "__main__":
    main()
