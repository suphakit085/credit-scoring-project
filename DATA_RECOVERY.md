# สถานะข้อมูลและวิธีกู้คืน

`data/raw/` ว่างเปล่าแล้ว เหลือเพียง `.gitkeep` ไฟล์ Kaggle ต้นฉบับหายไป
เอกสารนี้บันทึกว่าอะไรยังเหลืออยู่ อะไรรันได้ และจะกู้ส่วนที่ขาดอย่างไร

`.gitignore` กัน `data/` และ `models/` ออกจาก git ทั้งหมด **ไฟล์ใน `data/processed/`
จึงไม่มีสำเนาอื่นนอกจากบนดิสก์เครื่องนี้** ให้สำรองออกนอกเครื่องก่อนทำอย่างอื่น

```bash
robocopy "C:\Credit Scoring\credit-scoring-project\data\processed" "E:\backup\credit-scoring\processed" /E /Z
```

## สิ่งที่ยังเหลืออยู่

| ไฟล์ | ขนาด | SHA-256 (ย่อ) |
|---|---:|---|
| train_cleaned.csv | 795 MB | `23c74c11c08b1e51` |
| test_cleaned.csv | 131 MB | `b41f47e818a98611` |
| prev_agg.csv | 263 MB | `c9daddf9d20fcacd` |
| bureau_agg.csv | 128 MB | `7704b4fdd7e72730` |
| install_agg.csv | 116 MB | `c4e649ec9553ee8e` |
| cc_agg.csv | 81 MB | `d487e001e1d72a58` |
| pos_agg.csv | 65 MB | `cf7821daf87d54b1` |

ตรวจความสมบูรณ์ได้ด้วย

```bash
sha256sum data/processed/*.csv
```

`train_cleaned.csv` ผ่านการตรวจแล้วว่าสมบูรณ์ การแบ่ง train/calibration/validation
ด้วย `random_state=42` ให้ผลตรงกับ metrics ที่บันทึกไว้ในโมเดลรุ่นก่อนหน้าทุกหลัก

## อะไรรันได้ อะไรรันไม่ได้

| สคริปต์ | สถานะ | เหตุผล |
|---|---|---|
| `src/model_training.py` | รันได้ | อ่าน `data/processed/train_cleaned.csv` |
| `src/check_data_quality.py` | รันได้ | อ่าน `data/processed/train_cleaned.csv` |
| `src/inference.py`, แอป Streamlit | รันได้ | อ่าน model bundle |
| `src/data_preprocessing.py` | **รันไม่ได้** | ต้องใช้ไฟล์ใน `data/raw/` |

การเทรนโมเดลใหม่ทั้งหมดจึงยังทำได้ตามปกติโดยไม่ต้องมีข้อมูลดิบ

## วิธีกู้ `data_preprocessing.py` ให้กลับมารันได้

ไฟล์ aggregate ทั้งห้าตัวยังอยู่ครบ จึงไม่ต้องดาวน์โหลดตารางประวัติขนาดใหญ่ใหม่
ดาวน์โหลดจาก Kaggle เฉพาะสองไฟล์นี้ลง `data/raw/`

- `application_train.csv`
- `application_test.csv`

แล้วสั่ง

```bash
python src/data_preprocessing.py --reuse-aggregates
```

แฟล็ก `--reuse-aggregates` จะใช้ `*_agg.csv` ที่มีอยู่แทนการอ่าน `bureau.csv`,
`previous_application.csv`, `POS_CASH_balance.csv`, `credit_card_balance.csv` และ
`installments_payments.csv` ใหม่ ซึ่งรวมกันมากกว่า 2 GB

ผลลัพธ์ที่ได้ควรมี checksum ตรงกับตารางด้านบน หากไม่ตรงแปลว่าไฟล์ Kaggle
คนละรุ่นกับที่ใช้เทรนโมเดลรุ่นปัจจุบัน
