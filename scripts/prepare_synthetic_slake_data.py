import os
import json

def setup_sample_slake_data():
    base_dir = "data/slake"
    img_dir = os.path.join(base_dir, "imgs")
    sample_sub_dir = os.path.join(img_dir, "xmlab1")
    
    os.makedirs(sample_sub_dir, exist_ok=True)
    
    # 1. Create mask.txt
    mask_txt_path = os.path.join(base_dir, "mask.txt")
    mask_content = "1: liver\n2: lung\n3: kidney\n4: heart\n"
    with open(mask_txt_path, "w", encoding="utf-8") as f:
        f.write(mask_content)
        
    # 2. Create sample source and mask image placeholders (valid PNG header / bytes or dummy)
    # 1x1 black PNG binary signature
    png_bytes = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0\x00\x00\x03\x01\x01\x00\x18\xdd\x8d\xb0\x00\x00\x00\x00IEND\xaeB`\x82'
    
    src_img_path = os.path.join(sample_sub_dir, "source.jpg")
    mask_img_path = os.path.join(sample_sub_dir, "mask.png")
    
    if not os.path.exists(src_img_path):
        with open(src_img_path, "wb") as f:
            f.write(png_bytes)
            
    if not os.path.exists(mask_img_path):
        with open(mask_img_path, "wb") as f:
            f.write(png_bytes)
        
    # 3. Create train.json
    train_json_path = os.path.join(base_dir, "train.json")
    json_data = [
        {
            "img_name": "xmlab1/source.jpg",
            "question": "Is there a lung present in the image?",
            "answer": "Yes",
            "q_lang": "en",
            "location": "chest",
            "modality": "CT",
            "answer_type": "CLOSED",
            "content_type": "Modality"
        },
        {
            "img_name": "xmlab1/source.jpg",
            "question": "Where is the kidney located?",
            "answer": "Abdomen",
            "q_lang": "en",
            "location": "abdomen",
            "modality": "CT",
            "answer_type": "OPEN",
            "content_type": "Location"
        },
        {
            "img_name": "xmlab1/source.jpg",
            "question": "What is the primary organ shown?",
            "answer": "Liver",
            "q_lang": "en",
            "location": "abdomen",
            "modality": "CT",
            "answer_type": "OPEN",
            "content_type": "Organ"
        },
        {
            "img_name": "xmlab1/source.jpg",
            "question": "Does the picture contain liver?",
            "answer": "Yes",
            "q_lang": "en",
            "location": "abdomen",
            "modality": "CT",
            "answer_type": "CLOSED",
            "content_type": "Organ"
        }
    ]
    
    with open(train_json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)
        
    print(f"[✓] Local sample dataset directory successfully set up at: {base_dir}")

if __name__ == "__main__":
    setup_sample_slake_data()
