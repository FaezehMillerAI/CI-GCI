import os
import json

def setup_sample_vqa_rad_data():
    base_dir = "data/VQA-RAD"
    img_dir = os.path.join(base_dir, "VQA_RAD Image Folder")
    os.makedirs(img_dir, exist_ok=True)
    
    # 1x1 black PNG binary signature
    png_bytes = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0\x00\x00\x03\x01\x01\x00\x18\xdd\x8d\xb0\x00\x00\x00\x00IEND\xaeB`\x82'
    
    sample_img_path = os.path.join(img_dir, "syn_rad_01.png")
    if not os.path.exists(sample_img_path):
        with open(sample_img_path, "wb") as f:
            f.write(png_bytes)
            
    json_path = os.path.join(base_dir, "VQA_RAD Dataset Public.json")
    json_data = [
        {
            "qid": "rad_1",
            "image_name": "syn_rad_01.png",
            "image_organ": "CHEST",
            "question": "Is there a pleural effusion present?",
            "answer": "Yes",
            "answer_type": "CLOSED",
            "question_type": "PRESENCE"
        },
        {
            "qid": "rad_2",
            "image_name": "syn_rad_01.png",
            "image_organ": "HEAD",
            "question": "Is the brain midline shifted?",
            "answer": "No",
            "answer_type": "CLOSED",
            "question_type": "PRESENCE"
        },
        {
            "qid": "rad_3",
            "image_name": "syn_rad_01.png",
            "image_organ": "ABDOMEN",
            "question": "Is the liver size normal?",
            "answer": "Yes",
            "answer_type": "CLOSED",
            "question_type": "ATTRIBUTE"
        },
        {
            "qid": "rad_4",
            "image_name": "syn_rad_01.png",
            "image_organ": "CHEST",
            "question": "Is pneumothorax identified?",
            "answer": "No",
            "answer_type": "CLOSED",
            "question_type": "PRESENCE"
        },
        {
            "qid": "rad_5",
            "image_name": "syn_rad_01.png",
            "image_organ": "CHEST",
            "question": "Are cardiomegaly signs visible?",
            "answer": "No",
            "answer_type": "CLOSED",
            "question_type": "PRESENCE"
        }
    ]
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)
        
    print(f"[✓] VQA-RAD sample dataset successfully set up at: {base_dir}")

if __name__ == "__main__":
    setup_sample_vqa_rad_data()
