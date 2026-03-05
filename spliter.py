import os
import re

def split_law_by_chapters(input_file):
    # 1. Tạo thư mục chứa các chương nếu chưa có
    output_dir = "chapters"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Đã tạo thư mục: {output_dir}")

    # 2. Đọc nội dung file luật
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 3. Sử dụng Regex để tìm các phần bắt đầu bằng "Chương..."
    # Pattern này tìm chữ "Chương" đứng đầu dòng, theo sau là số La Mã/số thường
    chapters = re.split(r'\n(?=Chương\s+[IVX\d]+)', content)

    # 4. Lưu từng chương ra file riêng
    for i, chapter_content in enumerate(chapters):
        if not chapter_content.strip():
            continue
            
        # Lấy tên chương (dòng đầu tiên) để đặt tên file
        lines = chapter_content.strip().split('\n')
        chapter_title = lines[0].replace(':', '').strip()
        # Loại bỏ các ký tự không hợp lệ cho tên file
        file_name = re.sub(r'[\\/*?:"<>|]', "", chapter_title) + ".txt"
        
        file_path = os.path.join(output_dir, file_name)
        
        with open(file_path, 'w', encoding='utf-8') as f_out:
            f_out.write(chapter_content.strip())
        
        print(f"Đã lưu: {file_path}")

if __name__ == "__main__":
    split_law_by_chapters("law.txt")