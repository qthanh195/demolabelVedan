import ollama

def read_ocr_ollama(image_label):

    response = ollama.generate(
        model = "qwen2.5-vl:7b",
        prompt = 'Hãy đọc toàn bộ văn bản trong ảnh này. Sau đó tìm giá trị Net Weight (số và đơn vị). Trả về kết quả dạng JSON: {\"text\": \"...\", \"net_weight\": \"...\"}.',
        format = "json",
        images = image_label
    )
    print("Keets qua____________")
    print(response)