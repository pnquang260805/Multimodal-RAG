# 日常Q＆A

## Problem
Tra cứu thông tin luật pháp ở nhật (荒川区) (thủ tục hành chính, đi lại, ...)
Muốn: hỏi bằng tiếng việt -> tra cứu -> trả lời tiếng việt + cite + các file nguồn ở cuối

## Scope v0.1 (MoSCoW)
**Must (MVP)**
- [ ] M1: Quantize model 7B về 4bit để chạy được trên card 4GB VRAM
- [ ] M2: Ingest 50 trang nguồn cho pháp luật vào vector store
- [ ] M3: Hỏi tiếng Việt -> search -> trả về output + citation + file gốc
- [ ] M4: Không tìm thấy nguồn -> "Tôi không biết"

**Should**
- [ ] S1: Luật lao đông, cư trú
- [ ] S2: Tag theo domain để filter trong vector database 
- [ ] S3: Eval script
- [ ] S4: Tích hợp vào telegram

**Could**
- [ ] C1: Hybrid search
- [ ] C2: LLM as judge
- [ ] C3: v0.2: agent gọi API trả về tuyến tàu chạy (chắc là tham khảo API google map), lưu tài liệu lên S3 như MinIO
- [ ] C4: UI admin, UI upload tài liệu

**Won't**
- Tự tìm/đặt phòng, tự tìm tuyến đường realtime
- Auth, multi-user, deploy
- Scrape giá nhà (do thay đổi liên tục và khá tốn công)
- OCR hoặc dùng VLM để xử lý các bảng hướng dẫn tị nạn có mũi tên hoặc map

## Keywords
+ Multimodal RAG
+ 