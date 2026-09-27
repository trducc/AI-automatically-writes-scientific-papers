# 🧪 Smart Research Lab - Hệ Thống Tự Động Viết Bài Báo Khoa Học (AI-Driven Scientific Paper Generation)

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![ReportLab](https://img.shields.io/badge/PDF_Engine-ReportLab-005588?style=for-the-badge)](https://www.reportlab.com/)
[![ICLR Standard](https://img.shields.io/badge/Paper_Format-ICLR_Preprint-000000?style=for-the-badge)](https://iclr.cc/)

> **Smart Research Lab** là hệ thống nghiên cứu và tự động khởi tạo bài báo khoa học chuẩn quốc tế (ICLR / NeurIPS Preprint) dựa trên kiến trúc **Multi-Agent** (Nhiều Agent nhóm chuyên biệt) và **Multi-Model / Multi-Hardware Engine** (Hỗ trợ Ollama Local GPU, Google Gemini, OpenAI GPT-4o, Anthropic Claude, DeepSeek).

---

## 📌 1. Các Tính Năng Nổi Bật

- 🤖 **Kiến trúc Multi-Agent Chuyên Nghiệp**:
  - **`Planner Agent`**: Đóng vai trò Lead AI Architect, tự động phân tích đề tài, thiết lập giả thuyết nghiên cứu (*Hypothesis*), đặt vấn đề, đề xuất phương pháp và xây dựng bộ chỉ số đánh giá thực nghiệm.
  - **`Researcher Agent`**: Đóng vai trò Senior Academic Librarian, truy vấn và tổng hợp các trích dẫn tài liệu tham khảo khoa học liên quan trực tiếp đến đề tài.
  - **`Writer Agent`**: Đóng vai trò Technical Author, soạn thảo toàn bộ nội dung bài báo khoa học theo độ sâu yêu cầu (*4 trang, 8 trang, 12 trang*), tự động xử lý và render công thức toán học LaTeX cùng định dạng Markdown.
  - **`Reviewer Agent`**: Đóng vai trò ICLR Senior Meta-Reviewer, tiến hành phản biện bài báo kín (*Double-blind Peer Review*), đánh giá 4 tiêu chí (*Faithfulness, Relevance, Task Completion, Reproducibility*), đưa ra điểm số tổng quan và quyết định Accept/Reject cùng nhận xét chi tiết.

- 🎨 **Định Dạng Bài Báo Chuẩn Quốc Tế (ICLR / AI-Scientist Preprint)**:
  - Font chữ serif **Times-Roman / Times-Bold** tiêu chuẩn xuất bản quốc tế.
  - Header đường kẻ đen trên đầu trang: `AI-Scientist Generated Preprint`.
  - Tiêu đề căn giữa viết hoa (UPPERCASE), khối tác giả phản biện kín (`Anonymous authors / Paper under double-blind review`).
  - Phần **ABSTRACT** thụt lề 2 bên và căn đều 2 lề (Justified).
  - Tự động chuyển đổi các công thức toán LaTeX (`$$\mathcal{L}_{\text{total}} = ...$$`, `$y_{l,h} = \sigma(...)$`) sang định dạng hiển thị ký hiệu toán học Unicode & Times-Italic mượt mà, không bị lỗi mã code thô.
  - Tự động xuất đồng thời cả file **PDF** và file mã nguồn **LaTeX (`.tex`)** tương ứng.

- ⚙️ **Hỗ Trợ Đa Động Cơ (Multi-Model & Multi-Hardware Engine)**:
  - **Local GPU Engine**: Tận dụng GPU máy cá nhân (NVIDIA CUDA) thông qua server **Ollama** (`llama3.2`, `qwen2.5-coder`, `mistral`, `vLLM`) để chạy hoàn toàn Offline miễn phí.
  - **Cloud API Engine**: Kết nối trực tiếp các Cloud LLM hàng đầu như **Google Gemini** (`gemini-2.5-flash`, `gemini-2.0-flash`, `gemini-1.5-flash`, `gemini-1.5-pro`), **OpenAI** (`gpt-4o`, `gpt-4o-mini`), **Anthropic** (`claude-3-5-sonnet`), **DeepSeek** (`deepseek-chat`). Tự động quét danh sách model khả dụng qua API Key (`genai.list_models()`).

---

## 🛠️ 2. Yêu Cầu Môi Trường & Cài Đặt

### 📋 Môi trường yêu cầu
- **Hệ điều hành**: Windows / Linux / macOS.
- **Python**: Phiên bản 3.11+.

### ⚙️ Hướng dẫn cài đặt bước qua bước

1. **Clone repository (hoặc mở thư mục dự án)**:
   ```bash
   cd AI-automatically-writes-scientific-papers
   ```

2. **Tạo và kích hoạt môi trường ảo Python (Virtual Environment)**:
   - **Windows (PowerShell)**:
     ```powershell
     py -3.11 -m venv env
     .\env\Scripts\Activate.ps1
     ```
   - **Linux / macOS**:
     ```bash
     python3.11 -m venv env
     source env/bin/activate
     ```

3. **Cài đặt các gói thư viện phụ thuộc**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Chuẩn bị dữ liệu thực nghiệm mẫu (Tùy chọn)**:
   ```bash
   python data/shakespeare_char/prepare.py
   python data/enwik8/prepare.py
   python data/text8/prepare.py
   ```

---

## 🚀 3. Hướng Dẫn Sử Dụng Giao Diện Web App (Streamlit UI)

Khởi chạy ứng dụng Streamlit bằng lệnh:

```bash
streamlit run app.py --server.port 8501
```

Sau khi chạy lệnh, truy cập giao diện trên trình duyệt tại địa chỉ: **`http://localhost:8501`**

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                        SMART RESEARCH LAB WEB APP                       │
├────────────────────────────────┬────────────────────────────────────────┤
│ 🎛️ SIDEBAR (CONTROL ROOM)      │ 📊 MAIN DASHBOARD (CONTENT PANELS)     │
│                                │                                        │
│ • Research Topic (Nhập đề tài) │ 1. Multi-Agent Workflow Status        │
│ • Execution Engine (Chọn GPU/  │ 2. 🤖 Multi-Agent Artifacts Breakdown  │
│   Cloud API / Demo)            │    [🧠 Planner] [📚 Researcher]        │
│ • API Key (Nhập key Gemini/    │    [✍️ Writer ] [⚖️ Reviewer  ]        │
│   OpenAI/DeepSeek)             │ 3. Training Dynamics & Heatmaps        │
│ • Paper Detail Depth (4/8/12P) │ 4. Interactive Report Viewer & Download│
│ • [🚀 Generate Paper Report]   │                                        │
└────────────────────────────────┴────────────────────────────────────────┘
```

---

### 🎛️ 3.1. Bảng Điều Khiển Sidebar (Research Control Room)

Bảng điều khiển bên trái cho phép thiết lập toàn bộ tham số chạy nghiên cứu:

1. **Research Topic (Đề tài nghiên cứu)**:
   - Nhập tên chủ đề bài báo khoa học mong muốn (ví dụ: *"Quantum Graph Neural Networks for Molecular Property Prediction"* hoặc *"Dynamic Attention Head Gating in Transformers"*).

2. **🤖 Model & Hardware Engine (Động cơ thực thi)**:
   - **🎮 Local GPU / Local LLM (Ollama / CUDA)**: Chạy trên card đồ họa local (NVIDIA GeForce RTX GPU). Bạn chọn target model Ollama (`llama3.2`, `qwen2.5-coder`) và endpoint local (`http://localhost:11434/v1`).
   - **☁️ Cloud API (OpenAI / Claude / Gemini / DeepSeek)**: Chọn dịch vụ Cloud API. Hệ thống hỗ trợ các model Gemini mới nhất (`gemini-2.5-flash`, `gemini-2.0-flash`, `gemini-1.5-flash`, `gemini-1.5-pro`), OpenAI `gpt-4o`, DeepSeek `deepseek-chat`, Claude 3.5.
   - **⚡ Fast Demo / CPU Smoke Test Mode**: Chế độ chạy thử nhanh offline với dữ liệu thực nghiệm lưu sẵn.

3. **API Key (Dành cho Cloud API)**:
   - Nhập mã API Key tương ứng (ví dụ: Google AI Studio Key cho Gemini). Key sẽ được lưu trong `session_state` trong suốt phiên làm việc.

4. **Paper Detail Depth (Độ sâu độ dài bài báo)**:
   - **Standard (4 Pages)**: Sinh ~2,000+ từ với 7 mục khoa học chuẩn.
   - **Detailed (8 Pages)**: Sinh ~4,500+ từ với đầy đủ các tiểu mục (1.1, 1.2, 3.1, 3.2, 4.1), bảng số liệu và phân tích ablation study.
   - **Comprehensive (12 Pages)**: Sinh ~7,500+ từ kéo dài 12 trang với phụ lục, phân tích độ trễ phần cứng và tài liệu tham khảo phong phú.

5. **Nút [🚀 Generate paper report]**:
   - Nhấn nút để kích hoạt luồng 4 Agent chạy tự động.

---

### 📊 3.2. Các Trang & Tab Chức Năng Trên Giao Diện Chính

Ứng dụng chia thành 5 trang chính điều hướng ở Sidebar:

#### 1. 📊 Dashboard & Metrics (Trang Tổng Quan & Tiến Trình Agent)
- **Multi-agent workflow**: Hiển thị thẻ trạng thái phản hồi thời gian thực của 4 Agent (`Planner`, `Researcher`, `Writer`, `Reviewer`).
- **🤖 Multi-Agent Artifacts & Execution Breakdown**: Gồm 4 Tab trực quan cho phép xem sản phẩm riêng biệt của từng Agent:
  - **Tab `🧠 Planner Agent`**: Xem bản đề xuất kế hoạch nghiên cứu, giả thuyết (*Hypothesis*), phương pháp & chỉ số thực nghiệm.
  - **Tab `📚 Researcher Agent`**: Xem danh mục các trích dẫn khoa học và dẫn chứng do Researcher tổng hợp.
  - **Tab `✍️ Writer Agent`**: Xem trực tiếp bản thảo bài báo dạng Markdown trước khi biên dịch PDF.
  - **Tab `⚖️ Reviewer Agent`**: Xem chi tiết bảng điểm phản biện ICLR, quyết định Pass/Fail và ý kiến đóng góp từ Reviewer.
- **Interactive Training Dynamics**: Biểu đồ Plotly tương tác hiển thị vết huấn luyện Loss qua 1,000 steps của 5 thử nghiệm (`Baseline`, `Gating L1 1e-4`, `Gating L1 1e-3`,...).
- **Attention Gate Heatmap**: Bảng nhiệt thể hiện tỷ lệ Sparsity của các đầu Attention Gate.

#### 2. 🧪 Text Generation Sandbox (Thử Nghiệm Sinh Văn Bản)
- Cho phép thử nghiệm nhanh các tham số suy luận LLM (`Temperature`, `Max Tokens`, `Top-K`, `Top-P`) và đo lường tốc độ suy luận (Latency ms, Tokens/sec).

#### 3. 🏛️ Architecture & Implementation (Kiến Trúc & Mã Nguồn)
- Trình bày công thức toán học hàm loss bổ sung và mã nguồn Python mẫu cho cơ chế Dynamic Attention Head Gating.

#### 4. 📂 Dataset Explorer (Khám Phá Dữ Liệu)
- Kiểm tra dung lượng và vocabulary của các tập dữ liệu `shakespeare_char`, `enwik8`, `text8`. Tự động tải & chuẩn bị dataset chỉ với 1 click, trích xuất mẫu dữ liệu ngẫu nhiên.

#### 5. 📑 Report & Artifacts (Báo Cáo & Tệp Xuất Bản)
- **Embedded PDF Report Viewer**: Trình xem file PDF bài báo trực tiếp ngay trên giao diện web.
- **Download Buttons**:
  - Nút **Download PDF**: Tải trực tiếp file PDF bài báo đã tạo.
  - Nút **Download complete artifact bundle**: Tải trọn bộ tệp nén ZIP chứa kết quả thực nghiệm, file `.tex`, `.json`, `.npy`, `.png`.

---

## 📂 4. Cấu Trúc Thư Mục Dự Án

```text
AI-automatically-writes-scientific-papers/
├── app.py                      # Giao diện chính Web App (Streamlit UI)
├── smart_research_lab.py       # Core Multi-Agent engine & ReportLab PDF generator
├── pretrained_reference.py     # Thư viện suy luận tham chiếu HuggingFace / Local
├── requirements.txt            # Danh sách gói thư viện cài đặt
├── README.md                   # File tài liệu hướng dẫn dự án
├── data/                       # Dữ liệu thực nghiệm (shakespeare_char, enwik8, text8)
│   ├── enwik8/
│   ├── shakespeare_char/
│   └── text8/
├── results/                    # Thư mục chứa kết quả đầu ra
│   ├── nanoGPT/                # Artifacts thực nghiệm nanoGPT
│   └── smart_research_lab/     # Các file PDF & LaTeX (.tex) bài báo được tạo ra
└── templates/                  # Thư mục chứa template nền tảng (nanoGPT, 2d_diffusion, grokking)
```

---

## 📄 5. Mẫu Xuất Bản Đầu Ra (Generated Outputs)

Mỗi lần chạy thành công, hệ thống sẽ tự động tạo cặp file bài báo chuẩn tại thư mục `results/smart_research_lab/`:
- **File PDF (`.pdf`)**: Định dạng PDF chuẩn ICLR / AI-Scientist Preprint với font Times-Roman, header kẻ ngang, abstract thụt lề và công thức toán render đẹp mắt.
- **File LaTeX (`.tex`)**: Mã nguồn LaTeX đi kèm sẵn sàng để biên dịch trực tiếp trên Overleaf hoặc TeXLive.

---

## ⚖️ License & Responsible Use

Dự án này được phát triển dựa trên định hướng mở rộng của **The AI Scientist**. Khi sử dụng bài báo được sinh tự động bởi hệ thống trong các công bố khoa học, vui lòng trích dẫn nguồn công khai:

> *"This manuscript was autonomously generated using Smart Research Lab (AI-Scientist Multi-Agent Framework)."*
