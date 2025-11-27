# 🎭 Sentiment Analysis - Phân Tích Cảm Xúc

## 📋 Mô Tả Dự Án

Đây là một mini project về **Machine Learning**, tập trung vào bài toán **phân tích cảm xúc (Sentiment Analysis)** để phân loại văn bản thành hai nhãn: **Tích cực (Positive)** và **Tiêu cực (Negative)**.

Dự án này được xây dựng nhằm mục đích học tập và thực hành các kỹ thuật Machine Learning cơ bản trong xử lý ngôn ngữ tự nhiên (NLP).

## 🎯 Mục Tiêu

- Hiểu và áp dụng các thuật toán Machine Learning cơ bản
- Làm quen với xử lý dữ liệu văn bản (Text Processing)
- Xây dựng mô hình phân loại cảm xúc
- Đánh giá hiệu suất mô hình
- Tạo giao diện demo đơn giản

## 📁 Cấu Trúc Dự Án

```
Sentiment_Analysis/
│
├── create_dataset.ipynb              # Notebook tạo dataset
├── sentiment_analysis.ipynb          # Notebook huấn luyện và đánh giá mô hình
├── sentiment_dataset_10000_v2.csv    # Dataset gồm 10,000 mẫu
├── sentiment_model.pkl               # Mô hình đã được huấn luyện
├── vectorizer.pkl                    # TF-IDF Vectorizer
├── app.py                            # Ứng dụng Streamlit
└── README.md                         # File này
```

## 📊 Dataset

### Thông Tin Dataset
- **Tổng số mẫu**: 10,000
- **Phân bố nhãn**: 
  - Positive: 5,000 mẫu (50%)
  - Negative: 5,000 mẫu (50%)
- **Cột dữ liệu**:
  - `ID`: Mã định danh
  - `Text`: Văn bản cần phân tích
  - `Label`: Nhãn (Positive/Negative)

### Ví Dụ Dữ Liệu

| ID | Text | Label |
|----|------|-------|
| 1 | This gadget is a game-changer for my daily routine. | Positive |
| 2 | The staff was lazy and unmotivated. Stay away. | Negative |
| 3 | What a reliable and durable item. | Positive |
| 4 | The interface is confusing and difficult to navigate. | Negative |

### Tạo Dataset

Dataset được tạo tự động bằng cách:
1. Định nghĩa các template câu mẫu cho cả Positive và Negative
2. Sử dụng từ điển từ đồng nghĩa để tạo biến thể
3. Áp dụng kỹ thuật Data Augmentation để tăng tính đa dạng
4. Trộn ngẫu nhiên để đảm bảo phân bố cân bằng

Chi tiết xem trong file `create_dataset.ipynb`

## 🤖 Mô Hình

### Kiến Trúc

```
Text Input
    ↓
TF-IDF Vectorizer
    ↓
Logistic Regression
    ↓
Prediction (0/1)
    ↓
Label (Negative/Positive)
```

### Các Bước Xây Dựng

1. **Tiền xử lý dữ liệu**
   - Làm sạch dữ liệu (loại bỏ giá trị null)
   - Chuẩn hóa văn bản (loại bỏ khoảng trắng thừa)
   - Mã hóa nhãn (Positive → 1, Negative → 0)

2. **Chia tập dữ liệu**
   - Tập train: 80% (8,000 mẫu)
   - Tập test: 20% (2,000 mẫu)
   - Stratified split để giữ tỉ lệ nhãn

3. **Feature Engineering**
   - **TF-IDF Vectorization**
     - N-gram range: (1, 2) - Unigram và Bigram
     - Min document frequency: 2
     - Max features: 20,000
     - Stop words: English
     - Sublinear TF: True

4. **Huấn luyện mô hình**
   - Thuật toán: **Logistic Regression**
   - Framework: scikit-learn

## 📈 Kết Quả

### Hiệu Suất Mô Hình

- **Accuracy**: ~100% trên tập test
- **Precision**: 1.00 (Positive), 1.00 (Negative)
- **Recall**: 1.00 (Positive), 1.00 (Negative)
- **F1-Score**: 1.00 (Positive), 1.00 (Negative)

### Confusion Matrix

```
                Predicted
              Neg    Pos
Actual  Neg  1000      0
        Pos     0   1000
```

> **Lưu ý**: Độ chính xác 100% cho thấy dataset có thể đơn giản hoặc có pattern rõ ràng. Trong thực tế, nên test trên dữ liệu thực tế để đánh giá khả năng tổng quát hóa.

## 🚀 Cài Đặt và Sử Dụng

### Yêu Cầu Hệ Thống

- Python 3.7+
- Jupyter Notebook
- Các thư viện Python cần thiết

### Cài Đặt Thư Viện

```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit joblib
```

### Chạy Notebook

1. **Tạo Dataset**:
   ```bash
   jupyter notebook create_dataset.ipynb
   ```

2. **Huấn luyện Mô hình**:
   ```bash
   jupyter notebook sentiment_analysis.ipynb
   ```

### Chạy Ứng Dụng Web

```bash
streamlit run app.py
```

Ứng dụng sẽ mở trên trình duyệt với các chức năng:
- **Dashboard**: Xem thông tin tổng quan về mô hình
- **Test Demo**: Nhập văn bản để dự đoán cảm xúc
- **Upload CSV**: Upload file CSV để phân tích hàng loạt

## 💡 Cách Sử Dụng Mô Hình

### Trong Python

```python
import joblib

# Load mô hình và vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Dự đoán cảm xúc
text = ["This product is amazing!"]
text_vector = vectorizer.transform(text)
prediction = model.predict(text_vector)

# 1 = Positive, 0 = Negative
print("Positive" if prediction[0] == 1 else "Negative")
```

### Qua Ứng Dụng Web

1. Mở ứng dụng bằng `streamlit run app.py`
2. Chọn tab "Test Demo"
3. Nhập văn bản cần phân tích
4. Nhấn "Dự đoán" để xem kết quả

## 🛠️ Công Nghệ Sử Dụng

| Công nghệ | Mục đích |
|-----------|----------|
| **Python** | Ngôn ngữ lập trình chính |
| **Pandas** | Xử lý dữ liệu dạng bảng |
| **NumPy** | Tính toán số học |
| **scikit-learn** | Xây dựng mô hình ML |
| **Matplotlib/Seaborn** | Trực quan hóa dữ liệu |
| **Streamlit** | Tạo giao diện web |
| **Joblib** | Lưu và load mô hình |

## 📚 Kiến Thức Học Được

### Machine Learning
- ✅ Chuẩn bị và tiền xử lý dữ liệu
- ✅ Chia tập train/test
- ✅ Feature Engineering với TF-IDF
- ✅ Huấn luyện mô hình Logistic Regression
- ✅ Đánh giá mô hình (Accuracy, Precision, Recall, F1-Score, Confusion Matrix)

### Natural Language Processing (NLP)
- ✅ Text Cleaning và Normalization
- ✅ Tokenization
- ✅ TF-IDF Vectorization
- ✅ N-gram models
- ✅ Stop words removal

### Data Science
- ✅ Exploratory Data Analysis (EDA)
- ✅ Data Augmentation
- ✅ Model Evaluation
- ✅ Data Visualization

## 🔍 Hướng Phát Triển

### Cải Thiện Dataset
- [ ] Thu thập dữ liệu thực tế từ reviews, tweets, comments
- [ ] Tăng độ phức tạp của văn bản
- [ ] Thêm nhãn Neutral (Trung lập)
- [ ] Cân bằng dữ liệu nếu có class imbalance

### Cải Thiện Mô Hình
- [ ] Thử nghiệm các thuật toán khác (SVM, Random Forest, XGBoost)
- [ ] Hyperparameter Tuning
- [ ] Cross-validation
- [ ] Ensemble methods
- [ ] Deep Learning (LSTM, BERT, PhoBERT cho tiếng Việt)

### Cải Thiện Ứng Dụng
- [ ] Thêm chức năng phân tích sentiment score (0-1)
- [ ] Visualize word cloud cho từ tích cực/tiêu cực
- [ ] Export báo cáo PDF
- [ ] Tích hợp API
- [ ] Deploy lên cloud (Heroku, Streamlit Cloud, AWS)

### Mở Rộng Tính Năng
- [ ] Phân tích cảm xúc đa ngôn ngữ
- [ ] Phát hiện spam/fake reviews
- [ ] Topic modeling
- [ ] Named Entity Recognition (NER)

## 📝 Tài Liệu Tham Khảo

### Thuật toán
- [Logistic Regression - Wikipedia](https://en.wikipedia.org/wiki/Logistic_regression)
- [TF-IDF - Wikipedia](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)

### Thư viện
- [scikit-learn Documentation](https://scikit-learn.org/)
- [Pandas Documentation](https://pandas.pydata.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)

### Khóa học
- [Machine Learning - Andrew Ng (Coursera)](https://www.coursera.org/learn/machine-learning)
- [Natural Language Processing - Coursera](https://www.coursera.org/specializations/natural-language-processing)

## 👨‍💻 Tác Giả

**Người học AI/ML**
- Đang trong quá trình học và thực hành Machine Learning
- Quan tâm đến NLP và Sentiment Analysis

## 📄 Giấy Phép

Dự án này được phát triển cho mục đích học tập. Bạn có thể tự do sử dụng và chỉnh sửa cho mục đích cá nhân.

## 🙏 Lời Cảm Ơn

Cảm ơn cộng đồng Machine Learning và các nguồn tài liệu mã nguồn mở đã hỗ trợ trong quá trình học tập.

---

**Ngày tạo**: Tháng 11, 2025  
**Phiên bản**: 1.0  
**Trạng thái**: ✅ Hoàn thành

