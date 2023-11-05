from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from nltk.stem import WordNetLemmatizer
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import tkinter as tk
from PIL import Image, ImageTk
import re
import nltk
nltk.download('wordnet')

# Đọc các file dữ liệu đầu vào
# reviews_train = []
# for line in open('full_train.txt', 'r', encoding = 'utf-8'):
#     reviews_train.append(line.strip())

reviews_train = []
for line in open('full_supper.txt', 'r', encoding = 'utf-8'):
    reviews_train.append(line.strip())

# reviews_test = []
# for line in open('full_test.txt', 'r', encoding = 'utf-8'):
#     reviews_test.append(line.strip())

# Làm sạch và tiền xử lý dữ liệu
REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\\[\\]]")
REPLACE_WITH_SPACE = re.compile("(<br\\s*/><br\\s*/>)|(\\-)|(\\/)")

def preprocess_reviews(reviews):
    reviews = [REPLACE_NO_SPACE.sub("", line.lower()) for line in reviews]
    reviews = [REPLACE_WITH_SPACE.sub(" ", line) for line in reviews]
    return reviews

reviews_train_clean = preprocess_reviews(reviews_train)
# reviews_test_clean = preprocess_reviews(reviews_test)

def get_lemmatized_text(corpus):
    lemmatizer = WordNetLemmatizer()
    return [' '.join([lemmatizer.lemmatize(word) for word in review.split()]) for review in corpus]

reviews_train_clean = get_lemmatized_text(reviews_train_clean)
# reviews_test_clean = get_lemmatized_text(reviews_test_clean)

stop_words = ['in', 'of', 'at', 'a', 'the']
ngram_vectorizer = CountVectorizer(binary = True, ngram_range = (1,3), stop_words = stop_words)
# ngram_vectorizer = CountVectorizer()
# ngram_vectorizer = TfidfVectorizer(ngram_range = (1, 2), stop_words = english_stop_words)
ngram_vectorizer.fit(reviews_train_clean)
X = ngram_vectorizer.transform(reviews_train_clean)
# X_test = ngram_vectorizer.transform(reviews_test_clean)

# ngram_vectorizer.fit(reviews_train)
# X = ngram_vectorizer.transform(reviews_train)
# X_test = ngram_vectorizer.transform(reviews_test)

# print(ngram_vectorizer.get_feature_names_out()[1736533])

# target = [1 if i < 12500 else 0 for i in range(25000)]
target1 = [1 if i < 25000 else 0 for i in range(50000)]
# X_train, X_val, y_train, y_val = train_test_split(X, target1, train_size = 0.75)

# accuracy = []
# regularization = [0.01, 0.05, 0.25, 0.5, 1]
# for c in regularization:
#     lr = LinearSVC(C = c, dual = True, max_iter = 1000)
#     lr.fit(X_train, y_train)
#     accuracy.append(accuracy_score(y_val, lr.predict(X_val)))
#     print("Độ chính xác với C = %s: %s" % (c, accuracy_score(y_val, lr.predict(X_val))))

# highestC = [regularization[i] for i in range(len(regularization)) if accuracy[i] == max(accuracy)][0]

# Độ chính xác với C = 0.01: 0.8928
# Độ chính xác với C = 0.05: 0.89408
# Độ chính xác với C = 0.25: 0.89392
# Độ chính xác với C = 0.5: 0.89376
# Độ chính xác với C = 1: 0.8936
# Độ chính xác cuối: 0.90068
# (11173, 1156, 1327, 11344)

# final_model = LogisticRegression(C = 1, max_iter = 1000)
final_model = LinearSVC(C = 0.01, dual = True, max_iter = 1000)
final_model.fit(X, target1)
# y_pred = final_model.predict(X_test)
# print("Độ chính xác cuối: %s" % accuracy_score(target, y_pred))

# Thử nghiệm mô hình
def display_text():
    input_text = text_entry.get("1.0", tk.END)  # Lấy đoạn văn bản từ ô nhập liệu
    input_text = REPLACE_NO_SPACE.sub("", input_text)
    input_text = REPLACE_WITH_SPACE.sub(" ", input_text)
    lemmatizer = WordNetLemmatizer()
    input_text = ' '.join([lemmatizer.lemmatize(word) for word in input_text.split()])
    
    textend = ngram_vectorizer.transform([input_text])
    y_pred = final_model.predict(textend)
    # y_pred = final_model.predict(textend.reshape(1, -1))
    
    ketqua = ""
    ketqua_dict = {1: "Kết quả đánh giá cảm xúc: Tích cực", 0: "Kết quả đánh giá cảm xúc: Tiêu cực"}
    ketqua = ketqua_dict[y_pred[0]]
    display_label.config(text = ketqua)  # Hiển thị đoạn văn bản trên nhãn

# Tạo cửa sổ giao diện
window = tk.Tk()
window.title("Phân tích cảm xúc các đánh giá IMDb")

# Đặt kích thước cửa sổ
window_width = 800
window_height = 500
window.geometry(f"{window_width}x{window_height}")

# Tạo ảnh nền
background_image = Image.open("Onepiece.jpg")
background_photo = ImageTk.PhotoImage(background_image)

# Tạo một label để hiển thị ảnh
background_label = tk.Label(window, image=background_photo)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

def _from_rgb(rgb):
    return "#%02x%02x%02x" % rgb

# Tạo nhãn và nhập văn bản
label = tk.Label(window, text="\n   Nhập đánh giá phim:\n", bg=_from_rgb((1, 38, 64)), fg="white", font=("Helvetica", 12, "bold"))
label.pack(anchor="w")

text_entry = tk.Text(window, height=20, width=50, wrap="word", borderwidth=2, relief="solid", bg=_from_rgb((252, 223, 181)))
text_entry.pack(anchor="w")  # Đặt neo về phía lề trái

display_button = tk.Button(window, text="Hiển thị đánh giá cảm xúc", borderwidth=2, relief="solid", height=2, width=30, command=display_text, bg=_from_rgb((248, 204, 165)))
display_button.pack()
display_button.place(x=500, y=345)

# Tạo nhãn để hiển thị đoạn văn bản
display_label = tk.Label(window, text="", height=5, width=35, bg=_from_rgb((248, 204, 165)), borderwidth=2, relief="solid", font=("Helvetica", 12, "bold"))
display_label.pack()
display_label.place(x=420, y=65)

# Chạy vòng lặp chính của giao diện
window.mainloop()
###################################################

# accuracy.append(accuracy_score(target, final_model.predict(X_test)))
# regularization.append('testError C:' + str(highestC))
# plt.plot(regularization, accuracy)
# plt.savefig("a1A.pdf")

# from sklearn.metrics import confusion_matrix
# tn, fp, fn, tp = confusion_matrix(y_pred, target).ravel()
# print(str((tn, fp, fn, tp)))

# Những từ "tích cực" và "tiêu cực" xuất hiện nhiều nhất
# feature_to_coef = {
#     word: coef for word, coef in zip(
#         cv.get_feature_names_out(), final_model.coef_[0]
#     )
# }

# for best_positive in sorted(
#         feature_to_coef.items(),
#         key=lambda x: x[1],
#         reverse=True)[:5]:
#     print(best_positive)

# print("\n")

# for best_negative in sorted(
#         feature_to_coef.items(),
#         key=lambda x: x[1])[:5]:
#     print(best_negative)

# ('excellent', 0.23168660287762674)    ('worst', -0.3484903268990883)
# ('perfect', 0.18552518046264543)      ('awful', -0.25993418728315604)
# ('great', 0.1758412640050427)         ('boring', -0.23984317933372)
# ('wonderful', 0.1620018283366827)     ('waste', -0.23826687511548666)
# ('amazing', 0.15394764582344697)      ('bad', -0.2175982892095277)

# ('worst', -0.3484903268990883)
# ('awful', -0.25993418728315604)
# ('boring', -0.23984317933372)
# ('waste', -0.23826687511548666)
# ('bad', -0.2175982892095277)