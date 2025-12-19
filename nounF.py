import pandas as pd
import torch
from math import log
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertJapaneseTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import Dataset
from fugashi import Tagger
import shutil
import os

hikensya_num = input("被験者番号を入力してください：")
print(f"被験者{hikensya_num}の嗜好を基にBERTをファインチューニングします．")

# ステップ1: 好みの映画と好みでない映画のIDを読み込む
liked_movies_file = f"./movies/{hikensya_num}_liked_movies.txt"
disliked_movies_file = f"./movies/{hikensya_num}_disliked_movies.txt"

with open(liked_movies_file, 'r') as file:
    liked_movie_ids = [line.strip() for line in file.readlines()]

with open(disliked_movies_file, 'r') as file:
    disliked_movie_ids = [line.strip() for line in file.readlines()]

# TF-IDF計算関数
def calculate_tfidf(all_reviews, total_movies):
    term_movie_count = Counter()
    tf_values = []

    for reviews in all_reviews:
        doc_terms = Counter()
        for review in reviews:
            words = review.split()
            term_count = Counter(words)
            doc_terms.update(term_count)
        tf_values.append(doc_terms)

        unique_terms = set(doc_terms.keys())
        for term in unique_terms:
            term_movie_count[term] += 1

    idf_values = {term: log(total_movies / (1 + count)) for term, count in term_movie_count.items()}

    tfidf_scores = []
    for tf in tf_values:
        tfidf_scores.append({term: tf_val * idf_values[term] for term, tf_val in tf.items()})

    return tfidf_scores, idf_values

# 名詞抽出とTF-IDF計算
def extract_reviews_and_nouns(movie_ids, total_movies, output_file):
    tagger = Tagger()
    all_reviews = []
    all_nouns = []

    for movie_id in movie_ids:
        file_path = f"./極性付きレビューファイル/{movie_id}.xlsx"
        try:
            data = pd.read_excel(file_path)
            positive_reviews = data[data.iloc[:, 1] == 1].iloc[:, 0].dropna().tolist()
            movie_reviews = []
            for review in positive_reviews:
                nouns = ' '.join([word.surface for word in tagger(review) if '名詞' in word.feature])
                if nouns:
                    all_reviews.append(review)
                    movie_reviews.append(nouns)
            #映画mの各レビューに名詞が含まれていればその各名詞を収集
            if movie_reviews:
                all_nouns.append(movie_reviews)
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

    if all_nouns:
        tfidf_scores, idf_values = calculate_tfidf(all_nouns, total_movies)

        total_scores = Counter()
        for doc_scores in tfidf_scores:
            for term, score in doc_scores.items():
                total_scores[term] += score

        sorted_nouns = total_scores.most_common()

       # 全単語ランキングをファイルに出力
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("単語, スコア\n")
            for term, score in sorted_nouns:
                f.write(f"{term}, {score:.6f}\n")
        print(f"名詞ランキングを {output_file} に保存しました。")

    return all_reviews, sorted_nouns,total_scores

#レビュースコア計算
def calculate_review_scores(reviews, total_scores):
    tagger = Tagger()
    review_scores = []

    for review in reviews:
        score = 0
        #各レビュー文に含まれる名詞を取得
        nouns = [word.surface for word in tagger(review) if '名詞' in word.feature]
        for noun in nouns:
            #各名詞のスコアをレビュー文のスコアに加算
            score += total_scores.get(noun, 0)
        review_scores.append((review, score))

    return review_scores

#outputディレクトリ消去用関数
def clean_output_directory(output_dir):
    if os.path.exists(output_dir):
        try:
            shutil.rmtree(output_dir)
            print(f"ディレクトリ {output_dir} を削除しました。")
        except Exception as e:
            print(f"ディレクトリ {output_dir} の削除中にエラーが発生しました: {e}")
    else:
        print(f"ディレクトリ {output_dir} は存在しません。")

# 好み/好みでない映画の肯定的なレビューを抽出し、名詞のtfidfランキングを保存
liked_reviews, liked_nouns, liked_total_scores = extract_reviews_and_nouns(liked_movie_ids, total_movies=10, output_file=f"./ランキング/{hikensya_num}_liked_名詞ランキング.txt")
disliked_reviews, disliked_nouns, disliked_total_scores = extract_reviews_and_nouns(disliked_movie_ids, total_movies=10, output_file=f"./ランキング/{hikensya_num}_disliked_名詞ランキング.txt")

liked_n_values = list(range(100, len(liked_reviews) , 100))
disliked_n_values = list(range(100, len(disliked_reviews) , 100))

# 要素数を揃える処理
if len(liked_n_values) < len(disliked_n_values):
    liked_n_values += [len(liked_reviews)] * (len(disliked_n_values) - len(liked_n_values))
elif len(disliked_n_values) < len(liked_n_values):
    disliked_n_values += [len(disliked_reviews)] * (len(liked_n_values) - len(disliked_n_values))

print(f"好みの映画で使用する名詞数のリスト: {liked_n_values}")
print(f"好みでない映画で使用する名詞数のリスト: {disliked_n_values}")

# ステップ8: 評価指標を定義
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

#初期値固定用
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

for liked_n, disliked_n in zip(liked_n_values, disliked_n_values):#Nを100~400,500に変化させる
    liked_top_n_nouns = [word for word, score in liked_nouns[:liked_n]]
    disliked_top_n_nouns = [word for word, score in disliked_nouns[:disliked_n]]

    liked_filtered_reviews = []
    liked_filtered_labels = []

    disliked_filtered_reviews = []
    disliked_filtered_labels = []

    #レビュー文のスコアを計算
    liked_scores = calculate_review_scores(liked_reviews, liked_total_scores)
    disliked_scores = calculate_review_scores(disliked_reviews, disliked_total_scores)
    print(f"抽出前の正例レビュー数:{len(liked_scores)}")
    print(f"抽出前の負例レビュー数:{len(disliked_scores)}")
    #レビューのスコア順にソート
    liked_scores.sort(key=lambda x: x[1],reverse=True)
    disliked_scores.sort(key=lambda x: x[1],reverse=True)
    #上位n件のレビューを抽出
    liked_scores = liked_scores[:liked_n]  # 上位 liked_n 件
    disliked_scores = disliked_scores[:disliked_n]  # 上位 disliked_n 件
    
    # フィルタリングされたレビューとラベルを更新
    liked_filtered_reviews = [r[0] for r in liked_scores]
    liked_filtered_labels = [1] * len(liked_filtered_reviews)

    disliked_filtered_reviews = [r[0] for r in disliked_scores]
    disliked_filtered_labels = [0] * len(disliked_filtered_reviews)

    # if choice=="2":
    #     if abs(len(liked_scores) - len(disliked_scores)) > 50:
    #         if len(liked_scores) > len(disliked_scores):
    #             liked_scores.sort(key=lambda x: x[1],reverse=True)
    #             liked_scores = liked_scores[:len(disliked_scores)]
    #             liked_filtered_reviews = [r[0] for r in liked_scores]
    #             liked_filtered_labels = liked_filtered_labels[:len(liked_filtered_reviews)]
    #         else:
    #             disliked_scores.sort(key=lambda x: x[1],reverse=True)
    #             disliked_scores = disliked_scores[:len(liked_scores)]
    #             disliked_filtered_reviews = [r[0] for r in disliked_scores]
    #             disliked_filtered_labels = disliked_filtered_labels[:len(disliked_filtered_reviews)]

    # データを結合
    filtered_reviews = liked_filtered_reviews + disliked_filtered_reviews
    filtered_labels = liked_filtered_labels + disliked_filtered_labels

    # ステップ4: データを分割（全レビューをファインチューニングに使用し、最初の10件をテストに使用）
    train_texts = filtered_reviews
    train_labels = filtered_labels

    test_texts = train_texts[:10]
    test_labels = train_labels[:10]

    #分類器の層の初期値のランダム初期化の値を固定
    set_seed(42)

    # ステップ5: トークナイズ
    tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese')
    # ステップ7: 事前学習済みBERTモデルをロード
    model = BertForSequenceClassification.from_pretrained('cl-tohoku/bert-base-japanese', num_labels=2)

    # 分類器の初期値を確認（オプション）
    print("初期分類器の重み:", model.classifier.weight)
    print("初期分類器のバイアス:", model.classifier.bias)

    train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=512)
    test_encodings = tokenizer(list(test_texts), truncation=True, padding=True, max_length=512)

    print(f"被験者{hikensya_num}の好みの映画のレビュー数は{len(liked_filtered_reviews)}です．")
    print(f"被験者{hikensya_num}の好みでない映画のレビュー数は{len(disliked_filtered_reviews)}です．")
    print(f"全レビュー数は{len(filtered_reviews)}です．")

    # ステップ6: PyTorch Datasetの作成
    class MovieReviewDataset(Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

    train_dataset = MovieReviewDataset(train_encodings, list(train_labels))
    test_dataset = MovieReviewDataset(test_encodings, list(test_labels))

    #トレーニング引数
    training_args = TrainingArguments(
        output_dir=f'./output/{hikensya_num}_tf・idfnoun',    # 出力ディレクトリ
        num_train_epochs=3,              # トレーニングエポック数
        per_device_train_batch_size=16,  # トレーニング時のバッチサイズ
        per_device_eval_batch_size=64,   # 評価時のバッチサイズ
        warmup_steps=500,                # 学習率スケジューラのウォームアップステップ数
        weight_decay=0.01,               # 重み減衰（L2正則化）
        #logging_dir='./logs',            # ログ保存ディレクトリ
        logging_dir='C:/Users/kazuma/logs',
        logging_steps=10,                # ログ出力間隔（ステップ数）
        evaluation_strategy="epoch",   # 各エポック終了時に評価
        save_strategy="epoch",         # 各エポック終了時にモデルを保存
        load_best_model_at_end=True      # 最良モデルを最後にロード
    )

    # ステップ10: Trainerの設定
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    # ステップ11: モデルをトレーニング
    trainer.train()

    # ステップ12: モデルを保存
    max_n = max(liked_n, disliked_n)  # liked_nとdisliked_nのうち大きい値を使用

    trainer.save_model(f"./models/nounmodels/{hikensya_num}/{max_n}")
    tokenizer.save_pretrained(f"./models/nounmodels/{hikensya_num}/{max_n}")
    

        # モデル作成後に出力ディレクトリを削除
    clean_output_directory(f"./output/{hikensya_num}_tf・idfnoun")
    
    # ステップ13: モデルを評価（テストデータで）
    predictions = trainer.predict(test_dataset)
    eval_results = compute_metrics(predictions)

    print(f"{max_n}個の名詞を用いたモデルの評価結果: {eval_results}")

print("すべての名詞数でのモデル作成が完了しました。")
