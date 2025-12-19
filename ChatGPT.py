# ChatGPT.py  (OpenAI 版・対話入力・比較手法/提案手法1対応・N=100..500反復)
# -*- coding: utf-8 -*-

import os, re, glob, time, math
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd

# ===== OpenAI (v1) =====
_HAS_OPENAI = False
try:
    from openai import OpenAI
    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report


# ---------------- ユーティリティ ----------------
def read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]

def normalize_text(s: str) -> str:
    s = str(s or "")
    s = re.sub(r"[『』「」【】（）\(\)\[\]・:：\-–—/…★☆■◆○●◇◎▲▼▽△※]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def detect_cols_review_polarity(df: pd.DataFrame) -> Tuple[str, Optional[str]]:
    # 本文列の推定
    text_col = None
    for c in df.columns:
        lc = str(c).lower()
        if "review" in lc and "text" in lc:
            text_col = c; break
    if text_col is None:
        for c in df.columns:
            if any(k in str(c) for k in ["レビュー","本文","text","review"]):
                text_col = c; break
    if text_col is None:
        text_col = df.columns[0]
    # 極性列（任意）
    pol_col = None
    for c in df.columns:
        lc = str(c).lower()
        if lc in {"polarity","sentiment","label","score"}:
            pol_col = c; break
        if any(k in lc for k in ["極性","sent","pos","neg"]):
            pol_col = c; break
    return text_col, pol_col


# ---------------- データ読み込み（学習：条件1＝映画IDベース） ----------------
def load_data2_train(
    liked_ids: List[str], disliked_ids: List[str],
    data2_glob: str, use_only_positive: bool
) -> Tuple[List[str], List[int]]:
    """
    データ2: movie_id ごとの極性付きレビュー xlsx を読み込み、
    好み映画→正例(1)、非好み映画→負例(0) として学習用データを作成。
    use_only_positive=True の場合は 極性 >= 1 のみ採用（条件1で有効）。
    """
    texts, labels = [], []

    def add_file(mid: str, label: int):
        for path in glob.glob(data2_glob.format(movie_id=str(mid))):
            try:
                df = pd.read_excel(path)
                text_col, pol_col = detect_cols_review_polarity(df)
                col_texts = df[text_col].astype(str)

                if use_only_positive and pol_col is not None:
                    pol = pd.to_numeric(df[pol_col], errors="coerce")
                    mask = pol >= 1  # 仕様：1以上を採用
                    col_texts = col_texts[mask]

                for t in col_texts.dropna().tolist():
                    t = normalize_text(t)
                    if not t: continue
                    texts.append(t)
                    labels.append(label)
            except Exception as e:
                print(f"[WARN] skip {path}: {e}")

    for mid in liked_ids:
        add_file(mid, 1)
    for mid in disliked_ids:
        add_file(mid, 0)

    if not texts:
        raise ValueError("学習データ（データ2）が空です。パスや列名、極性列の有無をご確認ください。")
    return texts, labels


# ---------------- データ読み込み（学習：条件2＝ランキングtxtベース, 分離版） ----------------
def load_rank_reviews_split(subject: str) -> Tuple[List[str], List[str]]:
    """
    ./ランキング/{subject}_liked_reviews.txt と ./ランキング/{subject}_disliked_reviews.txt を読み込み。
    形式：1行目ヘッダ、2行目以降「レビュー, スコア」のように末尾がスコア。
    レビュー本文にカンマが含まれる前提で、行末の最後のカンマで rsplit する。
    スコアは無視。順序はファイル上の「上からの順」を維持する。
    戻り値: (pos_texts, neg_texts)
    """
    base = "./ランキング"
    liked_fp = os.path.join(base, f"{subject}_liked_reviews.txt")
    disliked_fp = os.path.join(base, f"{subject}_disliked_reviews.txt")

    if not os.path.exists(liked_fp):
        raise FileNotFoundError(f"正例ファイルが見つかりません: {liked_fp}")
    if not os.path.exists(disliked_fp):
        raise FileNotFoundError(f"負例ファイルが見つかりません: {disliked_fp}")

    def parse_file(path: str) -> List[str]:
        out = []
        with open(path, "r", encoding="utf-8") as f:
            lines = [ln.rstrip("\n") for ln in f]
        if not lines:
            return out
        for ln in lines[1:]:  # 1行目はヘッダ
            ln = ln.strip()
            if not ln:
                continue
            if "," in ln:
                head, _tail = ln.rsplit(",", 1)
                review = normalize_text(head)
            else:
                review = normalize_text(ln)
            if review:
                out.append(review)
        return out

    pos_texts = parse_file(liked_fp)
    neg_texts = parse_file(disliked_fp)

    if not pos_texts and not neg_texts:
        raise ValueError("ランキングtxt由来の学習データが空です。ファイル内容をご確認ください。")
    return pos_texts, neg_texts


# ---------------- データ読み込み（評価; 全体平均で y_true 作成） ----------------
def load_data3_eval_global_mean(
    subject: str, liked_ids: List[str], data3_glob: str
) -> Tuple[List[str], np.ndarray, List[str], List[str]]:
    """
    データ3: 好み映画IDごとの Experiment_movieID_{id}.xlsx を収集し、
    4列目（「映画の推薦レビューである」）の **全映画を通した平均** を閾値にして y_true を作成。
    戻り値: texts, y_true, movie_ids_per_row, src_files_per_row
    """
    texts_all: List[str] = []
    scores_all: List[float] = []
    mids_all: List[str] = []
    srcs_all: List[str] = []

    tmp_rows = []
    for mid in liked_ids:
        path = data3_glob.format(subject=str(subject), movie_id=str(mid))
        files = glob.glob(path)
        if not files:
            continue
        for xp in files:
            try:
                df = pd.read_excel(xp)
                if df.shape[1] < 4:
                    print(f"[WARN] 4列未満: {xp} -> スキップ")
                    continue
                text_col = df.columns[0]      # 本文
                score_col = df.columns[3]     # 推薦度

                texts = df[text_col].astype(str)
                scores = pd.to_numeric(df[score_col], errors="coerce")

                for t, s in zip(texts.tolist(), scores.tolist()):
                    t = normalize_text(t)
                    if not t or pd.isna(s):  # スコア欠損は除外
                        continue
                    tmp_rows.append((t, float(s), str(mid), os.path.basename(xp)))
            except Exception as e:
                print(f"[WARN] skip {xp}: {e}")

    if not tmp_rows:
        raise ValueError("データ3（評価用）が空です。パスや列数をご確認ください。")

    global_mean = float(np.mean([row[1] for row in tmp_rows]))
    for t, s, mid, src in tmp_rows:
        texts_all.append(t)
        scores_all.append(s)
        mids_all.append(mid)
        srcs_all.append(src)

    y_true = np.array([1 if s >= global_mean else 0 for s in scores_all], dtype=int)
    return texts_all, y_true, mids_all, srcs_all


# ---------------- LLM（OpenAI ChatGPT）版 ----------------
LLM_MODEL_DEFAULT = "gpt-4o-mini"

def summarize_rubric(client: OpenAI, model: str, positives: List[str], negatives: List[str], max_each=150) -> str:
    """
    ChatGPTに正例・負例のレビュー群を与え、嗜好ルーブリックを抽出。
    - 正例/負例を対等に扱う
    - どちらの群でも「映画を肯定的に評価している」レビューを中心に参照
    - 正例/負例は分類ラベルであり、内容の極性（ポジ/ネガ）ではないことを明記
    """
    pos = "\n---\n".join(positives[:max_each])
    neg = "\n---\n".join(negatives[:max_each])

    # prompt = (
    #     "あなたは日本語映画レビューの嗜好分析の専門家です。\n"
    #     "以下の2群（好みの映画／好みでない映画）のレビュー例を分析し、"
    #     "それぞれの群において、映画を肯定的に評価しているレビューを中心に参照しながら、"
    #     "各群に見られる特徴をできるだけ多く、網羅的に挙げてください。\n"
    #     "出力は次のJSON形式で、好み映画・好みでない映画の両方について同等の粒度で特徴を示してください。\n\n"
    #     "{\n"
    #     '  "好み映画特徴": [ "<短い日本語フレーズ1>", "<短い日本語フレーズ2>", ... ],\n'
    #     '  "好みでない映画特徴": [ "<短い日本語フレーズ1>", "<短い日本語フレーズ2>", ... ]\n'
    #     "}\n\n"
    #     "重要：ここでいう『正例』『負例』は二値分類上のクラス名であり、\n"
    #     "『正例＝ポジティブな内容』『負例＝ネガティブな内容』という意味ではありません。\n"
    #     "あくまで、好み映画（正例群）と好みでない映画（負例群）というラベルを区別するために用いています。\n"
    #     "したがって、どちらの群においても、映画を肯定的に評価しているレビューを参照してください。\n\n"
    #     "特徴数の上限は設けず、レビュー全体に共通する傾向を抽出してください。\n"
    #     "文体・語彙・内容・感情・評価軸など、あらゆる観点から要約してください。\n\n"
    #     "【好み映画のレビュー（正例）】\n" + pos + "\n\n"
    #     "【好みでない映画のレビュー（負例）】\n" + neg
    # )
    
    prompt = (
        "あなたは日本語映画レビューの嗜好分析の専門家です。\n"
        "以下の2群（嗜好映画A群・嗜好映画B群）のレビュー例を分析し、"
        "各群でユーザが『映画を肯定的に評価していると思われる部分』のみを参照してください。\n"
        "特に、映画の“特徴”に着目して抽出してください。特徴とは、監督名、俳優名、登場人物名、シリーズ名、"
        "用語・ガジェット（例: T-1000, ライトセーバー）、ジャンル（SF/サスペンス等）、"
        "演出・音楽・カメラワーク・VFX、世界観・テーマ、年代/製作国 など固有/半固有の要素を指します。\n"
        "また、その特徴をユーザが『どのように好む／重視する／魅力を感じる』のかという嗜好の向きも併せて表現してください。\n"
        "出力は次のJSON形式で、両群とも同等の粒度で、短い日本語フレーズを多数列挙してください。\n\n"
        "{\n"
        '  "嗜好映画A群特徴": [ "<特徴> を <好み方>", "<特徴> を <好み方>", ... ],\n'
        '  "嗜好映画B群特徴": [ "<特徴> を <好み方>", "<特徴> を <好み方>", ... ]\n'
        "}\n\n"
        "重要1：どちらの群でも“肯定的に評価しているレビュー”を参照してください。\n"
        "重要2：価値判断語（悪い/単調/退屈/期待外れ 等）は使用せず、中立・記述的に表現してください。"
        "必要な場合は『〜を重視しない/控えめ/よりXを志向』など肯定的パラフレーズに置換してください。\n"
        "重要3：抽象語だけでなく、可能な限り固有名や具体語（監督・役名・用語）を含めてください。"
        "例：『ジェームズ・キャメロンの緻密な演出を高く評価』『T-1000の無機質な追跡表現に緊張感を好む』\n"
        "特徴数の上限は設けません。文体・語彙・内容・感情の方向性・演出・構成・映像表現など、"
        "嗜好を示す具体的属性にフォーカスしてください。\n\n"
        "【嗜好映画A群】\n" + pos + "\n\n"
        "【嗜好映画B群】\n" + neg
    )

    resp = client.chat.completions.create(
        model=model, temperature=0,
        messages=[{"role":"user","content":prompt}],
    )
    return resp.choices[0].message.content.strip()

def llm_score(client: OpenAI, model: str, rubric: str, reviews: List[str], batch=20, sleep=0.35) -> np.ndarray:
    """
    ルーブリックに基づき各レビューを 0-100 の整数でスコアリング。
    1行1数値のみで返すようプロンプト指定。→ 0〜1に正規化して返す。
    """
    scores = []
    for i in range(0, len(reviews), batch):
        chunk = reviews[i:i+batch]
        prompt = (
            "以下の嗜好ルーブリックに基づき、各レビューが推奨にどれだけ合致するかを0-100の整数で評価し、"
            "各行に1つずつ数値のみを出力してください。説明や記号は禁止です。\n"
            f"【ルーブリック】\n{rubric}\n\n" + "\n-----\n".join(chunk)
        )
        resp = client.chat.completions.create(
            model=model, temperature=0,
            messages=[{"role":"user","content":prompt}],
        )
        lines = [ln.strip() for ln in resp.choices[0].message.content.strip().splitlines() if ln.strip()]
        for ln in lines:
            m = re.search(r"(\d{1,3})", ln)
            scores.append(min(100, max(0, int(m.group(1)))) if m else 0)
        time.sleep(sleep)
    return np.array(scores[:len(reviews)], dtype=float)/100.0


# ---------------- ローカル（SVM）版 ----------------
def train_local_svm(texts_train: List[str], y_train: List[int]) -> Pipeline:
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=40000, ngram_range=(1,2), min_df=2)),
        ("clf", LinearSVC())
    ])
    pipe.fit(texts_train, y_train)
    return pipe

def local_score(pipe: Pipeline, texts: List[str]) -> np.ndarray:
    df = pipe.decision_function(texts)
    mx, mn = float(np.max(df)), float(np.min(df))
    if mx == mn:
        return np.ones_like(df) * 0.5
    return (df - mn) / (mx - mn)


# ---------------- しきい値最適化 ----------------
def choose_threshold_by_max_f1(scores: np.ndarray, y_true: np.ndarray):
    cand = np.unique(np.round(scores, 3))
    best = (0.0, 0.0, 0.0, 0.5)  # P,R,F1,thr
    for thr in cand:
        yp = (scores >= thr).astype(int)
        p,r,f,_ = precision_recall_fscore_support(y_true, yp, average="binary", zero_division=0)
        if f > best[2]:
            best = (p,r,f,float(thr))
    return best


# ---------------- 対話入力 ----------------
def get_user_inputs():
    print("被験者番号を入力してください（例: 1）", flush=True)
    subject = input("subject: ").strip()
    while not re.fullmatch(r"\d+", subject):
        subject = input("数字で入力してください（例: 1）: ").strip()

    print("\n実行モードを選択してください：", flush=True)
    print("  1) 従来方式（映画IDベースでデータ2から学習 → 出力: ./実験結果/被験者{subject}GPT/比較手法）", flush=True)
    print("  2) 提案手法1（ランキングtxtから学習 → 出力: ./実験結果/被験者{subject}GPT/提案手法1/{N}）", flush=True)
    mode = input("番号を入力してください [1/2]: ").strip()
    while mode not in ("1","2"):
        mode = input("無効な入力です。1 または 2 を指定してください: ").strip()

    out_prefix = f"sub{subject}_run1"

    print(f"\n[INFO] 被験者番号: {subject}")
    print(f"[INFO] 実行モード: {mode}")
    print(f"[INFO] 出力プレフィックス: {out_prefix}\n")

    return subject, mode, out_prefix


# ---------------- 共通：評価・保存ルーチン ----------------
def evaluate_and_save(out_dir: str, out_prefix: str, scores: np.ndarray, y_true: np.ndarray,
                      mids_per_row: List[str], srcs_per_row: List[str], X_eval: List[str],
                      rubric_text: Optional[str]=None, rubric_name: Optional[str]=None):
    os.makedirs(out_dir, exist_ok=True)

    P,R,F1,thr = choose_threshold_by_max_f1(scores, y_true)
    y_pred = (scores >= thr).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    print("\n=== Evaluation (max F1 threshold) ===")
    print(f"Precision: {P:.4f}  Recall: {R:.4f}  F1: {F1:.4f}  | thr={thr:.3f}")
    print("Confusion matrix [rows=true 0/1, cols=pred 0/1]:")
    print(cm)
    print("\nDetailed report:\n", classification_report(y_true, y_pred, digits=4))

    pred_path = os.path.join(out_dir, f"{out_prefix}_predictions.xlsx")
    out_pred = pd.DataFrame({
        "movie_id": mids_per_row,
        "src_file": srcs_per_row,
        "review_text": X_eval,
        "y_true": y_true,
        "score": scores,
        "y_pred": y_pred
    })
    out_pred.to_excel(pred_path, index=False)

    prtbl_path = os.path.join(out_dir, f"{out_prefix}_pr_table.xlsx")
    rows = []
    for t in np.linspace(scores.min(), scores.max(), 21):
        yp = (scores >= t).astype(int)
        p, r, f, _ = precision_recall_fscore_support(y_true, yp, average="binary", zero_division=0)
        rows.append({"しきい値": float(t), "適合率": p, "再現率": r, "F値": f})
    pd.DataFrame(rows).to_excel(prtbl_path, index=False)

    if rubric_text and rubric_name:
        rubric_path = os.path.join(out_dir, f"{out_prefix}_{rubric_name}.txt")
        with open(rubric_path, "w", encoding="utf-8") as f:
            f.write(rubric_text)
        print("[INFO] ルーブリックを保存:", rubric_path)

    print(f"\n[DONE] 保存先: {out_dir}")
    print(f"      予測: {pred_path}")
    print(f"      PR表: {prtbl_path}")


# ---------------- メイン ----------------
def main():
    print("[BOOT] ChatGPT.py started.", flush=True)
    subject, mode, out_prefix = get_user_inputs()

    # パス
    data2_glob = "./極性付きレビューファイル/{movie_id}.xlsx"
    data3_glob = "./被験者{subject}/既知の映画群/Experiment_movieID_{movie_id}.xlsx"

    # 好み/非好みの映画ID（評価データ3で使用）
    liked_path = f"./movies/{subject}_liked_movies.txt"
    disliked_path = f"./movies/{subject}_disliked_movies.txt"
    liked_ids = read_lines(liked_path) if os.path.exists(liked_path) else []
    disliked_ids = read_lines(disliked_path) if os.path.exists(disliked_path) else []
    if not liked_ids:
        raise FileNotFoundError(f"好みIDファイルが見つかりません: {liked_path}")
    if not disliked_ids:
        print(f"[WARN] 非好みIDファイルが見つかりません: {disliked_path} (空として続行)")

    # ② 評価データ（全体平均で y_true 作成）— mode 共通
    X_eval, y_true, mids_per_row, srcs_per_row = load_data3_eval_global_mean(subject, liked_ids, data3_glob)
    print(f"[INFO] 評価レビュー: {len(X_eval)} 件  (ファイル数: {len(set(srcs_per_row))})")

    use_llm = (os.getenv("OPENAI_API_KEY") is not None) and _HAS_OPENAI

    # ---------------- Mode 1: 従来方式（比較手法） ----------------
    if mode == "1":
        # 出力先：比較手法
        out_dir = os.path.join(".", "実験結果", f"被験者{subject}GPT", "比較手法")
        os.makedirs(out_dir, exist_ok=True)

        # 学習（極性>=1 既定ON）
        X_train, y_train = load_data2_train(liked_ids, disliked_ids, data2_glob, use_only_positive=True)
        print(f"[INFO] 学習レビュー: {len(X_train)} 件 (pos={sum(y_train)} / neg={len(y_train)-sum(y_train)})")

        if use_llm:
            print("[INFO] OpenAI LLM で分類します。")
            client = OpenAI()
            pos_examples = [t for t,l in zip(X_train, y_train) if l==1][:200]
            neg_examples = [t for t,l in zip(X_train, y_train) if l==0][:200]
            rubric = summarize_rubric(client, LLM_MODEL_DEFAULT, pos_examples, neg_examples)
            scores = llm_score(client, LLM_MODEL_DEFAULT, rubric, X_eval)
            evaluate_and_save(out_dir, f"{out_prefix}_mode1_idtrain",
                              scores, y_true, mids_per_row, srcs_per_row, X_eval,
                              rubric_text=rubric, rubric_name="rubric")
        else:
            print("[INFO] ローカルSVMで分類します（LLM未使用）。")
            pipe = train_local_svm(X_train, y_train)
            scores = local_score(pipe, X_eval)
            evaluate_and_save(out_dir, f"{out_prefix}_mode1_idtrain",
                              scores, y_true, mids_per_row, srcs_per_row, X_eval)

        return

    # ---------------- Mode 2: 提案手法1（ランキングtxt・N反復） ----------------
    # 出力先：提案手法1/{N}
    pos_texts, neg_texts = load_rank_reviews_split(subject)
    Pn, Nn = len(pos_texts), len(neg_texts)
    print(f"[INFO] ランキング由来（正例={Pn} / 負例={Nn}）")

    # 実行する N の上限を決定
    N_max = min(500, (max(Pn, Nn) // 100) * 100)
    Ns = [n for n in [100,200,300,400,500] if n <= N_max]
    if not Ns:
        # 100未満でも何かしら動かしたい場合は最小 100 で試すが、仕様通りならここで停止
        raise ValueError(f"十分なレビュー数がありません（正例={Pn}, 負例={Nn}）。少なくともどちらかが100以上必要です。")

    if use_llm:
        client = OpenAI()

    for N in Ns:
        out_dir = os.path.join(".", "実験結果", f"被験者{subject}GPT", "提案手法1", f"{N}")
        os.makedirs(out_dir, exist_ok=True)

        use_pos = pos_texts[:min(N, Pn)]
        use_neg = neg_texts[:min(N, Nn)]
        X_train = use_pos + use_neg
        y_train = [1]*len(use_pos) + [0]*len(use_neg)

        print(f"\n[INFO] N={N} 学習レビュー: {len(X_train)} 件 (pos={len(use_pos)} / neg={len(use_neg)})")
        run_prefix = f"{out_prefix}_mode2_ranktrain_N{N}"

        if use_llm:
            pos_examples = use_pos[:200]
            neg_examples = use_neg[:200]
            rubric = summarize_rubric(client, LLM_MODEL_DEFAULT, pos_examples, neg_examples)
            scores = llm_score(client, LLM_MODEL_DEFAULT, rubric, X_eval)
            evaluate_and_save(out_dir, run_prefix,
                              scores, y_true, mids_per_row, srcs_per_row, X_eval,
                              rubric_text=rubric, rubric_name="rubric")
        else:
            print("[INFO] ローカルSVMで分類します（LLM未使用）。")
            pipe = train_local_svm(X_train, y_train)
            scores = local_score(pipe, X_eval)
            evaluate_and_save(out_dir, run_prefix,
                              scores, y_true, mids_per_row, srcs_per_row, X_eval)

    print("\n[ALL DONE] 提案手法1の全 N 実験を完了しました。")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback as tb
        print("[FATAL] Uncaught exception:", e, flush=True)
        tb.print_exc()
        input("エラーが発生しました。詳細を表示しました。何かキーを押すと終了します...")
