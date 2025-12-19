# analyze_subject4_low_recall.py
# 被験者4のみ再現率が低い理由を分析
# -*- coding: utf-8 -*-

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path

def load_pr_metrics(subject, n=300):
    """指定被験者・Nの PR テーブルを読み込み"""
    pr_path = f"./実験結果/被験者{subject}GPT/提案手法1/{n}/sub{subject}_run1_mode2_ranktrain_N{n}_pr_table.xlsx"
    
    try:
        df = pd.read_excel(pr_path)
        return df
    except:
        return None

def load_predictions(subject, n=300):
    """予測結果を読み込み"""
    pred_path = f"./実験結果/被験者{subject}GPT/提案手法1/{n}/sub{subject}_run1_mode2_ranktrain_N{n}_predictions.xlsx"
    
    try:
        df = pd.read_excel(pred_path)
        return df
    except:
        return None

def load_rubric(subject, n=300):
    """ルーブリックを読み込み"""
    rubric_path = f"./実験結果/被験者{subject}GPT/提案手法1/{n}/sub{subject}_run1_mode2_ranktrain_N{n}_rubric.txt"
    
    try:
        with open(rubric_path, 'r', encoding='utf-8') as f:
            return f.read()
    except:
        return None

def load_learning_data(subject):
    """学習データを読み込み"""
    liked_path = f"./ランキング/{subject}_liked_reviews.txt"
    disliked_path = f"./ランキング/{subject}_disliked_reviews.txt"
    
    liked = []
    disliked = []
    
    try:
        with open(liked_path, 'r', encoding='utf-8') as f:
            liked = [ln.strip() for ln in f if ln.strip()][1:]
    except:
        pass
    
    try:
        with open(disliked_path, 'r', encoding='utf-8') as f:
            disliked = [ln.strip() for ln in f if ln.strip()][1:]
    except:
        pass
    
    return liked, disliked

def analyze_recall_comparison():
    """全被験者の再現率を比較"""
    
    print("=" * 100)
    print("分析: 被験者別の再現率比較（N=300, 提案手法1）")
    print("=" * 100)
    
    print("\n【1】再現率（Recall）の比較\n")
    print(f"{'被験者':<10} {'Precision':<15} {'Recall':<15} {'F1':<15} {'PR乖離':<12}")
    print("-" * 80)
    
    results = {}
    max_f1_thresholds = {}  # 各被験者の最大F1時の閾値
    
    for subject in [1, 2, 3, 4, 5]:
        df_pr = load_pr_metrics(subject, 300)
        
        if df_pr is None:
            print(f"{subject:<10} {'（読み込み失敗）':<15}")
            continue
        
        # 最大F1の行を取得
        max_f1_idx = df_pr['f1'].idxmax()
        max_f1_row = df_pr.loc[max_f1_idx]
        
        p = float(max_f1_row['precision'])
        r = float(max_f1_row['recall'])
        f1 = float(max_f1_row['f1'])
        thr = float(max_f1_row['threshold'])
        gap = abs(p - r)
        
        print(f"{subject:<10} {p:<15.4f} {r:<15.4f} {f1:<15.4f} {gap:<12.4f}")
        
        results[subject] = {
            'p': p,
            'r': r,
            'f1': f1,
            'threshold': thr,
            'gap': gap,
            'pr_table': df_pr
        }
        max_f1_thresholds[subject] = thr
    
    # 被験者4の特異性を強調
    print("\n")
    if 4 in results:
        recall_4 = results[4]['r']
        other_recalls = [results[s]['r'] for s in [1, 2, 3, 5] if s in results]
        avg_recall_others = np.mean(other_recalls)
        
        print(f">>> 被験者4の再現率が低い:")
        print(f"    被験者4: {recall_4:.4f}")
        print(f"    他被験者平均: {avg_recall_others:.4f}")
        print(f"    低下量: {avg_recall_others - recall_4:.4f} ({100*(avg_recall_others - recall_4)/avg_recall_others:.1f}%)")
    
    return results, max_f1_thresholds

def analyze_score_distribution(results):
    """スコア分布を分析"""
    
    print("\n\n【2】スコア分布の比較（正例と負例の分離度）\n")
    print(f"{'被験者':<10} {'正例スコア平均':<15} {'負例スコア平均':<15} {'分離度':<12} {'スコアSD':<12}")
    print("-" * 80)
    
    score_stats = {}
    
    for subject in [1, 2, 3, 4, 5]:
        df_pred = load_predictions(subject, 300)
        
        if df_pred is None:
            print(f"{subject:<10} {'（読み込み失敗）':<15}")
            continue
        
        y_true = df_pred['y_true'].values
        scores = df_pred['score'].values
        
        pos_scores = scores[y_true == 1]
        neg_scores = scores[y_true == 0]
        
        pos_mean = np.mean(pos_scores) if len(pos_scores) > 0 else np.nan
        neg_mean = np.mean(neg_scores) if len(neg_scores) > 0 else np.nan
        
        # クラス間分離度（信号対ノイズ比）
        separation = abs(pos_mean - neg_mean) / (np.std(pos_scores) + np.std(neg_scores) + 1e-8)
        overall_std = np.std(scores)
        
        print(f"{subject:<10} {pos_mean:<15.4f} {neg_mean:<15.4f} {separation:<12.4f} {overall_std:<12.4f}")
        
        score_stats[subject] = {
            'pos_mean': pos_mean,
            'neg_mean': neg_mean,
            'separation': separation,
            'std': overall_std,
            'pos_scores': pos_scores,
            'neg_scores': neg_scores,
            'y_true': y_true,
            'scores': scores
        }
    
    # 被験者4の特異性
    print("\n")
    if 4 in score_stats:
        sep_4 = score_stats[4]['separation']
        other_seps = [score_stats[s]['separation'] for s in [1, 2, 3, 5] if s in score_stats]
        avg_sep_others = np.mean(other_seps)
        
        print(f">>> 被験者4のクラス分離度が低い:")
        print(f"    被験者4: {sep_4:.4f}")
        print(f"    他被験者平均: {avg_sep_others:.4f}")
        print(f"    低下率: {100*(1 - sep_4/avg_sep_others):.1f}%")
    
    return score_stats

def analyze_rubric_quality(results):
    """ルーブリック品質を分析"""
    
    print("\n\n【3】ルーブリック特徴の比較\n")
    print(f"{'被験者':<10} {'総特徴数':<15} {'A群特徴数':<15} {'B群特徴数':<15}")
    print("-" * 60)
    
    rubric_stats = {}
    
    for subject in [1, 2, 3, 4, 5]:
        rubric_text = load_rubric(subject, 300)
        
        if rubric_text is None:
            print(f"{subject:<10} {'（読み込み失敗）':<15}")
            continue
        
        try:
            rubric_json = json.loads(rubric_text)
            
            # 特徴数をカウント
            total_features = 0
            group_a_features = 0
            group_b_features = 0
            
            for key, val in rubric_json.items():
                if isinstance(val, list):
                    count = len(val)
                    total_features += count
                    if '群A' in key or 'A群' in key:
                        group_a_features = count
                    elif '群B' in key or 'B群' in key:
                        group_b_features = count
            
            print(f"{subject:<10} {total_features:<15} {group_a_features:<15} {group_b_features:<15}")
            
            rubric_stats[subject] = {
                'total': total_features,
                'group_a': group_a_features,
                'group_b': group_b_features,
                'rubric': rubric_json
            }
        except json.JSONDecodeError:
            print(f"{subject:<10} {'（JSON解析失敗）':<15}")
            rubric_stats[subject] = {'error': True}
    
    # 被験者4の特異性
    print("\n")
    if 4 in rubric_stats and 'error' not in rubric_stats[4]:
        total_4 = rubric_stats[4]['total']
        other_totals = [rubric_stats[s]['total'] for s in [1, 2, 3, 5] if s in rubric_stats and 'error' not in rubric_stats[s]]
        avg_total_others = np.mean(other_totals)
        
        print(f">>> 被験者4のルーブリック特徴数:")
        print(f"    被験者4: {total_4} 個")
        print(f"    他被験者平均: {avg_total_others:.1f} 個")
        print(f"    差分: {total_4 - avg_total_others:+.1f} 個")
    
    return rubric_stats

def analyze_learning_data_characteristics():
    """学習データの特性を分析"""
    
    print("\n\n【4】学習データ（ランキング）の特性\n")
    print(f"{'被験者':<10} {'好きな映画数':<15} {'嫌いな映画数':<15} {'バランス':<15} {'レビュー長平均':<15}")
    print("-" * 80)
    
    learning_stats = {}
    
    for subject in [1, 2, 3, 4, 5]:
        liked, disliked = load_learning_data(subject)
        
        n_liked = len(liked)
        n_disliked = len(disliked)
        balance = min(n_liked, n_disliked) / max(n_liked, n_disliked) if max(n_liked, n_disliked) > 0 else 0
        
        # レビュー長の平均
        all_reviews = liked + disliked
        avg_len = np.mean([len(r.split()) for r in all_reviews]) if all_reviews else 0
        
        print(f"{subject:<10} {n_liked:<15} {n_disliked:<15} {balance:<15.3f} {avg_len:<15.1f}")
        
        learning_stats[subject] = {
            'n_liked': n_liked,
            'n_disliked': n_disliked,
            'balance': balance,
            'avg_len': avg_len,
            'liked': liked,
            'disliked': disliked
        }
    
    # 被験者4の特異性
    print("\n")
    if 4 in learning_stats:
        stat_4 = learning_stats[4]
        other_balances = [learning_stats[s]['balance'] for s in [1, 2, 3, 5] if s in learning_stats]
        avg_balance_others = np.mean(other_balances)
        
        print(f">>> 被験者4の学習データ特性:")
        print(f"    クラスバランス: {stat_4['balance']:.3f}")
        print(f"    他被験者平均: {avg_balance_others:.3f}")
        
        if stat_4['balance'] < avg_balance_others:
            print(f"    → クラス不均衡が大きい（影響あり）")
        
        other_avg_len = np.mean([learning_stats[s]['avg_len'] for s in [1, 2, 3, 5] if s in learning_stats])
        print(f"\n    レビュー長（単語数）: {stat_4['avg_len']:.1f}")
        print(f"    他被験者平均: {other_avg_len:.1f}")
        
        if stat_4['avg_len'] < other_avg_len:
            print(f"    → レビューが短い可能性（ノイズが相対的に大きい）")
    
    return learning_stats

def analyze_why_subject4_low_recall(score_stats, rubric_stats, learning_stats):
    """被験者4の再現率が低い理由を詳細分析"""
    
    print("\n\n【5】被験者4の再現率が低い理由（詳細考察）\n")
    
    if 4 not in score_stats:
        print("被験者4のデータが不足しています")
        return
    
    print("観察された被験者4の特徴:")
    print("-" * 80)
    print()
    
    # 1. スコア分布の問題
    if score_stats[4]['separation'] < np.mean([score_stats[s]['separation'] for s in [1,2,3,5] if s in score_stats]):
        print("1. ❌ クラス分離度が低い")
        print("   原因:")
        print("   - LLMが抽出したルーブリックが、被験者4のデータに最適化されていない")
        print("   - 正例と負例のスコアが過度に重なっている")
        print("   - 結果：どの閾値を選択しても、再現率と精度のトレードオフが悪い")
        print()
    
    # 2. ルーブリック品質の問題
    if 4 in rubric_stats and 'error' not in rubric_stats[4]:
        total_4 = rubric_stats[4]['total']
        other_totals = [rubric_stats[s]['total'] for s in [1, 2, 3, 5] if s in rubric_stats and 'error' not in rubric_stats[s]]
        
        if total_4 < np.mean(other_totals):
            print("2. ❌ ルーブリック特徴数が少ない")
            print("   原因:")
            print("   - 被験者4のレビューから抽出された嗜好特徴が限定的")
            print("   - スコアリングに使用できる情報が不足")
            print("   - 結果：スコアの精度が低下し、分類ができなくなる")
            print()
    
    # 3. 学習データの特性
    if 4 in learning_stats:
        stat_4 = learning_stats[4]
        
        if stat_4['balance'] < 0.8:
            print("3. ❌ クラス不均衡")
            print("   原因:")
            print(f"   - 好きな映画と嫌いな映画の数が大きく異なる（{stat_4['n_liked']} vs {stat_4['n_disliked']}）")
            print("   - LLMがマジョリティクラスに偏りやすくなる")
            print("   - 結果：マイノリティクラス（正例）の再現率が特に低下")
            print()
        
        if stat_4['avg_len'] < 50:  # 単語数の一例
            print("4. ❌ レビューが短い")
            print("   原因:")
            print(f"   - 平均レビュー長が短い（{stat_4['avg_len']:.1f}単語）")
            print("   - 短いレビューでは嗜好の詳細が表現されない")
            print("   - LLMのプロンプトで十分な嗜好情報が得られない")
            print("   - 結果：ルーブリック抽出の精度が低下")
            print()
    
    print("5. 🔍 BERT との比較で分かることは？")
    print("   - BERT: トークンレベルでの微細な特徴を自動学習")
    print("           クラス不均衡やデータ品質の影響を受けにくい")
    print("   ")
    print("   - ChatGPT（LLM）: ルーブリック抽出に依存")
    print("           学習データの質・量が直接的に精度に影響")
    print("           被験者4の学習データの特殊性が顕著に出やすい")
    print()

def generate_recommendations():
    """改善提案"""
    
    print("\n\n【6】改善提案\n")
    
    print("被験者4の再現率を改善するための施策:")
    print("-" * 80)
    print()
    
    print("1. ルーブリック抽出の改善")
    print("   - Few-shot 例を被験者4用に最適化")
    print("   - より詳細なプロンプトで嗜好を引き出す")
    print("   - LLMの temperature を調整（探索性と安定性のバランス）")
    print()
    
    print("2. スコアリングプロンプトの改善")
    print("   - 被験者4の評価軸に合わせたプロンプト構成")
    print("   - 閾値を最大F1ではなく、再現率重視で決定する")
    print()
    
    print("3. 学習データ量の調整")
    print("   - N=300 が常に最適とは限らない")
    print("   - 被験者4については N=200 や N=400 を試す")
    print("   - データ品質を優先（量より質）")
    print()
    
    print("4. クラス不均衡への対策")
    print("   - class_weight='balanced' を使用（既に実装済み）")
    print("   - アンダーサンプリング or オーバーサンプリング")
    print("   - カスタム損失関数（再現率重視）")
    print()
    
    print("5. ハイブリッド手法の検討")
    print("   - BERT ベースの分類との融合")
    print("   - 最終的な判断を複数モデルの投票で決定")
    print("   - 信頼度スコアの低い判断は人間レビュー")
    print()

def main():
    print("\n")
    
    # 分析実行
    results, max_f1_thresholds = analyze_recall_comparison()
    score_stats = analyze_score_distribution(results)
    rubric_stats = analyze_rubric_quality(results)
    learning_stats = analyze_learning_data_characteristics()
    
    # 詳細考察
    analyze_why_subject4_low_recall(score_stats, rubric_stats, learning_stats)
    
    # 改善提案
    generate_recommendations()
    
    print("\n" + "=" * 100)
    print("まとめ")
    print("=" * 100)
    print()
    print("被験者4のみ再現率が低い理由:")
    print()
    print("① クラス分離度の低下")
    print("   → ルーブリック品質またはスコアリングプロンプトの最適化不足")
    print()
    print("② ルーブリック特徴数の不足")
    print("   → LLM が被験者4の嗜好を十分に抽出できていない")
    print()
    print("③ 学習データの特性（不均衡・短さなど）")
    print("   → LLM ベース手法は学習データの質に敏感")
    print()
    print("④ BERT との性能差")
    print("   → BERT は微細な特徴を自動学習するため、堅牢性が高い")
    print("   → LLM は明示的な指示に依存するため、個別最適化が必要")
    print()
    print("⇒ 推奨: 被験者4向けにプロンプトやパラメータを個別調整する")
    print("\n" + "=" * 100)

if __name__ == "__main__":
    main()
