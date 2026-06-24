"""Bhagavad Gita retrieval-based question-answering model."""

import os
import re
import json
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity


class GitaQAModel:
    """Retrieval-based Bhagavad Gita question-answering model."""

    def __init__(self):
        self.verse_vectorizer = None
        self.qa_vectorizer = None
        self.verses_df = None
        self.train_qa = None
        self.test_qa = None
        self.verse_matrix = None
        self.qa_matrix = None
        self.verse_qa_idx = None
        self.metrics = {}

    def fit(self, verses_df, qa_df, test_size=0.2, random_state=42):
        self.verses_df = verses_df.copy()
        self.verses_df['doc_id'] = self.verses_df.apply(
            lambda r: f"{int(r['chapter'])}.{int(r['verse'])}", axis=1
        )
        self.verses_df['text'] = (
            'chapter ' + self.verses_df['chapter'].astype(str) +
            ' verse ' + self.verses_df['verse'].astype(str) + ' ' +
            self.verses_df['english'] + ' ' + self.verses_df['hindi']
        )

        train_qa, test_qa = train_test_split(qa_df, test_size=test_size, random_state=random_state)
        self.train_qa = train_qa.copy()
        self.test_qa = test_qa.copy()
        self.train_qa['verse_key'] = self.train_qa.apply(
            lambda r: f"{int(r['chapter_no'])}.{int(r['verse_no'])}", axis=1
        )

        self.verse_vectorizer = TfidfVectorizer(
            max_features=8000, stop_words='english', ngram_range=(1, 3), sublinear_tf=True
        )
        self.verse_matrix = self.verse_vectorizer.fit_transform(self.verses_df['text'])

        self.qa_vectorizer = TfidfVectorizer(
            max_features=8000, stop_words='english', ngram_range=(1, 2), sublinear_tf=True
        )
        self.qa_matrix = self.qa_vectorizer.fit_transform(self.train_qa['question'])

        self.verse_qa_idx = {
            k: self.train_qa.index[self.train_qa['verse_key'] == k].tolist()
            for k in self.train_qa['verse_key'].unique()
        }
        return self

    def ask(self, question, top_k_verses=3):
        q_vec = self.verse_vectorizer.transform([question])
        sims = cosine_similarity(q_vec, self.verse_matrix)[0]
        top_verse_idx = np.argsort(sims)[::-1][:top_k_verses]

        verses = []
        for vi in top_verse_idx:
            row = self.verses_df.iloc[vi]
            verses.append({
                'chapter': int(row['chapter']),
                'verse': int(row['verse']),
                'english': row['english'],
                'hindi': row['hindi'],
                'sanskrit': row['sanskrit'],
                'score': float(sims[vi]),
            })

        best_score, best_answer, best_q = -1, '', ''
        for v in verses:
            vk = f"{v['chapter']}.{v['verse']}"
            if vk not in self.verse_qa_idx:
                continue
            idxs = self.verse_qa_idx[vk]
            sub_matrix = self.qa_vectorizer.transform(self.train_qa.loc[idxs, 'question'])
            q_sims = cosine_similarity(self.qa_vectorizer.transform([question]), sub_matrix)[0]
            j = int(np.argmax(q_sims))
            if q_sims[j] > best_score:
                best_score = q_sims[j]
                best_answer = self.train_qa.loc[idxs[j], 'answer']
                best_q = self.train_qa.loc[idxs[j], 'question']

        if not best_answer:
            q_sims = cosine_similarity(self.qa_vectorizer.transform([question]), self.qa_matrix)[0]
            j = int(np.argmax(q_sims))
            best_answer = self.train_qa.iloc[j]['answer']
            best_q = self.train_qa.iloc[j]['question']

        return {
            'question': question,
            'answer': best_answer,
            'matched_question': best_q,
            'verses': verses,
        }

    @staticmethod
    def _token_f1(pred, ref):
        pred_tokens = set(re.findall(r'\w+', pred.lower()))
        ref_tokens = set(re.findall(r'\w+', ref.lower()))
        if not pred_tokens or not ref_tokens:
            return 0.0
        common = pred_tokens & ref_tokens
        if not common:
            return 0.0
        p = len(common) / len(pred_tokens)
        r = len(common) / len(ref_tokens)
        return 2 * p * r / (p + r)

    def evaluate(self):
        verse_top1 = verse_top3 = 0
        answer_f1_total = answer_sim_total = 0
        verse_correct_f1 = []

        for _, row in self.test_qa.iterrows():
            target = f"{int(row['chapter_no'])}.{int(row['verse_no'])}"
            result = self.ask(row['question'])
            retrieved = [f"{v['chapter']}.{v['verse']}" for v in result['verses']]

            if retrieved[0] == target:
                verse_top1 += 1
            if target in retrieved[:3]:
                verse_top3 += 1

            pred, ref = result['answer'], row['answer']
            f1 = self._token_f1(pred, ref)
            sim = cosine_similarity(
                self.qa_vectorizer.transform([pred]),
                self.qa_vectorizer.transform([ref]),
            )[0][0]
            answer_f1_total += f1
            answer_sim_total += sim
            if target in retrieved[:3]:
                verse_correct_f1.append(f1)

        n = len(self.test_qa)
        self.metrics = {
            'test_samples': n,
            'train_samples': len(self.train_qa),
            'verse_top1_accuracy': round(verse_top1 / n * 100, 2),
            'verse_top3_accuracy': round(verse_top3 / n * 100, 2),
            'answer_token_f1': round(answer_f1_total / n * 100, 2),
            'answer_cosine_similarity': round(answer_sim_total / n * 100, 2),
            'answer_f1_when_verse_top3': round(np.mean(verse_correct_f1) * 100, 2) if verse_correct_f1 else 0,
            'overall_knowledge_score': round((verse_top3 / n * 0.4 + answer_f1_total / n * 0.6) * 100, 2),
        }
        return self.metrics

    def save(self, path='gita_model'):
        os.makedirs(path, exist_ok=True)
        joblib.dump(self, os.path.join(path, 'gita_qa_model.joblib'))
        with open(os.path.join(path, 'metrics.json'), 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"Model saved to {path}/")

    @staticmethod
    def load(path='gita_model'):
        return joblib.load(os.path.join(path, 'gita_qa_model.joblib'))
