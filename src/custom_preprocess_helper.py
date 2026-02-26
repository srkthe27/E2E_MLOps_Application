import re
import nltk
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

class CustomNLPDatasetOp:
    def __init__(self,df,text_col,target_col=None,preprocessor=None):
        self.df = df
        self.text_col = text_col
        self.target_col = target_col
        self.preprocessor = preprocessor

    def _dataset_info(self):
        print("Dataset Information:")
        print(self.df.info())
        print("\nMissing Values per Column:")
        print(self.df.isnull().sum())
        print("\nDataset Description:")
        print(self.df.describe(include='all'))

    def _remove_duplicates_to_lowercase(self, verbose=False):
        before = len(self.df)
        df = self.df.drop_duplicates().copy()
        after = len(df)

        if verbose:
            print(f"Removed {before - after} duplicate rows")

        df[self.text_col] = df[self.text_col].astype(str).str.lower()
        return df
    
    def _apply_preprocessing(self, df):
        if self.preprocessor is None:
            raise ValueError("Preprocessor not provided")
        
        if self.text_col not in df.columns:
            raise ValueError(f"Column '{self.text_col}' not found in DataFrame")
            
        features_df = (
            df[self.text_col]
            .apply(self.preprocessor.transform_text)
            .apply(pd.Series)
        )
        features_df[self.text_col] = df[self.text_col].values

        if self.target_col and self.target_col in df.columns:
            features_df[self.target_col] = df[self.target_col].values

        return features_df
    
    # Main Function
    def run_dataset_operations(self, verbose=False):
        self._dataset_info()
        cleaned_df = self._remove_duplicates_to_lowercase(verbose=verbose)
        final_df = self._apply_preprocessing(cleaned_df)
        return final_df

class CustomNLPPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.custom_stop_words = self.stop_words - {'not', 'but', 'however', 'no', 'yet'}
        self.lemmatizer = WordNetLemmatizer()

    def _preprocess(self, text):
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'<.*?>', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def _count_stopwords(self,text):
        return sum(1 for word in text.split() if word in self.custom_stop_words)

    def _word_counts(self,text):
        return len(text.split())

    def _count_punctuation_chars(self, text):
        punctuation = '.,!?;:"\'()[]{}-'
        return sum(1 for char in text if char in punctuation)

    def _remove_special_characters(self, text):
        """
        Keeps only letters, numbers, spaces, and basic punctuation (! ? . ,)
        """
        text = str(text)
        return re.sub(r'[^A-Za-z0-9\s!?.,]', '', text)

    def _remove_stopwords(self, text):
        text = ' '.join([word for word in text.split() if word not in self.custom_stop_words])
        return text

    def _lemmantize_text(self, text):
        text = ' '.join([self.lemmatizer.lemmatize(word) for word in text.split()])
        return text
    
    # Main transformation function
    def transform_text(self, text):
        """
        Call this Main transformation function to get all features
        1. clean_text
        2. no_of_stopwords
        3. word_count
        4. punctuation_chars
        5. char_count
        Returns a dictionary with all features
        """
        text = str(text).lower()
        text = self._preprocess(text)
        text = self._remove_special_characters(text)

        stopword_count = self._count_stopwords(text)
        word_count = self._word_counts(text)
        punctuation_count = self._count_punctuation_chars(text)

        text = self._remove_stopwords(text)
        text = self._lemmantize_text(text)

        return {
        "clean_text": text,
        "no_of_stopwords": stopword_count,
        "word_count": word_count,
        "punctuation_chars": punctuation_count,
        "char_count": len(text)
    }

    # For single user inference
    def predict_transform(self, text, text_col="review"):
        features = self.transform_text(text)
        df = pd.DataFrame([features])
        df[text_col] = text
        return df

class CustomVisualizationHelper:
    def __init__(self,df,target,word_count,no_of_stopwords,use_clean_text=True):
        self.df = df
        self.target = target
        self.word_count = word_count
        self.use_clean_text = use_clean_text
        self.no_of_stopwords = no_of_stopwords
        self.wordcloud = WordCloud(width=800, height=400, background_color='white')
        self.vectorizer = CountVectorizer(ngram_range=(2, 2), stop_words='english')

        self.text_col = 'clean_text' if use_clean_text else 'review'
        if self.text_col not in self.df.columns:
            raise ValueError(f"Column '{self.text_col}' not found in DataFrame")

    def _basic_plots(self):

        if self.target not in self.df.columns:
            print("Target column not found. Skipping label-based plots.")
            return

        sns.countplot(x=self.target, data=self.df)
        plt.show()

        if self.df[self.target].value_counts().min() < 2:
            print("Not enough samples for KDE plot")
            return

        sns.kdeplot(self.df[self.df[self.target] == 1][self.word_count], label='Positive', fill=True)
        sns.kdeplot(self.df[self.df[self.target] == 0][self.word_count], label='Negative', fill=True)
        plt.legend()
        plt.show()

        sns.boxplot(data=self.df,x=self.target,y=self.word_count)
        plt.show()

        sns.kdeplot(self.df[self.df[self.target] == 1][self.no_of_stopwords], label='Positive', fill=True)
        sns.kdeplot(self.df[self.df[self.target] == 0][self.no_of_stopwords], label='Negative', fill=True)
        plt.legend()
        plt.show()

    def _get_top_ngrams(self, corpus, n=None):

        self.vectorizer.fit(corpus)
        bag_of_words = self.vectorizer.transform(corpus)
        sum_words = bag_of_words.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in self.vectorizer.vocabulary_.items()]

        return sorted(words_freq, key=lambda x: x[1], reverse=True)[:n]

    def _create_ngram(self):
        top_25_bigrams = self._get_top_ngrams(self.df[self.text_col], 25)
        top_25_bigrams_df = pd.DataFrame(top_25_bigrams, columns=['bigram', 'count'])

        plt.figure(figsize=(12, 8))
        sns.barplot(data=top_25_bigrams_df, x='count', y='bigram', palette='magma')
        plt.title('Top 25 Most Common Bigrams')
        plt.xlabel('Count')
        plt.ylabel('Bigram')
        plt.show()

    def _generate_wordcloud(self):
        text = ' '.join(self.df[self.text_col])
        wc = self.wordcloud.generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis("off")
        plt.show()

    def _plot_top_n_words(self,n=20):
        """Plot the top N most frequent words in the dataset."""

        words = ' '.join(self.df[self.text_col]).split()
        counter = Counter(words)
        most_common_words = counter.most_common(n)
        words, counts = zip(*most_common_words)

        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(counts), y=list(words))
        plt.title(f'Top {n} Most Frequent Words')
        plt.xlabel('Frequency')
        plt.ylabel('Words')
        plt.show()

    # Main visualization function
    def visualize(self):
        self._basic_plots()
        self._create_ngram()
        self._generate_wordcloud()
        self._plot_top_n_words()