import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from text_preprocessing import TextPreprocessor
from text_augment import synonym_replacement
import warnings
warnings.filterwarnings("ignore")


# ✅ Augmentation хийх функц
def augment_spam(df, augment_count=500, n_replace=2):
    spam_df = df[df["Category"] == "spam"]
    augmented_texts = []
    for _ in range(augment_count):
        sample = spam_df.sample(1)
        text = sample["Message"].values[0]
        label = sample["Category"].values[0]
        augmented_text = synonym_replacement(text, n=n_replace)
        augmented_texts.append({"Category": label, "Message": augmented_text})
    aug_df = pd.DataFrame(augmented_texts)
    return pd.concat([df, aug_df]).reset_index(drop=True)


# ✅ Функц: өгөгдлөөр загвар сургах ба үр дүн буцаах
def train_and_evaluate(X, y, model_type="NB"):
    preprocessor = TextPreprocessor()
    X_vec = preprocessor.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=0.2, random_state=42)

    if model_type == "NB":
        model = MultinomialNB()
    else:
        model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    return acc, report


# ✅ Өгөгдөл ачаалж бэлдэх
df = pd.read_csv("spam.csv")
df = df.rename(columns={"v1": "Category", "v2": "Message"}) if "v1" in df.columns else df

# ✅ Original өгөгдлөөр
X_orig = df["Message"]
y_orig = df["Category"]

# ✅ Augmented өгөгдөл
df_aug = augment_spam(df, augment_count=500)
X_aug = df_aug["Message"]
y_aug = df_aug["Category"]

# ✅ Загваруудыг сургаж харьцуулах
# ✅ Загваруудыг сургаж харьцуулах
models = ["NB", "LR"]
for model in models:
    print(f"\n----- {model} Model (Original Data) -----")
    acc_o, rep_o = train_and_evaluate(X_orig, y_orig, model_type=model)
    print("Accuracy:", acc_o)
    for label in rep_o:
        if label.lower() == 'spam':
            print("Spam class F1-score:", rep_o[label]["f1-score"])

    print(f"\n----- {model} Model (Augmented Data) -----")
    acc_a, rep_a = train_and_evaluate(X_aug, y_aug, model_type=model)
    print("Accuracy:", acc_a)
    for label in rep_a:
        if label.lower() == 'spam':
            print("Spam class F1-score:", rep_a[label]["f1-score"])

