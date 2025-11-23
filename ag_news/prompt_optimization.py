import dspy
import datasets
from typing import Literal

# load AG News dataset
ag_news = datasets.load_dataset("fancyzhx/ag_news")

# Define the label mapping once
text_labels = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

# Define a function to apply the transformation
def map_label_to_text(sample):
    sample["text_label"] = text_labels[sample["label"]]
    return sample

ag_news_mapped = ag_news.map(map_label_to_text)

train_set = ag_news_mapped["train"]
test_set = ag_news_mapped["test"]

train_samples = train_set.shuffle(seed=42).select(range(100))
test_samples = test_set.shuffle(seed=42).select(range(100))

# write signatures for prompt optimization
class ArticleClassification(dspy.Signature):
    """Classify news articles into categories."""
    text: str = dspy.InputField(desc="News article text.")
    label: Literal[*text_labels.values()] = dspy.OutputField(desc="Category of the news article.")
    confidence: float = dspy.OutputField()

classify = dspy.Predict(ArticleClassification)
LM = dspy.LM(model="openai/Qwen/Qwen3-0.6B", api_base="http://0.0.0.0:8000/v1", api_key="dummy")
dspy.configure(lm=LM)

def calculate_accuracy(predictions, references):
    correct = sum(p == r for p, r in zip(predictions, references))
    return correct / len(references)

def predict_sample(sample):
    result = classify(text=sample["text"])
    sample["predicted_label"] = result.label
    sample["predicted_confidence"] = result.confidence
    return sample

results = test_samples.map(predict_sample, num_proc=1)

accuracy = calculate_accuracy(results["predicted_label"], results["text_label"])
print(f"Accuracy: {accuracy * 100:.2f}%")
# 71.00% accuracy on 100 test samples without any optimized prompt