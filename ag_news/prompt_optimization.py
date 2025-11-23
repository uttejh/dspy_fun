import dspy
import datasets
from typing import Literal


def make_signature(text_labels):
    class ArticleClassification(dspy.Signature):
        text: str = dspy.InputField()
        label: Literal[*text_labels] = dspy.OutputField()
        confidence: float = dspy.OutputField()

    return ArticleClassification


def calculate_accuracy(predictions, references):
    correct = sum(p == r for p, r in zip(predictions, references))
    return correct / len(references)


def predict_sample(sample):
    result = classify(text=sample["text"])
    sample["predicted_label"] = result.label
    sample["predicted_confidence"] = result.confidence
    return sample


def load_data():
    # load AG News dataset
    ag_news = datasets.load_dataset("fancyzhx/ag_news")

    # Define the label mapping once
    text_labels = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

    ag_news_mapped = ag_news.map(lambda x: {"text_label": text_labels[x["label"]]}, batched=False)
    train_set = ag_news_mapped["train"]
    test_set = ag_news_mapped["test"]

    train_samples = train_set.shuffle(seed=42).select(range(100))
    test_samples = test_set.shuffle(seed=42).select(range(100))

    return train_samples, test_samples


def validate_answer(ground_truth, predicted, trace=None):
    return ground_truth.label == predicted.label


def create_examples(dataset):
    examples = []
    for sample in dataset:
        example = dspy.Example(
            text=sample["text"],
            label=sample["text_label"]
        ).with_inputs("text")
        examples.append(example)

    return examples

if __name__ == "__main__":
    train_set, test_set = load_data()

    ArticleClassification = make_signature(list(set(train_set["text_label"])))
    classify = dspy.Predict(ArticleClassification)
    LM = dspy.LM(model="openai/Qwen/Qwen3-0.6B", api_base="http://0.0.0.0:8000/v1", api_key="dummy")
    dspy.configure(lm=LM)

    training_examples = create_examples(train_set)
    testing_examples = create_examples(test_set)

    evaluator = dspy.Evaluate(devset=testing_examples, num_threads=1, display_progress=True, display_table=5)
    evaluator(classify, metric=validate_answer)

    tp = dspy.MIPROv2(metric=validate_answer, auto="light")
    optimized_classify = tp.compile(classify, trainset=training_examples, max_labeled_demos=0, max_bootstrapped_demos=0)

    evaluator(optimized_classify, metric=validate_answer)
    print(dspy.inspect_history())
