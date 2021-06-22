from typing import List
import typing
import math

file = open("./stopwords.csv")
stop_words = [line.strip() for line in file]

file = open("./spam.csv")
spam_emails = []
ham_emails = []
for line in file:
    if line.startswith("spam"):
        spam_emails.append(line[5:-4])
    else:
        ham_emails.append(line[4:-4])


def extract_words(text: str):
    return list(set(text.split(" ")))


def flatten_list(lists: List[List[any]], unique: bool = False):
    flat = []
    for ls in lists:
        flat += ls
    return list(set(flat)) if unique else flat


def extract_words_dict(text: str):
    return dict(text.split(" "))


def calculate_words_prob(emails: List[str], words: List[str]):
    freq = {}
    for word in words:
        emails_with_word = 0
        for email in emails:
            if word in email:
                emails_with_word += 1
        freq[word] = (emails_with_word + 1) / (len(emails) + 2)
    return freq


def get_known_words(dictionary: typing.Set[str], words):
    return list(filter(lambda word: word in dictionary, words))


def remove_stop_words(list_: List[str]):
    return list(filter(lambda word: word not in stop_words, list_))


FrequencyCounter = typing.Dict[str, float]


class Model:
    def __init__(self, spam_emails: List[str], ham_emails: List[str], spam_words: List[str], ham_words: List[str], spamicity: FrequencyCounter, hamicity: FrequencyCounter, p_spam: float, p_ham: float) -> None:
        self.spam_emails = spam_emails
        self.ham_emails = ham_emails
        self.spam_words = spam_words
        self.ham_words = ham_words
        self.spamicity = spamicity
        self.hamicity = hamicity
        self.spam_prob = p_spam
        self.ham_prob = p_ham


def generate_classifier_model(spam_emails: List[str], ham_emails: List[str]):
    spam_words = flatten_list([extract_words(spam_email)

                              for spam_email in spam_emails], unique=True)
    ham_words = flatten_list([extract_words(ham_email)

                             for ham_email in ham_emails], unique=True)

    spamicity = calculate_words_prob(spam_emails, spam_words)
    hamicity = calculate_words_prob(ham_emails, ham_words)

    p_spam = len(spam_emails) / (len(spam_emails) + len(ham_emails))
    p_ham = len(ham_emails) / (len(spam_emails) + len(ham_emails))

    return Model(spam_emails=spam_emails, ham_emails=ham_emails, spam_words=spam_words, ham_words=ham_words, spamicity=spamicity, hamicity=hamicity, p_spam=p_spam, p_ham=p_ham)


def classify_email(email: str, model: Model):
    dictionary = set(model.ham_words + model.spam_words)
    words = extract_words(email)
    known_words = get_known_words(dictionary, words)
    reduced_words = remove_stop_words(known_words)

    probs = []
    for word in reduced_words:
        if word in model.spamicity:
            word_spamicity = model.spamicity[word]
        else:
            word_spamicity = 1 / (len(model.spam_emails) + 2)
        if word in model.hamicity:
            word_hamicity = model.hamicity[word]
        else:
            word_hamicity = 1 / (len(model.ham_emails) + 2)
        prob_word_is_spam = (word_spamicity * model.spam_prob) / (
            (word_spamicity * model.spam_prob) + (word_hamicity * model.ham_prob))
        probs.append(prob_word_is_spam)

    final_prob = 1
    for prob in probs:
        final_prob *= prob

    is_spam = final_prob >= 0.5
    return (email, is_spam, final_prob)

if __name__ == "__main__":
    # training phase: select half of all training set
    scale = 0.7
    training_spam_emails = spam_emails[:math.floor(len(spam_emails) * scale)]
    training_ham_emails = ham_emails[:math.floor(len(ham_emails) * scale)]
    model = generate_classifier_model(
        training_spam_emails, training_ham_emails)

    num_correct = 0
    test_spam_emails = spam_emails[len(training_spam_emails):]
    test_ham_emails = ham_emails[len(training_ham_emails):]
    test_emails = flatten_list([
        [(email, True) for email in test_spam_emails],
        [(email, False) for email in test_ham_emails]
    ])

    for email, label in test_emails:
        email, is_spam, confidence = classify_email(email, model)
        if is_spam == label:
            num_correct += 1
        # print(
        #     f"Email: {email}\nSpam: {is_spam}\nConfidence: {confidence}%\nCorrect: {label == is_spam}\n")

    accuracy = num_correct / len(test_emails) * 100
    print(f"The model completed with an accuracy of {accuracy}%")
