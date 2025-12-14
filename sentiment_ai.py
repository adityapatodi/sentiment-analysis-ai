from sklearn.feature_extraction.text import TfidfVectorizer # Imports tool to convert text to numbers
from sklearn.naive_bayes import MultinomialNB # Imports AI model for text classification

# Instructions for user

print("\nPaste all training sentences in one line separated by a ';'.")
print("Format: Sentence, Label")
print("Example: This is great,Positive, It is so bad, Negative\n")

# System

training_data = []

raw_data = input("Paste training data here: \n")

for pair in raw_data.split(";"):
     if "," in pair:
          sentence, label = pair.split(",")
          training_data.append((sentence.strip(), label.strip()))

print(f"\nTotal training examples: {len(training_data)} examples")

texts = []
labels = []

for example in training_data:
      sentence = example[0]
      label = example[1]
      texts.append(sentence)
      labels.append(label)

vectoriser = TfidfVectorizer() # Creates an object to process each sentence
features = vectoriser.fit_transform(texts) # Converts sentences in numeric features that the AI can understand

model = MultinomialNB() # Creates AI model object
model.fit(features, labels) # Trains the model using numeric features and labels

while True:
    new_sentence = input("\nEnter a sentence to classify: ")

    new_features = vectoriser.transform([new_sentence]) # Converts the new sentence to numbers using vocabulary from before
    prediction = model.predict(new_features) # Asks AI to predict the label

    print(f"The AI predicts: {prediction[0]}.")
