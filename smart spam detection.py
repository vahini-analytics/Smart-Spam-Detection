import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


# 1. Load dataset
data = pd.read_csv("C:/Users/home/Downloads/mail_data.csv")


# 2. Separate messages and labels
messages = data["Message"]
labels = data["Category"]


# 3. Split data
mess_train, mess_test, lab_train, lab_test = train_test_split(
    messages,
    labels,
    test_size=0.20,
    random_state=42
)


# 4. Convert text into numbers
cv = CountVectorizer(stop_words="english")

features = cv.fit_transform(mess_train)


# 5. Create model
model = MultinomialNB()


# 6. Train model
model.fit(features, lab_train)


# 7. Test model
features_test = cv.transform(mess_test)

predictions = model.predict(features_test)

accuracy = accuracy_score(lab_test, predictions)


# 8. Prediction function
def predict(text):

    input_text = cv.transform([text])

    result = model.predict(input_text)

    return result[0]


# 9. Streamlit interface
st.title("📧 Email Spam Detection")

st.write(
    "Enter an email or message to check "
    "whether it is Spam or Not Spam."
)


# 10. Display accuracy
st.info(f"Model Accuracy: {accuracy * 100:.2f}%")


# 11. User input
user_message = st.text_area(
    "Enter your email/message:",
    height=150
)


# 12. Check message
if st.button("Check Message"):

    if user_message.strip() == "":
        st.warning("Please enter a message.")

    else:

        result = predict(user_message)

        if result == "spam":
            st.error("🚨 SPAM MESSAGE")

        else:
            st.success("✅ NOT SPAM")
