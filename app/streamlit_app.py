#Add the imports and project path
from pathlib import Path
import sys

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import predict_category_with_confidence


#Configure the page

st.set_page_config(
    page_title="AI News Classifier",
    page_icon="📰",
    layout="centered",
)

# Add page heading and description
st.title("📰 AI News Classifier")

st.write(
    "Enter a news headline or short article. "
    "The model will classify it as World, Sports, Business, or Sci/Tech."
)



#Add the text input

user_text = st.text_area(     #st.text_area() returns the text entered by the user
    label="News text",
    placeholder="Example: Apple announces new AI features for iphone users",
    height=160,
)


#Prediction button
predict_button = st.button(
    "Predict Category",
    type="primary",
    use_container_width=True,
)


#Handle the button click
if predict_button:
    if user_text.strip() == "":
        st.warning("Please enter a news headline or article.")
    else:
        predicted_id, predicted_label, confidence = (
            predict_category_with_confidence(user_text)
        )

        st.subheader("Prediction Result")

        st.success(f"Predicted Category: {predicted_label}")

        st.metric(
            label="Confidence",
            value=f"{confidence:.2f}%",
    
        )

        st.caption(f"Predicted class ID: {predicted_id}")



st.divider()

st.subheader("Model Details")

st.markdown(
    """
- **Model:** Logistic Regression
- **Features:** TF-IDF
- **Dataset:** AG News
- **Categories:** World, Sports, Business, Sci/Tech
"""
)


# Add error handling

if predict_button:
    if user_text.strip() == "":
        st.warning("Please enter a news headling or articel.")
    else:
        try:
            predicted_id, predicted_label, confidence = (
                predict_category_with_confidence(user_text)
            )

            st.subheader("Prediction Result")

            st.succes(f"Predicted category: {predicted_label}")

            st.metric(
                label="Confidence",
                value=f"{confidence:2f}%",
            )

            st.caption(f"Predicted class ID: {predicted_id}")

        except FileNotFoundError:
            st.error(
                "The saved model file was not found."
                "Run the trainig script before starting the app."
            )
        
        except Exception as error:
            st.error(f"Prediction failed: {error}")