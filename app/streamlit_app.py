import streamlit as st
import pandas as pd

def main():
    st.title("Credit Scoring App")
    st.write("Enter applicant details to get a credit score.")

    # Input fields
    income = st.number_input("Annual Income", min_value=0.0, value=50000.0)
    # Add more inputs as needed

    if st.button("Calculate Score"):
        # Placeholder for scoring logic
        st.success("Score calculated!")

if __name__ == "__main__":
    main()
