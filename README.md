# LoanWise - Smarter Loan Risk Prediction for Confident Banking Decisions

**Unlock the power of smart lending with LoanWise!**  
LoanWise is an AI-powered loan eligibility predictor designed to help users and financial institutions make confident borrowing decisions. With advanced analytics and data-driven insights, our app provides quick and reliable predictions for loan eligibility.

---

## Key Features

- **AI-Powered Insights:** LoanWise leverages advanced machine learning models to predict loan eligibility instantly.  
- **Financial Data Analytics:** Analyze key metrics such as income and credit history to forecast loan risks.  
- **User-Friendly Interface:** Powered by **Streamlit**, the app offers an intuitive and interactive experience.  
- **Currency Conversion & Data Cleaning:** Automatically processes data and adjusts for different financial contexts.  
---

## Demo

Try it out now: [LoanWise Web App](https://loanwise.streamlit.app/)

---

## How It Works

1. **Data Processing:**  
   - Reads the input dataset and applies necessary transformations (e.g., currency conversion).  
   - Drops irrelevant columns such as `ID`, `CITY`, `STATE`, and `CURRENT_HOUSE_YRS` to focus on key predictors.

2. **Machine Learning Model:**  
   - Uses **Scikit-learn models** loaded from `.pkl` files to make predictions.  
   - Encodes categorical data using `LabelEncoder`.

---

## Setup Instructions

To run LoanWise locally:

1. Clone the repository and navigate to the project folder:
   ```bash
    git clone https://github.com/mahmoudalrefaey/loan-prediction-streamlit.git
    cd loan-prediction-streamlit
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application using Streamlit:
   ```bash
   streamlit run main.py
   ```
---

## File Structure

- **main.py:** Contains the core logic for the Streamlit app, data processing, and visualizations.
- **datasets/Training Data.csv:** Source dataset used for predictions.
- **models/model.pkl:** Pre-trained machine learning model (if applicable).
- **requirements.txt:** Lists all the Python dependencies.

---

## Technologies Used

- **Streamlit** – Front-end framework for web applications.  
- **Scikit-learn** – Machine learning library for predictive modeling.  
- **Pandas** – Data manipulation and analysis.  
- **Pickle** – For loading the machine learning model.

---

## Future Enhancements

- **Support for Multiple Loan Products:** Extend the model to predict eligibility for different types of loans.  
- **Improved UI/UX:** Add more customization options and improve interactivity.

---

## Contact

For any questions or support, please contact:  
Email: [Email me](mailto:dev.mahmoudrefaey@gmail.com)
LinkedIn: [Mahmoud Alrefaey](https://linkedin.com/in/mahmoudmalrefaey)
