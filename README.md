# Frankline & Co. LP Trading Terminal Beta
**Trading Terminal is a Comprehensive Trading App that allows Analysts to visualize, Research, Predict and carry out algorithmic trading on either stocks/crypto. Beware, this is not an advisory Investment app but just a base tool for investment decisions and you carry further liability when utilizing its CCXT and interactive brokerage aspect**

## 🏗️ **How It's Built - Main Features and Framework**

- **Streamlit** - To create the web app UI and interactivity 
- **YFinance** - To fetch financial data from Yahoo Finance API
- **Langchain & Langsmith** - To build the Multimodial agentic bot
- **Plotly.express** - To create interactive financial charts
- **Alpaca/CCXT** - Main Trading agents
- **AWS EC2** - Run the app in cloud
- **Docker** -Main container used for ec2 updates
- **Open AI** - Powers the Langchain agentic bot

## 🎯 **Key Features**

- **Real-time data** - Fetch latest prices and fundamentals 
- **Financial charts** - Interactive historical and forecast charts
- **ARIMA forecasting** - Make statistically robust predictions
- **Backtesting** - Evaluate model performance
- **Responsive design** - Works on all devices

## 🎯 **Project Updates **
* Beta is up and running, Thanks to the Cloud Provision by AWS Grant for businesses that allowed me to utilize powerful EC2 instance to run the heavy Backtesting models and test out this proof of concept enabling a transition from streamlit cloud
* 80% of the code base is working : Still preparing stable pipelines for Finviz screeners, and snippets that require external API calls that have subscriptions
* I am working on the AI Multi-modial Chatbot to utilize relevant code, generate data and report in real-time how the market is doing and the most potential profit trades to do.
* On side, I am continously improving the UI/UX by livening the charts and appending sentiment analysis, alongside the bot ux 
* I decided to Re-structure the code into individual folders for readability and future updates, everything is still in the app.py and will break the repo down into individual components and start updating and continously testing

 ## **Issues** 
 * Since we are running on AWS EC2, I am implementing a secure SSL pipeline to allow website access without warnings of security for some users
 * remote running best runner R & D : tmux or screen
 * TensorRT error for running the ML segments : compute optimization

## 🚀 **Getting Started**

### **Local Installation**

1. Clone the repo

```bash
git clone https://github.com/FranklineMisango/Trading-Terminal.git
```

2. Install requirements

```bash
pip install -r requirements.txt
```

3. Change directory
```bash
cd streamlit_app
```

4. Run the app

```bash
streamlit run app.py
```

The app will be live at ```http://localhost:8501```

## 📈 **Future Roadmap**

- **AI powered by Multimodial agents to do the analysis**
- **More advanced forecasting models**
- **More Quantitative trading strategies**
- **Robust Portfolio optimization and tracking**
- **Additional fundamental data**
- **User account system**


## **⚖️ Disclaimer**
**This is not financial advice! Use forecast data to inform your own investment research. No guarantee of trading performance.**
"""