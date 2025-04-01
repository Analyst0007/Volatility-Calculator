# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 15:30:38 2025

@author: Hemal
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Stock Volatility Calculator",
    page_icon="📈",
    layout="wide"
)

def calculate_volatility(ticker_symbol, period="2y"):
    """
    Calculate daily, weekly, and monthly volatility for a given stock
    
    Parameters:
    ticker_symbol (str): Stock ticker symbol (e.g., 'AAPL', 'MSFT')
    period (str): Period to download data for (default: '2y' = 2 years)
    
    Returns:
    tuple: Daily, weekly, and monthly volatility and the stock_data
    """
    # Download historical data
    with st.spinner(f"Downloading historical data for {ticker_symbol}..."):
        stock_data = yf.download(ticker_symbol, period=period)
    
    if stock_data.empty:
        st.error(f"No data found for {ticker_symbol}")
        return None, None, None, None
    
    # Calculate daily returns using Close price
    stock_data['Daily_Return'] = stock_data['Close'].pct_change()
    
    # Remove NaN values
    returns = stock_data['Daily_Return'].dropna()
    
    if len(returns) == 0:
        st.error(f"Insufficient data for {ticker_symbol}")
        return None, None, None, None
    
    # Calculate daily volatility (standard deviation of daily returns)
    daily_volatility = float(returns.std())
    
    # Calculate weekly volatility
    # Method 1: Using the formula daily_volatility * sqrt(5)
    weekly_volatility_estimate = daily_volatility * np.sqrt(5)
    
    # Method 2: Calculating actual weekly returns
    weekly_returns = stock_data['Close'].resample('W').ffill().pct_change().dropna()
    weekly_volatility_actual = float(weekly_returns.std())
    
    # Calculate monthly volatility
    # Method 1: Using the formula daily_volatility * sqrt(21)
    monthly_volatility_estimate = daily_volatility * np.sqrt(21)
    
    # Method 2: Calculating actual monthly returns
    monthly_returns = stock_data['Close'].resample('M').ffill().pct_change().dropna()
    monthly_volatility_actual = float(monthly_returns.std())
    
    # Annualized volatilities
    annualized_daily = daily_volatility * np.sqrt(252)
    annualized_weekly = weekly_volatility_actual * np.sqrt(52)
    annualized_monthly = monthly_volatility_actual * np.sqrt(12)
    
    results = {
        "Ticker": ticker_symbol,
        "Period": period,
        "Data Points": len(stock_data),
        "Daily Volatility": {
            "Value": daily_volatility,
            "Percentage": daily_volatility*100,
            "Annualized Value": annualized_daily,
            "Annualized Percentage": annualized_daily*100
        },
        "Weekly Volatility": {
            "Estimated Value": weekly_volatility_estimate,
            "Estimated Percentage": weekly_volatility_estimate*100,
            "Actual Value": weekly_volatility_actual,
            "Actual Percentage": weekly_volatility_actual*100,
            "Annualized Value": annualized_weekly,
            "Annualized Percentage": annualized_weekly*100
        },
        "Monthly Volatility": {
            "Estimated Value": monthly_volatility_estimate,
            "Estimated Percentage": monthly_volatility_estimate*100,
            "Actual Value": monthly_volatility_actual,
            "Actual Percentage": monthly_volatility_actual*100,
            "Annualized Value": annualized_monthly,
            "Annualized Percentage": annualized_monthly*100
        }
    }
    
    # Calculate rolling returns and volatility for the plots
    stock_data['20d_Vol'] = stock_data['Daily_Return'].rolling(window=20).std() * np.sqrt(20)  # 20 trading days
    stock_data['60d_Vol'] = stock_data['Daily_Return'].rolling(window=60).std() * np.sqrt(60)  # 60 trading days
    
    return daily_volatility, weekly_volatility_actual, monthly_volatility_actual, stock_data, results

def plot_volatility_plotly(stock_data, ticker_symbol):
    """Create interactive Plotly chart for stock price and volatility"""
    
    # Create figure with secondary y-axis
    fig = make_subplots(rows=2, cols=1, 
                         shared_xaxes=True, 
                         vertical_spacing=0.1, 
                         row_heights=[0.7, 0.3],
                         subplot_titles=(f'{ticker_symbol} Close Price', 'Rolling Volatility (%)'))
    
    # Add stock price trace to top subplot
    fig.add_trace(
        go.Scatter(x=stock_data.index, y=stock_data['Close'], name='Close Price', line=dict(color='blue')),
        row=1, col=1
    )
    
    # Add 20-day volatility to bottom subplot
    fig.add_trace(
        go.Scatter(x=stock_data.index, y=stock_data['20d_Vol']*100, name='20-Day Volatility', line=dict(color='orange')),
        row=2, col=1
    )
    
    # Add 60-day volatility to bottom subplot
    fig.add_trace(
        go.Scatter(x=stock_data.index, y=stock_data['60d_Vol']*100, name='60-Day Volatility', line=dict(color='red')),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=700,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis2_title="Date",
        yaxis_title="Price",
        yaxis2_title="Volatility (%)",
        hovermode="x unified"
    )
    
    # Update y-axes
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volatility (%)", row=2, col=1)
    
    return fig

def main():
    st.title("📈 Stock Volatility Calculator")
    st.write("Calculate and visualize daily, weekly, and monthly volatility for any stock")
    
    # Create sidebar for inputs
    st.sidebar.header("Inputs")
    
    # Get ticker symbol
    ticker = st.sidebar.text_input("Enter stock ticker symbol (e.g., AAPL, MSFT, GOOGL)", "AAPL").strip().upper()
    
    # Select time period
    period_options = {
        "1mo": "1 Month",
        "3mo": "3 Months",
        "6mo": "6 Months",
        "1y": "1 Year",
        "2y": "2 Years",
        "5y": "5 Years",
        "10y": "10 Years",
        "max": "Maximum Available"
    }
    
    period = st.sidebar.selectbox(
        "Select time period for historical data:",
        list(period_options.keys()),
        format_func=lambda x: period_options[x],
        index=4  # Default to 2y
    )
    
    # Add calculate button
    if st.sidebar.button("Calculate Volatility"):
        daily_vol, weekly_vol, monthly_vol, stock_data, results = calculate_volatility(ticker, period)
        
        if daily_vol is not None:
            # Display company info
            try:
                ticker_info = yf.Ticker(ticker).info
                company_name = ticker_info.get('longName', ticker)
                st.header(f"{company_name} ({ticker}) Volatility Analysis")
                
                # Display stock info in columns
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Current Price", f"INR{ticker_info.get('currentPrice', 'N/A'):,.2f}")
                with col2:
                    st.metric("Market Cap", f"{ticker_info.get('marketCap', 0)/1000000000:,.2f}B")
                with col3:
                    st.metric("52 Week High", f"INR{ticker_info.get('fiftyTwoWeekHigh', 'N/A'):,.2f}")
            except:
                st.header(f"{ticker} Volatility Analysis")
            
            # Display results in expandable sections
            with st.expander("Volatility Results", expanded=True):
                # Create three columns for different volatility results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.subheader("Daily Volatility")
                    st.metric("Daily", f"{results['Daily Volatility']['Percentage']:.2f}%")
                    st.metric("Annualized", f"{results['Daily Volatility']['Annualized Percentage']:.2f}%")
                
                with col2:
                    st.subheader("Weekly Volatility")
                    st.metric("Weekly", f"{results['Weekly Volatility']['Actual Percentage']:.2f}%")
                    st.metric("Annualized", f"{results['Weekly Volatility']['Annualized Percentage']:.2f}%")
                
                with col3:
                    st.subheader("Monthly Volatility")
                    st.metric("Monthly", f"{results['Monthly Volatility']['Actual Percentage']:.2f}%")
                    st.metric("Annualized", f"{results['Monthly Volatility']['Annualized Percentage']:.2f}%")
                
                st.write("---")
                st.write("**Explanation:**")
                st.write("""
                - **Daily Volatility:** Standard deviation of daily returns
                - **Weekly Volatility:** Standard deviation of weekly returns
                - **Monthly Volatility:** Standard deviation of monthly returns
                - **Annualized:** Volatility converted to annual basis (daily × √252, weekly × √52, monthly × √12)
                """)
            
            # Plot stock price and volatility
            st.subheader("Stock Price and Volatility Chart")
            fig = plot_volatility_plotly(stock_data, ticker)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show raw data if requested
            with st.expander("Show Raw Data"):
                st.dataframe(stock_data)
                
                # Add download button for CSV
                csv = stock_data.to_csv().encode('utf-8')
                st.download_button(
                    label="Download Data as CSV",
                    data=csv,
                    file_name=f'{ticker}_volatility_data.csv',
                    mime='text/csv',
                )
    
    # Show instructions at first
    else:
        st.info("👈 Enter a stock ticker and select a time period in the sidebar, then click 'Calculate Volatility'")
        st.write("""
        ## About Stock Volatility
        
        Volatility is a statistical measure of the dispersion of returns for a given security or market index. 
        In most cases, the higher the volatility, the riskier the security.
        
        This app calculates three types of volatility:
        
        1. **Daily Volatility:** The standard deviation of daily returns
        2. **Weekly Volatility:** The standard deviation of weekly returns
        3. **Monthly Volatility:** The standard deviation of monthly returns
        
        All volatilities are also converted to an annualized basis for easier comparison.
        """)

if __name__ == "__main__":
    main()