import yfinance as yf
import pandas as pd

# 삼성전자 주식 데이터 다운로드 (005930.KS는 삼성전자의 코스피 코드)
ticker = '005930.KS'

# 2023년 주식 데이터 다운로드
data_2023 = yf.download(ticker, start='2023-01-01', end='2023-12-31')
data_2023.to_csv('Samsung_Stock_Price_Train.csv')

# 2024년 주식 데이터 다운로드
data_2024 = yf.download(ticker, start='2024-01-01', end='2024-12-31')
data_2024.to_csv('Samsung_Stock_Price_Test.csv')

print("데이터 다운로드 완료")