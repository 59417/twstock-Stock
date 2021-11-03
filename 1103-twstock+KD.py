""" Twstock - KD 回測 """

# twstock doc
# https://twstock.readthedocs.io/zh_TW/latest/quickstart.html

#【twstock】最輕鬆的方式取得準確的台灣證交所資料!又是給你一篇手把手教學!
# https://pixnashpython.pixnet.net/blog/post/43244764


# In[ 2021/07 - 2021/10 股票資料 ]

#pip install twstock
#pip install mplfinance

import twstock
import pandas as pd
import matplotlib as plt
import mplfinance as mpf

twstock.__update_codes()  # 更新

target_stock = input('輸入股票代號: ')  # 以 1301台塑 示範

# 擷取 YY/MM 至今股票資訊
stock1301 = twstock.Stock(target_stock)
target_price = stock1301.fetch_from(2021, 7)  # 從2021/07開始擷取
cols = ['date', 'capacity', 'turnover', 
        'open', 'high', 'low', 'close', 
        'change', 'trascation']
df = pd.DataFrame(columns = cols, data = target_price)
#df.to_csv('stock.csv', index = False)

# mplfinance 模組中對於交易量的辨認是Volume這個字
# 所以我們使用pandas的df.rename()功能調整表頭
df.rename(columns = {'turnover': 'volume'}, inplace = True) 
df

df2 = df.iloc[:, 1:]
df2.index = df['date']
df2

# In[ 均線繪圖 ]

# 均線種類	說明	                分析意義
# 5日均線  	周線，極短線操作指標	飆股跌破周線可能是出場時機
# 10日均線	雙周線，短線操作指標	強勢股跌破雙周線可能會進入短期整理
# 20日均線	月線，多頭操作指標	    跌破月線可能會進入短期空頭修正格局
# 60日均線	季線，中期操作指標	    跌破季線可能會進入中期空頭修正格局
# 240日均線	年線，長期操作指標	    跌破季線可能會進入長期空頭修正格局

plt.rcParams['font.family'] = ['Arial Unicode MS']  # 正常顯示中文
# 設定顏色: (漲, 跌) = (紅, 綠)
mc = mpf.make_marketcolors(up = 'r', down = 'g', inherit = True)
# 設定樣式風格 
s  = mpf.make_mpf_style(base_mpf_style = 'yahoo', marketcolors = mc, 
                        rc = {'font.family': 'Arial Unicode MS'})  # 解決中文亂碼
kwargs = dict(type = 'candle', mav = (5, 20, 60),  # 移動平均線
              volume = True,  # 顯示成交量
              figratio = (10, 8), figscale = 0.75, title = '1301 台塑', style = s)
mpf.plot(df2, **kwargs)
 
# In[ matplolib ]

# 查看matplolib內建樣式
plt.style.available


s2 = mpf.make_mpf_style(base_mpl_style = 'seaborn-talk', marketcolors = mc, 
                        rc = {'font.family': 'Arial Unicode MS'})  # 解決中文亂碼
kwargs2 = dict(type = 'candle', mav = (5), volume = True, figratio = (10, 8), 
               figscale = 0.75, title = '1301 台塑', style = s2)
mpf.plot(df2, **kwargs2)


# In[ Ta-lib ]
    
import talib

# 計算 KD指標
df2["K"],df2["D"] = talib.STOCH(df2['high'], df2['low'], df2['close'], 
                                fastk_period=9, slowk_period=3, slowk_matype=1, 
                                slowd_period=3, slowd_matype=1)
# KD指標最常用的週期為 (9,3,3) / Ta-lib 預設為 (5,3,3)
# > fastk_period 設定為 9 (回溯到前面9天的資料 以算出第10天的KD值)
# > slowk_matype/slowd_matype (平滑的種類) 設為 1 
df2



# In[ KD指標 判斷買賣時機 ]

import numpy as np

# 黃金/死亡交叉
cross = [0]
for i in range(1, len(df2)):
    sK = df2.K[i]
    s0K = df2.K[i-1]
    sD = df2.D[i]
    s0D = df2.D[i-1]
    
    if (s0K < s0D) and (sK > sD):
        cross.append(1)  # 買
    elif (s0K > s0D) and (sK < sD):
        cross.append(-1)  # 賣
    else:
        cross.append(0)
df2["signal"] = cross


# 標記買入/賣出點 (做圖用)
buymark = []
sellmark = []
for i in range(len(df2)):
    buy = df2["high"][i] + 3
    sell = df2["low"][i] - 3
    if (df2["signal"][i] == 1):
        buymark.append(buy) 
        sellmark.append(np.nan)
    elif (df2["signal"][i] == -1):
        buymark.append(np.nan)
        sellmark.append(sell)
    else:
        buymark.append(np.nan)
        sellmark.append(np.nan)
df2['buy'] = buymark
df2['sell'] = sellmark


# 做圖: 時間從 2021/08 - now
plt.rcParams['font.family'] = ['Arial Unicode MS']  # 正常顯示中文
mc = mpf.make_marketcolors(up = 'r', down = 'g', inherit = True)  # (漲, 跌) = (紅, 綠)
s  = mpf.make_mpf_style(base_mpf_style = 'yahoo', marketcolors = mc, 
                        rc = {'font.family': 'Arial Unicode MS'})  # 解決中文亂碼
add_plot = [mpf.make_addplot(df2["buy"][22:], scatter = True, markersize = 50, 
                             marker = 'v', color = 'r'),
            mpf.make_addplot(df2["sell"][22:], scatter = True, markersize = 50, 
                             marker = '^', color = 'g'),
            mpf.make_addplot(df2["K"][22:], panel = 2, color = "b"),
            mpf.make_addplot(df2["D"][22:], panel = 2, color = "r")]
kwargs = dict(type = 'candle', volume = True, figsize = (20, 10),
              title = '1301台塑', style = s, addplot = add_plot)
mpf.plot(df2.iloc[22:, :], **kwargs)

df2.to_csv('/Users/a17/Documents/AI/stock.csv')


# In[ KD預測回測 ]

# 統計敘述
buy_t = df2['close'][np.isnan(df2["buy"]) == False]  # 7
sell_t = df2['close'][np.isnan(df2["sell"]) == False]  # 7
print(f'買進次數: {len(buy_t)}次\n賣出次數: {len(sell_t)}次')
'''   買進次數: 7次 賣出次數: 7次   '''

# 計算每次投資報酬率
return_rate = []
for i in range(len(sell_t)):
    rate = round((sell_t[i] - buy_t[i]) / buy_t[i] * 100, 2)
    return_rate.append(rate)
return_rate
print("該策略最高報酬: " + str(sorted(return_rate)[-1]) + " %")  # 10.73 %
print("該策略最高虧損: " + str(abs(sorted(return_rate)[0])) + " %")  # 2.7 %

# 計算勝率
win = len([i for i in return_rate if i > 0])
lose = len([i for i in return_rate if i <= 0])
print("總獲利次數 : " + str(win) + "次")  # 3
print("總虧損次數 : " + str(lose) + "次")  # 4
print("總交易次數 : " + str(win + lose) + "次")  # 7
print("勝率為: " + str(round(win / len(return_rate) * 100, 2)) + "%")  # 42.86%

# 計算累積報酬率
cum_return = [0]
for i in range(len(return_rate)):
    cum = round(return_rate[i] + cum_return[i], 2)
    cum_return.append(cum)
print("該策略總報酬為: " + str(cum_return[-1]) + "%")  # 6.95%
print("該策略平均每次報酬為: " + str(round(cum_return[-1]/(win + lose),2)) + "%")  # 0.99%
#cum_return

plt.pyplot.plot(cum_return, "ro-")
plt.show()


# =============================================================================
# print(f'買進次數: {len(buy_t)}次\n賣出次數: {len(sell_t)}次')
# print("該策略最高報酬: " + str(sorted(return_rate)[-1]) + " %") 
# print("該策略最高虧損: " + str(abs(sorted(return_rate)[0])) + " %")  
# print("總獲利次數 : " + str(win) + "次") 
# print("總虧損次數 : " + str(lose) + "次")  
# print("總交易次數 : " + str(win + lose) + "次")  
# print("勝率為: " + str(round(win / len(return_rate) * 100, 2)) + "%")  
# print("該策略總報酬為: " + str(cum_return[-1]) + "%") 
# print("該策略平均每次報酬為: " + str(round(cum_return[-1]/(win + lose),2)) + "%")  
# =============================================================================





# In[ twstock 基本指令 ]

''' 千萬要注意
.fetch_from()這個方法等於去多次拜訪證交所，最後把資料拼接回傳給你，所以時間拉太長拜訪頻率高，可能執行一次還沒拿到資料就被拒絕連線了!其他方法的呼叫如果次數很多也請記得加上time.sleep()。 '''


# pip install twstock
import twstock
import pandas as pd

# 更新
twstock.__update_codes()

# 查看交易所商品清單
tickers = twstock.twse
tk_key = tuple(tickers.keys())
tickers[tk_key[0]]  # '1101'
'''
StockCodeInfo(type='股票', code='1101', name='台泥', ISIN='TW0001101004', 
              start='1962/02/09', market='上市', group='水泥工業', CFI='ESVUFR')
'''
df_tickers = pd.DataFrame(tickers).T

# 判斷股票是否在清單裡面
'0050' in tickers  # True
# 查詢代號是否為上市股票
'2330' in twstock.twse
True
# 查詢代號是否為上櫃股票
'2330' in twstock.tpex
False

# 取得證交所股票data
stock = twstock.Stock('2330')  # .Stock 取得歷史股票資訊
st_symbol = stock.sid  # 回傳股票代號
st_date = stock.date  # 時間 (近31日開盤資料) > 愈前面越舊/愈後面愈新
st_cap = stock.capacity  # 總成交股數 (股)
st_trn = stock.turnover  # 總成交金額 (新台幣/元)
st_open = stock.open  # 開
st_high = stock.high  # 高
st_low = stock.low  # 低
st_close = stock.price  # 收
st_chg = stock.change  # 漲跌價差
st_tran = stock.transaction  # 成交筆數

# 轉成DataFrame
st_data = {'Symbol': st_symbol,
           'Capacity': st_cap,
           'Turnover': st_trn,
           'Open': st_open, 
           'High': st_high, 
           'Low': st_low, 
           'Close': st_close,
           'Change': st_chg,
           'Transaction': st_tran}
st_df = pd.DataFrame(st_data, index=st_date)
'''
           Symbol   Open   High    Low  Close
2021-09-07   2330  634.0  634.0  623.0  623.0
2021-09-08   2330  622.0  627.0  612.0  619.0
2021-09-09   2330  612.0  620.0  610.0  619.0
'''

# 取得其他期間歷史資料
jul15 = stock.fetch(2015, 7)  # 獲取 2015/07 之股票資料
'''
Data(date=datetime.datetime(2015, 7, 1, 0, 0), 
     capacity=33330373, turnover=4704327416, 
     open=140.0, high=142.5, low=139.0, close=141.0, 
     change=0.5, transaction=9303)
'''
#stock.fetch_31()      # 獲取近 31 日開盤之股票資料
#from_jan21 = stock.fetch_from(2021, 1)  # 獲取 2021/01 至今日之資料(可能會被連線拒絕)

# 取得即時資料
stock = twstock.realtime.get('2330')
'''
{'timestamp': 1634884200.0,
 'info': {'code': '2330',
  'channel': '2330.tw',
  'name': '台積電',
  'fullname': '台灣積體電路製造股份有限公司',
  'time': '2021-10-22 14:30:00'},
 'realtime': {'latest_trade_price': '600.0000',
  'trade_volume': '2086',
  'accumulate_trade_volume': '13975',
  'best_bid_price': ['599.0000',
   '598.0000',
   '597.0000',
   '596.0000',
   '595.0000'],
  'best_bid_volume': ['177', '343', '77', '293', '440'],
  'best_ask_price': ['600.0000',
   '601.0000',
   '602.0000',
   '603.0000',
   '604.0000'],
  'best_ask_volume': ['36', '580', '1381', '1210', '1171'],
  'open': '600.0000',
  'high': '602.0000',
  'low': '594.0000'},
 'success': True}
'''
print(stock['success']) # 確認是否回報有誤
# True

# 取得多檔即時資料
stocks = twstock.realtime.get(['2330', '2317', '3008'])
df_realtime = pd.DataFrame(stocks)


# In[ 基本股票資訊分析 ]

# 目前股票持續上升之天數
stock.continuous(stock.price)

# 計算n日平均價格
ave5 = stock.moving_average(stock.price, 5)  
ave10 = stock.moving_average(stock.price, 10) 
# 計算n日平均交易量
ave5_cap = stock.moving_average(stock.capacity, 5)  
ave10_cap = stock.moving_average(stock.capacity, 10)

'''
# 乖離率(BIAS)是衡量「目前股價與平均線」差距的指標，也就是目前股價偏離平均線的百分比。

            目前價 - 移動平均價
  乖離率 =  -------------------
                移動平均價

# 計算乖離率需要搭配股價及股價的均線來看，例如20日乖離率，要用股價和20日均線比較
# 乖離率可分為正乖離率與負乖離率，若股價在均線之上，稱為正乖離；股價在均線之下，稱為負乖離。

  1.當股價觸碰到「正乖離線」，不要 追高買進， 未來幾天可能會有一波 股價下跌的修正。
    乖離率過大時，表示股票在高點，有很大的機率會下跌

  2.當股價觸碰到「負乖離線」，不要 殺低賣出， 未來幾天可能會有一波 股價上漲的反彈。
    乖離率過小時，表示股票在低點，有很大的機率會上漲
'''
# 計算五日、十日乖離值
# ma_bias_ratio(day1, day2) 計算的是長短天期之間的乖離率, 不是收盤價的乖離率
# day1 為短天期, day2 為長天期, 分別計算其移動平均值, 將得到的短天期減掉長天期即得
'''   bias[-1] = ave5[-1] - ave10[-1]   '''
bias = stock.ma_bias_ratio(5, 10)  # 傳回短天期 day1 與長天期 day2 均價之乖離值


# In[ 交易訊號 ]

# 內建的買賣點分析 BestFourPoint(分析4大買賣點)
# 以技術面最簡單的移動平均線與成交量數據來判斷一檔股票的買進或賣出時機
'''
買進訊號：
o 量大收紅
o 量縮價不跌
o 三日均價由下往上
o 三日均價大於六日均價

賣出訊號：
o 量大收黑
o 量縮價跌
o 三日均價由上往下
o 三日均價小於六日均價
'''
stock2330 = twstock.Stock('2330')
bfp = twstock.BestFourPoint(stock2330)
bfp.best_four_point_to_buy()   # 判斷是否為四大買點
# '三日均價大於六日均價'
bfp.best_four_point_to_sell()  # 判斷是否為四大賣點
# '量縮價跌, 三日均價由上往下'
bfp.best_four_point()          # 綜合判斷
# (True, '三日均價大於六日均價')
# 如果為買點，回傳 (True, msg)，如果為賣點，回傳 (False, msg)， 如果皆不符合，回傳 None。

