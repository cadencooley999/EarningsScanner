import requests
import yfinance as yf
import time
import datetime
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.interpolate import interp1d
import sqlite3

print("")

class Asset():
    def __init__(self, price):
        self.price = price

    def _get_price(self):
        return self.price
    
    def _set_price(self, new_price):
        self.price = new_price

class Equity(Asset):
    def __init__(self, price, ticker, company_name, term_struct_slope, volume_30d, iv_rv_30d, next_earnings):
        super().__init__(price)
        self.ticker = ticker
        self.company_name = company_name
        self.term_struct_slope = term_struct_slope
        self.volume_30d = volume_30d
        self.iv_rv_30d = iv_rv_30d
        self.next_earnings = next_earnings
class SQLiteConnect():
    def __init__(self):
        self.name = "ticker_store"
        self.connection = sqlite3.connect(self.name)
        self.cur = self.connection.cursor()
        self.cur.execute("""CREATE TABLE IF NOT EXISTS ticker_store (
            ticker TEXT PRIMARY KEY,
            last_price REAL,
            last_term_struct_slope REAL,
            last_30d_volume REAL,
            last_30d_IVRV REAL,
            last_recommendation TEXT,
            earnings_date TEXT,
            last_updated TIMESTAMP Default CURRENT_TIMESTAMP                        
        )""")

    def add_ticker(self, ticker, ticker_data, recommendation):
        self.cur.execute("""INSERT OR REPLACE INTO ticker_store (ticker, last_price, last_term_struct_slope, last_30d_volume, last_30d_IVRV, last_recommendation, earnings_date) VALUES (?,?,?,?,?,?,?)""", (ticker, ticker_data.price, ticker_data.term_struct_slope, ticker_data.volume_30d, ticker_data.iv_rv_30d, recommendation, ticker_data.next_earnings))
        self.connection.commit()

    def read_all_tickers(self):
        self.cur.execute("""SELECT * FROM ticker_store""")
        rows = self.cur.fetchall()
        print("-"*10, "Saved tickers", "-"*10)
        for i, row in enumerate(rows):
            print(f"{i+1}.", row[0])
        return rows
    
    def remove_ticker(self, tick):
        self.cur.execute("""Delete FROM ticker_store WHERE ticker = ?""", (tick,))
        
    def delete_all_tickers(self):
        self.cur.execute("""Delete from ticker_store""")

    def close_connection(self):
        self.connection.commit()
        self.connection.close()

class TickerAPIConnect():

    def __init__(self, ticker):
        self.ticker = ticker
        self.stock = yf.Ticker(self.ticker)
        self.options = None

    def populate_company_data(self):
        ### Metrics to track (term structure slopes {IV over time}, term structure ratio, 30 avg volume, IV/RV ratios)
        ### the more negative the slope (below 5th decile), the higher the returns
        ### Higher pre earnings trading volume (at or above 6th decile) = higher returns
        ### Higher IV/RV Ratio (at or above 5th decile) = higher returns
        success = True
        try:
            current_price = float(self._fetch_current_price())
        except:
            current_price = 0.0
            success = False
            print("error fetching current price")
        try:
            term_slope = float(self._fetch_term_structure_slope())
        except:
            term_slope = 0.0
            print("error fetching term slope")
            success = False
        try:
            dvol = float(self._fetch_30d_vol())
        except:
            dvol = 0.0
            print("error fetching 3od vol")
            success = False
        try:
            iv_rv_ratio = float(self._fetch_iv_rv_ratio())
        except:
            iv_rv_ratio = 0.0
            print("error fetching iv/rv")
            success = False
        next_earnings = self._fetch_next_earnings()
        company_data = Equity(price=current_price, ticker=self.ticker, company_name=self._get_company_name(), term_struct_slope=term_slope, volume_30d=dvol, iv_rv_30d=iv_rv_ratio, next_earnings=next_earnings)
        return company_data, success 
    
    def _yang_zhang_rv_alg(self, price_data, window=30, trading_periods=252, return_last_only=True):
        log_ho = (price_data['High'] / price_data['Open']).apply(np.log)
        log_lo = (price_data['Low'] / price_data['Open']).apply(np.log)
        log_co = (price_data['Close'] / price_data['Open']).apply(np.log)
        
        log_oc = (price_data['Open'] / price_data['Close'].shift(1)).apply(np.log)
        log_oc_sq = log_oc**2
        
        log_cc = (price_data['Close'] / price_data['Close'].shift(1)).apply(np.log)
        log_cc_sq = log_cc**2
        
        rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
        
        close_vol = log_cc_sq.rolling(window=window).sum() * (1.0 / (window - 1.0))
        open_vol = log_oc_sq.rolling(window=window).sum() * (1.0 / (window - 1.0))
        window_rs = rs.rolling(window=window).sum() * (1.0 / (window - 1.0))
        
        k = 0.34 / (1.34 + ((window + 1) / (window - 1)))
        result = (open_vol + k * close_vol + (1 - k) * window_rs).apply(np.sqrt) * np.sqrt(trading_periods)
        
        if return_last_only:
            return result.iloc[-1]
        else:
            return result.dropna()
    
    def _format_dates(self):
        self.options = self.stock.options
        exp_dates = self.options # list of expiration strings

        if not exp_dates:
            raise ValueError(f"No options found for {self.ticker}")

        today = datetime.today().date()
        
        # --- Convert expiration strings to dates ---
        exp_dates = [datetime.strptime(d, "%Y-%m-%d").date() for d in exp_dates]
        exp_dates = sorted(exp_dates)

        return today, exp_dates
    
    def _fetch_current_price(self):
        price_history = self.stock.history(period="1d", interval="1m")
        latest_price = price_history["Close"].iloc[-1]

        return latest_price

    def _fetch_term_structure_slope(self):
        stock = self.stock
        
        today, exp_dates = self._format_dates()

        nearest_exp = exp_dates[0]

        # --- Find the expiration closest to 45 days out ---
        target = today + timedelta(days=45)
        exp_45d = min(exp_dates, key=lambda d: abs((d - target).days))

        # --- Helper to compute ATM IV at a given expiration ---
        def get_atm_iv(exp_date):
            chain = stock.option_chain(exp_date.strftime("%Y-%m-%d"))
            calls, puts = chain.calls, chain.puts
            underlying_price = stock.history(period="1d")["Close"].iloc[0]

            # nearest strike
            call_idx = (calls['strike'] - underlying_price).abs().idxmin()
            put_idx = (puts['strike'] - underlying_price).abs().idxmin()

            call_iv = calls.loc[call_idx, 'impliedVolatility']
            put_iv  = puts.loc[put_idx, 'impliedVolatility']
            return (call_iv + put_iv) / 2.0

        iv_near = get_atm_iv(nearest_exp)
        iv_45d  = get_atm_iv(exp_45d)

        return (iv_45d - iv_near)/(exp_45d - nearest_exp).days

    def _fetch_30d_vol(self):
        stock = self.stock
        try:
            price_history = stock.history(period="3mo")
            avg_volume_30d = price_history['Volume'].rolling(window=30).mean().iloc[-1]
            return avg_volume_30d
        except:
            print("error fetching 30d volume")
            return 0.0

    
    def _fetch_iv_rv_ratio(self):
        stock = self.stock
        today, exp_dates = self._format_dates()

        # --- Step 1: Collect ATM IVs across expirations ---
        atm_iv = {}
        spot = stock.history(period="1d")["Close"].iloc[0]

        for exp_date in exp_dates:
            try:
                chain = stock.option_chain(exp_date.strftime("%Y-%m-%d"))
                calls, puts = chain.calls, chain.puts

                if calls.empty or puts.empty:
                    continue

                call_idx = (calls['strike'] - spot).abs().idxmin()
                put_idx = (puts['strike'] - spot).abs().idxmin()

                call_iv = calls.loc[call_idx, 'impliedVolatility']
                put_iv = puts.loc[put_idx, 'impliedVolatility']

                atm_iv[exp_date] = (call_iv + put_iv) / 2.0
            except Exception:
                continue

        if not atm_iv:
            raise ValueError("No valid ATM IVs found for term structure.")

        # --- Step 2: Interpolate to 30-day IV ---
        dtes = [(d - today).days for d in atm_iv.keys()]
        ivs = list(atm_iv.values())

        spline = interp1d(dtes, ivs, kind="linear", fill_value="extrapolate")
        iv30 = float(spline(30))

        # --- Step 3: Realized volatility (close-to-close) ---
        price_history = stock.history(period="3mo")
        rv30 = self._yang_zhang_rv_alg(price_history)

        # --- Step 4: Ratio ---
        return iv30 / rv30
    
    def _get_company_name(self):
        t = self.stock
        info = t.info
        # “longName” is the full company name (e.g. “Microsoft Corporation”)
        return info.get("longName") or info.get("shortName") or None
    
    def _fetch_next_earnings(self):
        try:
            earnings_info = self.stock.calendar

            if not earnings_info:  # empty or None
                return ""

            next_date = earnings_info["Earnings Date"][0]  # first value in the list

            # normalize to string YYYY-MM-DD
            return next_date.strftime("%Y-%m-%d") if hasattr(next_date, "strftime") else str(next_date)

        except Exception as e:
            print(f"An error occurred while fetching next earnings data: {e}")
            return ""

    
class EarningsCalendarScraper():

    def __init__(self, date):
        self.date = date

    def _scrapeNasdaqCalendar(self):
        ### returns json of companies
        calendar_json = ""
        calendar_url = f"https://api.nasdaq.com/api/calendar/earnings?date={self.date}"
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json, text/plain, */*"
        }
        try: 
            resp = requests.get(calendar_url, headers=headers, timeout=10)
            resp.raise_for_status()
            # print(str(resp.json())[:500])
            calendar_json = resp.json()
            return calendar_json
        except:
            print("Error fetching calendar json")
            return ""

    def _parseCalendarJson(self, json):
        ### returns list of stock tickers
        tickers = []

        try: 
            for i in json["data"]["rows"]:
                tickers.append((i["symbol"], i["name"]))
        except:
            print()

        self.__tickers = tickers

    def _displayTickerInfo(self):
        print(f"Found {len(self.__tickers)} companies with earnings reports on this date")
        print("Top 5: \n")
        for i in self.__tickers[:5]:
            print(f"{i[0]} - ({i[1]})")

    def getTop5Tickers(self):
        json = self._scrapeNasdaqCalendar()
        self._parseCalendarJson(json)
        self._displayTickerInfo()
        return self.__tickers[:5] 
    
    def getTop5NoDisplay(self):
        json = self._scrapeNasdaqCalendar()
        self._parseCalendarJson(json)
        return self.__tickers[:5] 

    
class TradeRecommender():

    def __init__(self, ticker_data):
        self.ticker_data = ticker_data
        self.term_struct_deciles = [0.0003, -0.0006, -0.0012, -0.0018, -0.0025, -0.0036, -0.0051, -0.0075, -0.0125, -0.6165]
        self.volume_deciles = [500004.5455, 611245.2381, 749909.3455, 924895.2243, 1147992.8571, 1454945.0000, 1878818.3091, 2536831.2782, 3749598.0952, 674397717.4289]
        self.iv_rv_deciles = [0.8859, 1.0119, 1.1002, 1.1770, 1.2543, 1.3358, 1.4320, 1.5602, 1.7772, 45.5341]

    def generate_recommendation(self):
        term_decile = self.get_term_decile()
        volume_decile = self.get_volume_decile()
        rv_decile = self.get_rv_decile()
        term_rating = "Good" if term_decile > 6 else "Ok" if term_decile == 6 else "Bad"
        volume_rating = "Good" if volume_decile > 6 else "Ok" if volume_decile == 6 else "Bad"
        rv_rating = "Good" if rv_decile > 5 else "Ok" if rv_decile == 5 else "Bad"
        print("="*10, f"Ticker: {self.ticker_data.ticker} ({self.ticker_data.company_name})", "="*10, "\n")
        if len(self.ticker_data.next_earnings) > 0:
            print(f"{"Next Earnings Date: ":25}{self.ticker_data.next_earnings}")
        print(f"{"Term structure slope: ":25}{self.ticker_data.term_struct_slope: .6f} [decile {term_decile}, - {term_rating}]")
        print(f"{"30d Volume: ":25}{self.ticker_data.volume_30d: .2f} [decile {volume_decile} - {volume_rating}]")
        print(f"{"30d IV/RV ratio ":25}{self.ticker_data.iv_rv_30d: .4f} [decile {rv_decile} - {rv_rating}]\n")
        print(f"FINAL RECOMMENDATION: {"Trade" if term_rating == "Good" and volume_rating == "Good" and rv_rating == "Good" else "Consider" if term_decile >= 6 and volume_decile >= 5 and rv_decile >= 4 else "Do not trade"}")
        return "Trade" if term_rating == "Good" and volume_rating == "Good" and rv_rating == "Good" else "Consider" if term_decile >= 6 and volume_decile >= 5 and rv_decile >= 4 else "Do not trade"

    def generateWithoutDisplay(self):
        term_decile = self.get_term_decile()
        volume_decile = self.get_volume_decile()
        rv_decile = self.get_rv_decile()
        term_rating = "Good" if term_decile > 6 else "Ok" if term_decile == 6 else "Bad"
        volume_rating = "Good" if volume_decile > 6 else "Ok" if volume_decile == 6 else "Bad"
        rv_rating = "Good" if rv_decile > 5 else "Ok" if rv_decile == 5 else "Bad"
        return "Trade" if term_rating == "Good" and volume_rating == "Good" and rv_rating == "Good" else "Consider" if term_decile >= 6 and volume_decile >= 5 and rv_decile >= 4 else "Do not trade"

    def get_term_decile(self):
        term_struct = self.ticker_data.term_struct_slope
        for i, x in enumerate(self.term_struct_deciles):
            if term_struct > x:
                return i + 1
        return 10
    
    def get_volume_decile(self):
        volume = self.ticker_data.volume_30d
        for i, x in enumerate(self.volume_deciles):
            if volume < x:
                return i + 1
        return 10

    def get_rv_decile(self):
        rv = self.ticker_data.iv_rv_30d
        for i, x in enumerate(self.iv_rv_deciles):
            if rv < x:
                return i + 1
        return 1

def search_by_date():
    while True:
        dateWanted = input("Please input a date to get earnings (yyyy-mm-dd): ")
        if len(dateWanted) == 10:
            break
        else:
            dateWanted = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
            print(f"Finding earnings reports for tomorrow {dateWanted}")
            break
    todayScraper = EarningsCalendarScraper(dateWanted)
    try:
        todaysTickers = todayScraper.getTop5Tickers()
    except Exception:
        print("Unable to find earnings for this date. May be a weekend")
        return

    if not todaysTickers:
        print("No earnings found for this date")
        return

    while True:
        chosenTicker = input("please enter a ticker to see data (0 to exit): ")
        if chosenTicker == "0":
            print("Exiting to main menu.")
            break
        else:
            tickerConn = TickerAPIConnect(chosenTicker.upper())
            tickerData, success = tickerConn.populate_company_data()
            if success:
                recommender = TradeRecommender(ticker_data=tickerData)
                recommendation = recommender.generate_recommendation()
                to_store = input("Save ticker? (y/n): ")
                if to_store == "y":
                    try:
                        conn = SQLiteConnect()
                        conn.add_ticker(chosenTicker, tickerData, recommendation)
                        conn.close_connection()
                        print("Ticker saved.")
                    except Exception as e:
                        print(f"Error storing: {e}")
            else:
                print(f"Incomplete data found for ticker {chosenTicker}. No Recommendation")

def searchByTicker():
    while True:
        chosenTicker = input("Enter a ticker to see data (0 to go back): ")
        if chosenTicker == "0":
            print("Exiting to main menu.")
            break
        tickerConn = TickerAPIConnect(chosenTicker.upper())
        tickerData, success = tickerConn.populate_company_data()
        if success:
            recommender = TradeRecommender(ticker_data=tickerData)
            recommendation = recommender.generate_recommendation()
            to_store = input("Save ticker? (y/n): ")
            if to_store == "y":
                try:
                    conn = SQLiteConnect()
                    conn.add_ticker(chosenTicker, tickerData, recommendation)
                    conn.close_connection()
                    print("Ticker saved.")
                except Exception as e:
                    print(f"Error storing: {e}")
        else:
            print(f"Incomplete data found for ticker {chosenTicker}. No Recommendation")

def scanNext5Days():
    ## GET NEXT 5 DATES
    ## SCAN TOP 5 TICKERS IN EACH
    ## STORE RECOMMENDED TICKERS
    ## DISPLAY FINAL TICKERS WITH DATE LET THEM SELECT
    ## DISPLAY INFO FOR SELECTED TICKER
    ## ALLOW THEM TO SAVE AND EXIT OR REPEAT SELECTION LOOP
    dates = [(datetime.now() + timedelta(days=1+x)).strftime("%Y-%m-%d") for x in range(5)]
    tickers = []
    ticker_data = []
    recommendedTickers = []

    print("\nDEBUG: ")

    def fetch_tickers(d):
        scraper = EarningsCalendarScraper(date=d)
        return scraper.getTop5NoDisplay()

    with ThreadPoolExecutor(max_workers=5) as ex:
        futures = {ex.submit(fetch_tickers, d): d for d in dates}
        for future in as_completed(futures):
            try:
                tickers.extend(future.result())
            except Exception as e:
                print(f"Error on {futures[future]}: {e}")

    def fetch_ticker_data(t):
        time.sleep(0.2)
        try:
            api_conn = TickerAPIConnect(t[0])
            return api_conn.populate_company_data()
        except Exception as e:
            print(f"[DEBUG] Failed to fetch data for {t}: {e}")
            return None, False

    
    with ThreadPoolExecutor(max_workers=5) as ex:
        futures = {ex.submit(fetch_ticker_data, t): t for t in tickers}
        for future in as_completed(futures):
            try:
                ticker_data.append(future.result())
            except Exception as e:
                print(f"Error on {futures[future]}: {e}")
        
    def getRecommendation(data_resp):
        if data_resp[1]:
            recommender = TradeRecommender(data_resp[0])
            rec = recommender.generateWithoutDisplay()
            return rec, data_resp[0]
        else:
            return "", None

    with ThreadPoolExecutor(max_workers=3) as ex:
        futures = {ex.submit(getRecommendation, td): td for td in ticker_data}
        for future in as_completed(futures):
            try:
                if future.result()[0] == "Trade":
                    recommendedTickers.append(future.result()[1])
            except Exception as e:
                print(f"Error on {futures[future]}: {e}")

    print("\nRESULT: \n")

    print("RECOMMENDED TICKERS")

    for i, tick in enumerate(recommendedTickers):
        print(f"{i+1}. {tick.ticker:6} - {tick.next_earnings}")

    print()

    while True:
        try:
            choice = input("Enter a ticker to see more info (0 to go back): ")
            valid = False
            if choice == "0":
                print("Exiting")
                break
            for tick in recommendedTickers:
                if tick.ticker == choice.upper():
                    valid = True
                    reco = TradeRecommender(ticker_data=tick)
                    rec = reco.generate_recommendation()
                    to_store = input("Save ticker? (y/n): ")
                    if to_store == "y":
                        try:
                            conn = SQLiteConnect()
                            conn.add_ticker(tick.ticker, tick, rec)
                            conn.close_connection()
                            print("Ticker saved.")
                        except Exception as e:
                            print(f"Error storing: {e}")
            if not valid:
                print("Not valid ticker")
        except Exception as e:
            print("Invalid Input")
            print(e)

def read_sql(conn):
    rows = conn.read_all_tickers()
    return rows

def view1_sql(conn, rows):
    while True:
        chosenTicker = input("Enter a ticker to see data (0 to go back): ")
        if chosenTicker == "0":
            print("Exiting to main menu.")
            break
        if chosenTicker.upper() in [row[0] for row in rows]:
            tickerConn = TickerAPIConnect(chosenTicker.upper())
            tickerData, success = tickerConn.populate_company_data()
            if success:
                print("Latest Data: ")
                recommender = TradeRecommender(ticker_data=tickerData)
                recommendation = recommender.generate_recommendation()
                to_store = input("Update ticker data? (y/n): ")
                if to_store == "y":
                    try:
                        conn.add_ticker(chosenTicker, tickerData, recommendation)
                        print("Ticker saved.")
                    except Exception as e:
                        print(f"Error storing: {e}")
            else:
                print(f"Incomplete data found for ticker {chosenTicker}. No Recommendation")
        else:
            print("Ticker not in library")


def rem_ticker(conn):
    while True:
        chosenTicker = input("Enter a ticker to delete it (0 to go back): ")
        if chosenTicker == "0":
            print("Exiting to main menu.")
            break
        try:
            conn.remove_ticker(chosenTicker.upper())
            print(f"Ticker {chosenTicker.upper()} removed")
        except Exception as e:
            print(f"Error removing ticker {chosenTicker.upper()}: {e}")
            print("Try again?")

def removeAllTickers(conn):
    conn.delete_all_tickers()

def operate_sql():
    while True:
        try: 
            conn = SQLiteConnect()
            rows = read_sql(conn)
            if len(rows) == 0:
                print("Empty")
            print("\n1. See ticker details\n"
            "2. Remove a ticker\n"
            "3. Remove all tickers\n")
            menu_choice = int(input("What would you like to do? (0 to go back): "))
            if menu_choice == 1:
                view1_sql(conn, rows)
                conn.close_connection()
            elif menu_choice == 2:
                rem_ticker(conn)
                conn.close_connection()
            elif menu_choice == 3:
                conn.delete_all_tickers()
                conn.close_connection()
            elif menu_choice == 0:
                conn.close_connection()
                print("Goodbye")
                break
            else: 
                conn.close_connection()
                print("Invalid Menu Choice")
        except Exception as e: 
            print("Incorrect input")
            print(e)


def display_welcome():
    print()
    print("-"*10, "Welcome to the Call Calendar Analysis Machine for Upcoming Earnings Reports", "-"*10)

def display_menu():
    print("\n1. Ticker Search\n"
    "2. Scan Next 5 days\n"
    "3. Search by date\n"
    "4. Open My Ticker Library\n")

def main():
    display_welcome()
    while True:
        try: 
            display_menu()
            menu_choice = int(input("What would you like to do today? (0 to exit): ")) 
            if menu_choice == 1:
                searchByTicker()
            elif menu_choice == 2:
                scanNext5Days()
            elif menu_choice == 3:
                search_by_date()
            elif menu_choice == 4:
                operate_sql()
            elif menu_choice == 0:
                print("Goodbye")
                break
            else: 
                print("Invalid Menu Choice")
        except Exception as e: 
            print("Incorrect input")
            print(e)

def test():
    conn = SQLiteConnect()
    # company_data = {"ticker": "CKK", "company_name":"Carnival", "price": 99, "term_struct_slope":-0.04, "30d_volume":12340.3, "30d_IV/RV":1.5}
    # conn.add_ticker("CKK", company_data, "recommendation", "dateWanted")
    print(conn.read_all_tickers())
    conn.close_connection()



if __name__ == "__main__":
    main()
