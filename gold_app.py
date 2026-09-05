import tkinter as tk
from tkinter import ttk, messagebox
import requests
import pandas as pd
import numpy as np

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle
import matplotlib.dates as mdates


# =========================================================
# SETTINGS
# =========================================================

SYMBOL = "XAU/USD"

# अपनी Twelve Data API key यहां डालें
API_KEY = "c968f3e4b6ba49f89b74c840f84c7d8e"

AUTO_REFRESH_SECONDS = 1

TIMEFRAMES = {
    "1M": "1min",
    "5M": "5min",
    "15M": "15min",
    "1H": "1h",
    "4H": "4h",
    "1D": "1day"
}


# =========================================================
# GLOBAL
# =========================================================

canvas = None
refresh_job = None
main_ax = None


# =========================================================
# GET LIVE DATA
# =========================================================

def get_data(timeframe):

    interval = TIMEFRAMES[timeframe]

    url = "https://api.twelvedata.com/time_series"

    params = {
        "symbol": SYMBOL,
        "interval": interval,
        "outputsize": 300,
        "apikey": API_KEY
    }

    response = requests.get(
        url,
        params=params,
        timeout=15
    )

    response.raise_for_status()

    result = response.json()

    if result.get("status") == "error":

        raise Exception(
            result.get(
                "message",
                "Twelve Data API error"
            )
        )

    if "values" not in result:

        raise Exception(
            "XAU/USD market data नहीं मिला।"
        )

    data = pd.DataFrame(
        result["values"]
    )

    data["datetime"] = pd.to_datetime(
        data["datetime"]
    )

    data = data.set_index(
        "datetime"
    )

    data = data.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume"
        }
    )

    for column in [
        "Open",
        "High",
        "Low",
        "Close"
    ]:

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    if "Volume" not in data.columns:

        data["Volume"] = 0

    data["Volume"] = pd.to_numeric(
        data["Volume"],
        errors="coerce"
    ).fillna(0)

    data = data.sort_index()

    data = data.dropna(
        subset=[
            "Open",
            "High",
            "Low",
            "Close"
        ]
    )

    if len(data) < 50:

        raise Exception(
            f"Not enough data. "
            f"Only {len(data)} candles received."
        )

    return data


# =========================================================
# INDICATORS
# =========================================================

def calculate_indicators(data):

    close = data["Close"]
    high = data["High"]
    low = data["Low"]
    volume = data["Volume"]

    # -----------------------------------------------------
    # EMA
    # -----------------------------------------------------

    data["EMA9"] = close.ewm(
        span=9,
        adjust=False
    ).mean()

    data["EMA18"] = close.ewm(
        span=18,
        adjust=False
    ).mean()

    data["EMA50"] = close.ewm(
        span=50,
        adjust=False
    ).mean()

    data["EMA200"] = close.ewm(
        span=200,
        adjust=False
    ).mean()

    # -----------------------------------------------------
    # VWAP
    # -----------------------------------------------------

    typical_price = (
        high + low + close
    ) / 3

    cumulative_volume = volume.cumsum()

    cumulative_volume = cumulative_volume.replace(
        0,
        np.nan
    )

    data["VWAP"] = (
        typical_price * volume
    ).cumsum() / cumulative_volume

    data["VWAP"] = data["VWAP"].ffill()

    # If volume is unavailable
    data["VWAP"] = data["VWAP"].fillna(
        typical_price
    )

    # -----------------------------------------------------
    # RSI
    # -----------------------------------------------------

    change = close.diff()

    gain = change.clip(
        lower=0
    )

    loss = -change.clip(
        upper=0
    )

    avg_gain = gain.rolling(
        14
    ).mean()

    avg_loss = loss.rolling(
        14
    ).mean()

    rs = avg_gain / avg_loss.replace(
        0,
        np.nan
    )

    data["RSI"] = (
        100 -
        (
            100 /
            (1 + rs)
        )
    )

    # -----------------------------------------------------
    # MACD
    # -----------------------------------------------------

    ema12 = close.ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = close.ewm(
        span=26,
        adjust=False
    ).mean()

    data["MACD"] = (
        ema12 - ema26
    )

    data["MACD_SIGNAL"] = (
        data["MACD"].ewm(
            span=9,
            adjust=False
        ).mean()
    )

    data["MACD_HIST"] = (
        data["MACD"] -
        data["MACD_SIGNAL"]
    )

    # -----------------------------------------------------
    # ATR
    # -----------------------------------------------------

    tr1 = high - low

    tr2 = (
        high -
        close.shift(1)
    ).abs()

    tr3 = (
        low -
        close.shift(1)
    ).abs()

    true_range = pd.concat(
        [
            tr1,
            tr2,
            tr3
        ],
        axis=1
    ).max(axis=1)

    data["ATR"] = (
        true_range.rolling(
            14
        ).mean()
    )

    # -----------------------------------------------------
    # SUPERTREND
    # -----------------------------------------------------

    middle = (
        high + low
    ) / 2

    upper = (
        middle +
        3 * data["ATR"]
    )

    lower = (
        middle -
        3 * data["ATR"]
    )

    trend = []

    current_trend = 1

    for i in range(len(data)):

        if i == 0:

            trend.append(
                current_trend
            )

            continue

        previous_upper = upper.iloc[i - 1]
        previous_lower = lower.iloc[i - 1]
        current_close = close.iloc[i]

        if pd.isna(previous_upper):

            trend.append(
                current_trend
            )

            continue

        if current_close > previous_upper:

            current_trend = 1

        elif current_close < previous_lower:

            current_trend = -1

        trend.append(
            current_trend
        )

    data["SUPERTREND"] = trend

    return data


# =========================================================
# MARKET STRUCTURE
# =========================================================

def get_structure(data):

    recent = data.tail(30)

    if len(recent) < 10:

        return "NEUTRAL"

    recent_high = recent["High"]

    recent_low = recent["Low"]

    last_high = recent_high.iloc[-1]

    last_low = recent_low.iloc[-1]

    previous_high = recent_high.iloc[:-1].max()

    previous_low = recent_low.iloc[:-1].min()

    if last_high > previous_high:

        return "BULLISH HH"

    if last_low < previous_low:

        return "BEARISH LL"

    return "RANGE"


# =========================================================
# LIQUIDITY
# =========================================================

def get_liquidity(data):

    if len(data) < 10:

        return "NONE"

    previous = data.iloc[-10:-1]

    current = data.iloc[-1]

    old_high = previous["High"].max()

    old_low = previous["Low"].min()

    if (
        current["High"] > old_high
        and
        current["Close"] < old_high
    ):

        return "HIGH SWEEP"

    if (
        current["Low"] < old_low
        and
        current["Close"] > old_low
    ):

        return "LOW SWEEP"

    return "NONE"


# =========================================================
# PREDICTION
# =========================================================

def calculate_prediction(data):

    last = data.iloc[-1]

    score = 0

    # EMA 9 / 18
    if last["EMA9"] > last["EMA18"]:

        score += 1

    else:

        score -= 1

    # Price / EMA50
    if last["Close"] > last["EMA50"]:

        score += 1

    else:

        score -= 1

    # Price / EMA200
    if last["Close"] > last["EMA200"]:

        score += 1

    else:

        score -= 1

    # VWAP
    if last["Close"] > last["VWAP"]:

        score += 1

    else:

        score -= 1

    # MACD
    if last["MACD"] > last["MACD_SIGNAL"]:

        score += 1

    else:

        score -= 1

    # RSI
    if last["RSI"] > 50:

        score += 1

    else:

        score -= 1

    # Supertrend
    if last["SUPERTREND"] == 1:

        score += 1

    else:

        score -= 1

    # Structure
    structure = get_structure(data)

    if "BULLISH" in structure:

        score += 1

    elif "BEARISH" in structure:

        score -= 1

    # Liquidity
    liquidity = get_liquidity(data)

    if liquidity == "LOW SWEEP":

        score += 1

    elif liquidity == "HIGH SWEEP":

        score -= 1

    # -----------------------------------------------------
    # SIGNAL
    # -----------------------------------------------------

    if score >= 5:

        signal = "BUY"

    elif score <= -5:

        signal = "SELL"

    else:

        signal = "WAIT"

    confidence = min(
        95,
        50 + abs(score) * 5
    )

    price = float(
        last["Close"]
    )

    atr = float(
        last["ATR"]
    )

    if (
        not np.isfinite(atr)
        or
        atr <= 0
    ):

        atr = price * 0.002

    # -----------------------------------------------------
    # ENTRY / SL / TP
    # -----------------------------------------------------

    entry = price

    if signal == "BUY":

        sl = price - (
            atr * 1.5
        )

        tp1 = price + (
            atr * 1.5
        )

        tp2 = price + (
            atr * 3
        )

    elif signal == "SELL":

        sl = price + (
            atr * 1.5
        )

        tp1 = price - (
            atr * 1.5
        )

        tp2 = price - (
            atr * 3
        )

    else:

        sl = price
        tp1 = price
        tp2 = price

    return {
        "signal": signal,
        "confidence": confidence,
        "entry": entry,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "structure": structure,
        "liquidity": liquidity,
        "score": score
    }


# =========================================================
# DRAW CANDLES
# =========================================================

def draw_candles(ax, data):

    if len(data) > 1:

        x1 = mdates.date2num(
            data.index[-2]
        )

        x2 = mdates.date2num(
            data.index[-1]
        )

        width = abs(
            x2 - x1
        ) * 0.65

    else:

        width = 0.01

    for i in range(len(data)):

        row = data.iloc[i]

        x = mdates.date2num(
            data.index[i]
        )

        open_price = float(
            row["Open"]
        )

        close_price = float(
            row["Close"]
        )

        high = float(
            row["High"]
        )

        low = float(
            row["Low"]
        )

        if close_price >= open_price:

            candle_color = "#00d084"

        else:

            candle_color = "#ff3b30"

        # Wick
        ax.plot(
            [x, x],
            [low, high],
            color=candle_color,
            linewidth=1
        )

        # Body
        bottom = min(
            open_price,
            close_price
        )

        height = abs(
            close_price -
            open_price
        )

        if height == 0:

            height = 0.01

        rectangle = Rectangle(
            (
                x - width / 2,
                bottom
            ),
            width,
            height,
            facecolor=candle_color,
            edgecolor=candle_color
        )

        ax.add_patch(
            rectangle
        )
# =========================================================
# CHART ZOOM & PAN
# =========================================================

def on_scroll(event):
    if event.inaxes is None:
        return

    ax = event.inaxes

    x_left, x_right = ax.get_xlim()
    y_bottom, y_top = ax.get_ylim()

    if event.button == "up":
        scale = 0.8
    elif event.button == "down":
        scale = 1.25
    else:
        return

    x_center = (x_left + x_right) / 2
    y_center = (y_bottom + y_top) / 2

    x_range = (x_right - x_left) * scale
    y_range = (y_top - y_bottom) * scale

    ax.set_xlim(x_center - x_range / 2, x_center + x_range / 2)
    ax.set_ylim(y_center - y_range / 2, y_center + y_range / 2)

    event.canvas.draw_idle()


def zoom_chart(scale):

    if main_ax is None or canvas is None:
        return

    x_left, x_right = main_ax.get_xlim()
    y_bottom, y_top = main_ax.get_ylim()

    x_center = (x_left + x_right) / 2
    y_center = (y_bottom + y_top) / 2

    x_range = (x_right - x_left) * scale
    y_range = (y_top - y_bottom) * scale

    main_ax.set_xlim(x_center - x_range / 2, x_center + x_range / 2)
    main_ax.set_ylim(y_center - y_range / 2, y_center + y_range / 2)

    canvas.draw_idle()


def zoom_in():
    zoom_chart(0.8)


def zoom_out():
    zoom_chart(1.25)


# =========================================================
# CREATE CHART
# =========================================================

def create_chart(data, result, timeframe):

    global canvas
    global main_ax

    fig = Figure(
        figsize=(15, 8),
        dpi=100,
        facecolor="#071018"
    )

    fig.canvas.mpl_connect("scroll_event", on_scroll)

    ax = fig.add_axes(
        [0.05, 0.38, 0.70, 0.54]
    )

    main_ax = ax

    macd_ax = fig.add_axes(
        [0.05, 0.21, 0.70, 0.12],
        sharex=ax
    )

    rsi_ax = fig.add_axes(
        [0.05, 0.06, 0.70, 0.11],
        sharex=ax
    )

    panel = fig.add_axes(
        [0.78, 0.06, 0.20, 0.86]
    )

    axes = [
        ax,
        macd_ax,
        rsi_ax
    ]

    for current_ax in axes:

        current_ax.set_facecolor(
            "#071018"
        )

        current_ax.tick_params(
            colors="white",
            labelsize=7
        )

        for spine in current_ax.spines.values():

            spine.set_color(
                "#26333f"
            )

        current_ax.grid(
            True,
            alpha=0.15
        )

    # =====================================================
    # CANDLES
    # =====================================================

    draw_candles(
        ax,
        data
    )

    # =====================================================
    # EMAs
    # =====================================================

    ax.plot(
        data.index,
        data["EMA9"],
        color="#00a8ff",
        linewidth=1.3,
        label="EMA 9"
    )

    ax.plot(
        data.index,
        data["EMA18"],
        color="#ff9800",
        linewidth=1.3,
        label="EMA 18"
    )

    ax.plot(
        data.index,
        data["EMA50"],
        color="#f5d90a",
        linewidth=1,
        label="EMA 50"
    )

    ax.plot(
        data.index,
        data["EMA200"],
        color="#ff3b30",
        linewidth=1,
        label="EMA 200"
    )

    # VWAP
    ax.plot(
        data.index,
        data["VWAP"],
        color="#b45cff",
        linewidth=1.3,
        label="VWAP"
    )

    # =====================================================
    # SUPPORT / RESISTANCE
    # =====================================================

    support = data["Low"].tail(50).min()

    resistance = data["High"].tail(50).max()

    ax.axhline(
        support,
        color="#00a8ff",
        linestyle=":",
        linewidth=1
    )

    ax.axhline(
        resistance,
        color="#ff9800",
        linestyle=":",
        linewidth=1
    )

    # =====================================================
    # ENTRY SL TP
    # =====================================================

    signal = result["signal"]

    if signal == "BUY":

        signal_color = "#00d084"

    elif signal == "SELL":

        signal_color = "#ff3b30"

    else:

        signal_color = "#ffd60a"

    ax.axhline(
        result["entry"],
        color="#00a8ff",
        linestyle="--",
        linewidth=1
    )

    if signal != "WAIT":

        ax.axhline(
            result["sl"],
            color="#ff3b30",
            linestyle="--",
            linewidth=1
        )

        ax.axhline(
            result["tp1"],
            color="#00d084",
            linestyle="--",
            linewidth=1
        )

        ax.axhline(
            result["tp2"],
            color="#00d084",
            linestyle=":",
            linewidth=1
        )

    # =====================================================
    # TITLE
    # =====================================================

    ax.text(
        0.01,
        0.95,
        f"XAU/USD | {timeframe}",
        transform=ax.transAxes,
        color="white",
        fontsize=13,
        fontweight="bold"
    )

    ax.legend(
        loc="upper left",
        fontsize=7,
        facecolor="#101c27",
        labelcolor="white"
    )

    # =====================================================
    # MACD
    # =====================================================

    macd_colors = np.where(
        data["MACD_HIST"] >= 0,
        "#00d084",
        "#ff3b30"
    )

    macd_ax.bar(
        data.index,
        data["MACD_HIST"],
        color=macd_colors,
        width=0.7
    )

    macd_ax.plot(
        data.index,
        data["MACD"],
        color="#00a8ff",
        linewidth=1
    )

    macd_ax.plot(
        data.index,
        data["MACD_SIGNAL"],
        color="#ff9800",
        linewidth=1
    )

    macd_ax.set_ylabel(
        "MACD",
        color="white",
        fontsize=8
    )

    # =====================================================
    # RSI
    # =====================================================

    rsi_ax.plot(
        data.index,
        data["RSI"],
        color="#b45cff",
        linewidth=1.3
    )

    rsi_ax.axhline(
        70,
        color="#ff3b30",
        linestyle="--",
        linewidth=0.7
    )

    rsi_ax.axhline(
        30,
        color="#00d084",
        linestyle="--",
        linewidth=0.7
    )

    rsi_ax.set_ylim(
        0,
        100
    )

    rsi_ax.set_ylabel(
        "RSI",
        color="white",
        fontsize=8
    )

    # =====================================================
    # INFORMATION PANEL
    # =====================================================

    panel.set_facecolor(
        "#0b151f"
    )

    panel.set_xticks([])
    panel.set_yticks([])

    for spine in panel.spines.values():

        spine.set_color(
            "#26333f"
        )

    panel.text(
        0.5,
        0.95,
        "MARKET PREDICTION",
        ha="center",
        color="white",
        fontsize=12,
        fontweight="bold"
    )

    panel.text(
        0.5,
        0.85,
        signal,
        ha="center",
        color=signal_color,
        fontsize=28,
        fontweight="bold"
    )

    panel.text(
        0.5,
        0.76,
        f"{result['confidence']:.0f}%",
        ha="center",
        color=signal_color,
        fontsize=22,
        fontweight="bold"
    )

    panel.text(
        0.5,
        0.71,
        "CONFIDENCE",
        ha="center",
        color="#9ba9b5",
        fontsize=8
    )

    panel.text(
        0.08,
        0.62,
        f"ENTRY\n{result['entry']:.2f}",
        color="#00a8ff",
        fontsize=9
    )

    panel.text(
        0.08,
        0.52,
        f"STOP LOSS\n{result['sl']:.2f}",
        color="#ff3b30",
        fontsize=9
    )

    panel.text(
        0.08,
        0.42,
        f"TP 1\n{result['tp1']:.2f}",
        color="#00d084",
        fontsize=9
    )

    panel.text(
        0.08,
        0.32,
        f"TP 2\n{result['tp2']:.2f}",
        color="#00d084",
        fontsize=9
    )

    last = data.iloc[-1]

    panel.text(
        0.08,
        0.25,
        "INDICATORS",
        color="white",
        fontsize=10,
        fontweight="bold"
    )

    ema_status = (
        "Bullish"
        if last["EMA9"] > last["EMA18"]
        else "Bearish"
    )

    vwap_status = (
        "Above"
        if last["Close"] > last["VWAP"]
        else "Below"
    )

    supertrend_status = (
        "Bullish"
        if last["SUPERTREND"] == 1
        else "Bearish"
    )

    panel.text(
        0.08,
        0.20,
        f"EMA 9/18: {ema_status}\n"
        f"VWAP: {vwap_status}\n"
        f"Supertrend: {supertrend_status}\n"
        f"RSI: {last['RSI']:.1f}\n"
        f"MACD: {last['MACD']:.2f}\n"
        f"Structure: {result['structure']}\n"
        f"Liquidity: {result['liquidity']}\n"
        f"Score: {result['score']}",
        color="#d6e0e8",
        fontsize=8,
        linespacing=1.5,
        va="top"
    )

    fig.suptitle(
        f"GOLD MARKET PREDICTION | {signal}",
        color=signal_color,
        fontsize=15,
        fontweight="bold",
        x=0.42,
        y=0.985
    )

        # Remove old chart
    if canvas is not None:

        canvas.get_tk_widget().destroy()

    canvas = FigureCanvasTkAgg(
        fig,
        master=chart_frame
    )

    canvas.mpl_connect("scroll_event", on_scroll)

    canvas.draw()

    canvas.get_tk_widget().pack(
        fill=tk.BOTH,
        expand=True
    )


# =========================================================
# ANALYZE
# =========================================================

def run_analysis():

    try:

        status_label.config(
            text="Connecting to live XAU/USD data..."
        )

        root.update_idletasks()

        timeframe = timeframe_var.get()

        data = get_data(
            timeframe
        )

        data = calculate_indicators(
            data
        )

        result = calculate_prediction(
            data
        )

        create_chart(
            data,
            result,
            timeframe
        )

        last_price = data["Close"].iloc[-1]

        status_label.config(
            text=(
                f"LIVE XAU/USD | "
                f"{timeframe} | "
                f"Price: {last_price:.2f} | "
                f"Signal: {result['signal']} | "
                f"Confidence: "
                f"{result['confidence']:.0f}%"
            )
        )

    except Exception as error:

        messagebox.showerror(
            "Market Data Error",
            str(error)
        )

        status_label.config(
            text="Unable to load market data."
        )


# =========================================================
# AUTO REFRESH
# =========================================================

def auto_refresh():

    run_analysis()

    global refresh_job

    refresh_job = root.after(
        AUTO_REFRESH_SECONDS * 1000,
        auto_refresh
    )


# =========================================================
# GUI
# =========================================================

root = tk.Tk()

root.title(
    "GOLD MARKET PREDICTION"
)

root.geometry(
    "1450x900"
)

root.configure(
    bg="#071018"
)


# =========================================================
# HEADER
# =========================================================

header = tk.Frame(
    root,
    bg="#0b151f",
    height=65
)

header.pack(
    fill=tk.X
)

header.pack_propagate(
    False
)

title = tk.Label(
    header,
    text="XAU/USD GOLD",
    bg="#0b151f",
    fg="white",
    font=("Arial", 20, "bold")
)

title.pack(
    side=tk.LEFT,
    padx=20
)


# =========================================================
# LIVE LABEL
# =========================================================

live_label = tk.Label(
    header,
    text="● LIVE",
    bg="#0b151f",
    fg="#00d084",
    font=("Arial", 11, "bold")
)

live_label.pack(
    side=tk.LEFT,
    padx=10
)


# =========================================================
# TIMEFRAME
# =========================================================

timeframe_var = tk.StringVar(
    value="5M"
)

timeframe_box = ttk.Combobox(
    header,
    textvariable=timeframe_var,
    values=list(
        TIMEFRAMES.keys()
    ),
    state="readonly",
    width=8
)

timeframe_box.pack(
    side=tk.RIGHT,
    padx=20
)


# =========================================================
# ANALYZE BUTTON
# =========================================================

analyze_button = tk.Button(
    header,
    text="ANALYZE",
    command=run_analysis,
    bg="#087f5b",
    fg="white",
    activebackground="#00a86b",
    font=("Arial", 11, "bold"),
    width=12
)

analyze_button.pack(
    side=tk.RIGHT,
    padx=5
)


# =========================================================
# CHART CONTROLS (Refresh / Zoom In / Zoom Out)
# =========================================================

refresh_button = tk.Button(
    header,
    text="🔄 Refresh",
    command=run_analysis,
    bg="#0b151f",
    fg="#00d084",
    activebackground="#1a2733",
    font=("Arial", 10, "bold"),
    relief="flat",
    width=10
)

refresh_button.pack(
    side=tk.RIGHT,
    padx=5
)

zoom_out_button = tk.Button(
    header,
    text="➖ Zoom Out",
    command=zoom_out,
    bg="#0b151f",
    fg="white",
    activebackground="#1a2733",
    font=("Arial", 10, "bold"),
    relief="flat",
    width=10
)

zoom_out_button.pack(
    side=tk.RIGHT,
    padx=5
)

zoom_in_button = tk.Button(
    header,
    text="➕ Zoom In",
    command=zoom_in,
    bg="#0b151f",
    fg="white",
    activebackground="#1a2733",
    font=("Arial", 10, "bold"),
    relief="flat",
    width=10
)

zoom_in_button.pack(
    side=tk.RIGHT,
    padx=5
)


# =========================================================
# CHART FRAME
# =========================================================

chart_frame = tk.Frame(
    root,
    bg="#071018"
)

chart_frame.pack(
    fill=tk.BOTH,
    expand=True,
    padx=8,
    pady=8
)


# =========================================================
# STATUS
# =========================================================

status_label = tk.Label(
    root,
    text="Select timeframe and click ANALYZE",
    bg="#0b151f",
    fg="#00d084",
    anchor="w",
    font=("Arial", 10)
)

status_label.pack(
    fill=tk.X,
    padx=8,
    pady=5
)


# =========================================================
# START
# =========================================================
root.after(1000, auto_refresh)
root.mainloop()