import pandas as pd
import ccxt
import yfinance as yf
import numpy as np

# ============================================================
#  Base: Interfaz de proveedor de datos
# ============================================================

class DataProvider:
    """Interfaz base para proveedores de datos OHLCV."""
    def get_ohlc(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        raise NotImplementedError("Debe implementarse en subclases.")


# ============================================================
#  Proveedor para Cripto (Binance vía CCXT)
# ============================================================

class CCXTProvider(DataProvider):
    """
    Proveedor de datos para cripto usando CCXT.
    NO reemplaza tu fetch_binance_ohlcv reparado.
    Solo se usa si querés una versión simple.
    Para tu optimizador, seguís usando fetch_binance_ohlcv().
    """

    def __init__(self, exchange_name="binance"):
        self.exchange = getattr(ccxt, exchange_name)({
            "enableRateLimit": True,
            "options": {"defaultType": "spot"}
        })

    def get_ohlc(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        """
        Descarga simple de OHLCV.
        NO reemplaza tu sistema reparado de recencia.
        Solo se usa si querés un fallback simple.
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        except Exception as e:
            print(f"[ERROR] CCXTProvider: {e}")
            return None

        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)

        return df.astype(float)


# ============================================================
#  Proveedor para Acciones / ETFs / Índices (Yahoo Finance)
# ============================================================

# ============================================================
#  Proveedor para Acciones / ETFs / Índices (Yahoo Finance)
# ============================================================

class YFinanceProvider(DataProvider):

    TF_MAP = {
        "1m":  "1m",
        "5m":  "5m",
        "15m": "15m",
        "1h":  "60m",
        "4h":  "60m",
        "1d":  "1d",
    }

    MAX_DAYS_TOTAL = {
        "1m":  7,
        "5m":  60,
        "15m": 60,
        "1h":  60,
        "4h":  60,
        "1d":  None,
    }

    CHUNK_DAYS = {
        "1m":  6,
        "5m":  29,
        "15m": 29,
        "1h":  29,
        "4h":  29,
        "1d":  3650,
    }

    def get_ohlc(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        if timeframe not in self.TF_MAP:
            print(f"[ERROR] YFinanceProvider: timeframe '{timeframe}' no soportado.")
            return None

        interval   = self.TF_MAP[timeframe]
        intraday   = timeframe in ["1m", "5m", "15m", "1h", "4h"]
        max_days   = self.MAX_DAYS_TOTAL[timeframe]
        chunk_days = self.CHUNK_DAYS[timeframe]
        ticker     = yf.Ticker(symbol)

        if not intraday:
            try:
                df = ticker.history(period="max", interval=interval,
                                    prepost=False, auto_adjust=False)
            except Exception as e:
                print(f"[ERROR] YFinanceProvider: {e}")
                return None
            return self._clean(df, timeframe, limit, symbol)

        now        = pd.Timestamp.now(tz="UTC")
        hard_start = now - pd.Timedelta(days=max_days - 1)
        chunks     = []
        end_dt     = now
        MAX_CHUNKS = 10

        for _ in range(MAX_CHUNKS):
            start_dt = max(end_dt - pd.Timedelta(days=chunk_days), hard_start)
            if start_dt >= end_dt:
                break

            try:
                chunk = ticker.history(
                    start=start_dt.strftime("%Y-%m-%d"),
                    end=end_dt.strftime("%Y-%m-%d"),
                    interval=interval,
                    prepost=True,
                    auto_adjust=False,
                )
            except Exception as e:
                print(f"[WARN] YFinanceProvider chunk: {e}")
                break

            if chunk is None or chunk.empty:
                break

            chunk.index = pd.to_datetime(chunk.index, utc=True)
            chunks.insert(0, chunk)

            if sum(len(c) for c in chunks) >= limit:
                break

            end_dt = chunk.index[0]
            if end_dt <= hard_start:
                break

        if not chunks:
            print(f"[ERROR] YFinanceProvider: No se recibieron datos para {symbol}.")
            return None

        df_full = pd.concat(chunks)
        df_full = df_full[~df_full.index.duplicated(keep="last")]
        df_full.sort_index(inplace=True)

        return self._clean(df_full, timeframe, limit, symbol)

    def _clean(self, df: pd.DataFrame, timeframe: str, limit: int, symbol: str) -> pd.DataFrame:
        # 1. Colapsar MultiIndex de Yahoo Finance si existiera
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]

        # Normalizar nombres de columnas a minúsculas
        df = df.rename(columns={
            "Open": "open", "High": "high",
            "Low": "low",   "Close": "close", "Volume": "volume"
        })

        if timeframe == "4h":
            df = df.resample("4h").agg({
                "open": "first", "high": "max",
                "low": "min",    "close": "last", "volume": "sum"
            }).dropna()

        df.index = pd.to_datetime(df.index, utc=True)
        
        # ============================================================
        # ELIMINAMOS EL FILTRO DE OUTLIERS INTERNO
        # (Yahoo Finance ya no corregirá nada, solo pasará los datos crudos)
        # ============================================================
        # El filtro de outliers se aplicará en clean_yfinance_outliers()
        # fuera de esta clase.
        
        # Asegurar que no hay valores nulos
        df = df.dropna()
        
        # Recortar al límite solicitado por el usuario
        total_disponible = len(df)
        if total_disponible < limit:
            print(
                f"[Aviso yfinance] Solicitaste {limit} velas, pero el servidor "
                f"gratuito de Yahoo solo retiene {total_disponible} velas continuas "
                f"para este timeframe."
            )
            actual_limit = total_disponible
        else:
            actual_limit = limit

        return df.tail(actual_limit).astype(float)