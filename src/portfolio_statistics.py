from dataclasses import dataclass
from datetime import datetime

import numpy as np


@dataclass
class PortfolioStatistics:
    total_return_pct: float
    cagr_pct: float
    annual_volatility_pct: float
    max_drawdown_pct: float
    rising_days: int
    falling_days: int
    flat_days: int
    avg_daily_return_pct: float
    avg_daily_gain_pct: float
    avg_daily_loss_pct: float
    best_day_pct: float
    worst_day_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    max_consecutive_gains: int
    max_consecutive_losses: int
    recovery_time_days: int | None
    transaction_costs_paid: float
    trades_executed: int
    total_days: int
    total_years: float
    initial_value: float
    final_value: float

    def to_dict(self) -> dict[str, float | int | str | None]:
        return {
            "total_return [%]": self.total_return_pct,
            "cagr [%]": self.cagr_pct,
            "annual_volatility [%]": self.annual_volatility_pct,
            "max_drawdown [%]": self.max_drawdown_pct,
            "-": "---",
            "rising_days": self.rising_days,
            "falling_days": self.falling_days,
            "flat_days": self.flat_days,
            "avg_daily_return [%]": self.avg_daily_return_pct,
            "avg_daily_gain [%]": self.avg_daily_gain_pct,
            "avg_daily_loss [%]": self.avg_daily_loss_pct,
            "best_day [%]": self.best_day_pct,
            "worst_day [%]": self.worst_day_pct,
            "--": "---",
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "---": "---",
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "max_consecutive_gains": self.max_consecutive_gains,
            "max_consecutive_losses": self.max_consecutive_losses,
            "recovery_time_days": self.recovery_time_days,
            "----": "---",
            "transaction_costs_paid": self.transaction_costs_paid,
            "trades_executed": self.trades_executed,
            "-----": "---",
            "total_days": self.total_days,
            "total_years": self.total_years,
            "initial_value": self.initial_value,
            "final_value": self.final_value,
        }


class PortfolioStatisticsCalculator:
    def __init__(self, risk_free_rate: float = 0.0, days_in_year: float = 365.25) -> None:
        self.risk_free_rate = risk_free_rate
        self.days_in_year = days_in_year

    @staticmethod
    def _max_consecutive(arr: np.ndarray, condition) -> int:
        if len(arr) == 0:
            return 0
        mask = condition(arr)
        padded = np.concatenate(([False], mask, [False]))
        edges = np.diff(padded.astype(int))
        starts = np.where(edges == 1)[0]
        ends = np.where(edges == -1)[0]
        return int(np.max(ends - starts)) if len(starts) > 0 else 0

    def calculate(
        self,
        values: list[float],
        days_of_investment: list[str],
        transaction_costs_paid: float = 0.0,
        trades_executed: int = 0,
    ) -> dict[str, float | int | str | None]:
        values_array = np.array(values)
        first_value = values_array[0]
        last_value = values_array[-1]
        first_day = datetime.strptime(days_of_investment[0], "%Y-%m-%d")
        last_day = datetime.strptime(days_of_investment[-1], "%Y-%m-%d")
        time_in_days = (last_day - first_day).days
        time_in_years = time_in_days / self.days_in_year

        daily_returns = np.diff(values_array) / values_array[:-1]
        total_return = ((last_value - first_value) / first_value) * 100

        if time_in_years > 0:
            cagr = (((last_value / first_value) ** (1 / time_in_years)) - 1) * 100
        else:
            cagr = 0

        daily_volatility = np.std(daily_returns)
        annual_volatility = daily_volatility * np.sqrt(self.days_in_year) * 100

        cumulative_max = np.maximum.accumulate(values_array)
        drawdowns = (values_array - cumulative_max) / cumulative_max * 100
        max_drawdown = np.min(drawdowns)

        rising_days = np.sum(daily_returns > 0)
        falling_days = np.sum(daily_returns < 0)
        flat_days = np.sum(daily_returns == 0)

        daily_returns_pct = daily_returns * 100
        avg_daily_return = np.mean(daily_returns_pct)

        positive_returns = daily_returns_pct[daily_returns_pct > 0]
        avg_daily_gain = np.mean(positive_returns) if len(positive_returns) > 0 else 0

        negative_returns = daily_returns_pct[daily_returns_pct < 0]
        avg_daily_loss = np.mean(negative_returns) if len(negative_returns) > 0 else 0

        best_day = np.max(daily_returns_pct) if len(daily_returns_pct) > 0 else 0
        worst_day = np.min(daily_returns_pct) if len(daily_returns_pct) > 0 else 0

        sharpe_ratio = (
            (cagr - self.risk_free_rate) / annual_volatility if annual_volatility != 0 else 0
        )

        excess_returns = daily_returns - (self.risk_free_rate / 100 / self.days_in_year)
        downside_returns = excess_returns[excess_returns < 0]
        downside_volatility = (
            np.std(downside_returns) * np.sqrt(self.days_in_year) * 100
            if len(downside_returns) > 0
            else 0
        )
        sortino_ratio = (
            (cagr - self.risk_free_rate) / downside_volatility
            if downside_volatility != 0
            else 0
        )

        calmar_ratio = cagr / abs(max_drawdown) if max_drawdown != 0 else 0

        max_consecutive_gains = self._max_consecutive(daily_returns, lambda x: x > 0)
        max_consecutive_losses = self._max_consecutive(daily_returns, lambda x: x < 0)

        win_rate = (rising_days / len(daily_returns)) * 100 if len(daily_returns) > 0 else 0

        total_gains = np.sum(positive_returns)
        total_losses = abs(np.sum(negative_returns))
        profit_factor = total_gains / total_losses if total_losses != 0 else 0

        max_dd_idx = np.argmin(drawdowns)
        recovery_time = None
        if max_dd_idx < len(values_array) - 1:
            peak_value = cumulative_max[max_dd_idx]
            recovery_idx = np.where(values_array[max_dd_idx:] >= peak_value)[0]
            if len(recovery_idx) > 0:
                recovery_time = int(recovery_idx[0])

        model = PortfolioStatistics(
            total_return_pct=round(total_return, 2),
            cagr_pct=round(cagr, 2),
            annual_volatility_pct=round(annual_volatility, 2),
            max_drawdown_pct=round(max_drawdown, 2),
            rising_days=int(rising_days),
            falling_days=int(falling_days),
            flat_days=int(flat_days),
            avg_daily_return_pct=round(avg_daily_return, 3),
            avg_daily_gain_pct=round(avg_daily_gain, 3),
            avg_daily_loss_pct=round(avg_daily_loss, 3),
            best_day_pct=round(best_day, 2),
            worst_day_pct=round(worst_day, 2),
            sharpe_ratio=round(sharpe_ratio, 3),
            sortino_ratio=round(sortino_ratio, 3),
            calmar_ratio=round(calmar_ratio, 3),
            win_rate=round(win_rate, 2),
            profit_factor=round(profit_factor, 3),
            max_consecutive_gains=max_consecutive_gains,
            max_consecutive_losses=max_consecutive_losses,
            recovery_time_days=recovery_time,
            transaction_costs_paid=round(transaction_costs_paid, 2),
            trades_executed=int(trades_executed),
            total_days=time_in_days,
            total_years=round(time_in_years, 2),
            initial_value=round(first_value, 2),
            final_value=round(last_value, 2),
        )

        return model.to_dict()
