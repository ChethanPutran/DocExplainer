from typing import List, Dict, Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR


class LearningPatternAnalyzer:
    """Analyze user learning patterns using time-series data."""

    def __init__(self) -> None:
        self.regression_model = LinearRegression()
        self.svr_model = SVR(kernel="rbf")

    def analyze_learning_trend(
        self,
        user_history: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Analyze the user's learning trend over time.

        Expected input format:

        [
            {
                "timestamp": "2026-09-01T10:00:00",
                "knowledge_gain": 0.5,
            },
            ...
        ]
        """

        if not user_history:
            return {
                "learning_rate": 0.0,
                "trend_direction": "stable",
                "consistency_score": 1.0,
                "predictions": [],
            }

        df = pd.DataFrame(user_history)

        required_columns = {
            "timestamp",
            "knowledge_gain",
        }

        missing_columns = required_columns - set(df.columns)

        if missing_columns:
            raise ValueError(
                f"Missing required fields: "
                f"{sorted(missing_columns)}"
            )

        # ----------------------------------------------------------
        # Prepare time series
        # ----------------------------------------------------------

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            errors="coerce",
        )

        if df["timestamp"].isna().any():
            raise ValueError(
                "Invalid timestamp found in user history."
            )

        df["knowledge_gain"] = pd.to_numeric(
            df["knowledge_gain"],
            errors="coerce",
        )

        if df["knowledge_gain"].isna().any():
            raise ValueError(
                "Invalid knowledge_gain found in user history."
            )

        df = df.sort_values(
            "timestamp"
        ).reset_index(drop=True)

        # ----------------------------------------------------------
        # Calculate cumulative knowledge
        # ----------------------------------------------------------

        df["cumulative_knowledge"] = (
            df["knowledge_gain"].cumsum()
        )

        # ----------------------------------------------------------
        # Handle a single observation
        # ----------------------------------------------------------

        if len(df) == 1:
            learning_rate = float(
                df["knowledge_gain"].iloc[0]
            )

            return {
                "learning_rate": learning_rate,
                "trend_direction": self._get_trend_direction(
                    learning_rate
                ),
                "consistency_score": 1.0,
                "predictions": [
                    float(df["cumulative_knowledge"].iloc[0])
                ],
            }

        # ----------------------------------------------------------
        # Linear regression
        # ----------------------------------------------------------

        X = np.arange(
            len(df),
            dtype=float,
        ).reshape(-1, 1)

        y = df[
            "cumulative_knowledge"
        ].to_numpy(dtype=float)

        self.regression_model.fit(X, y)

        learning_rate = float(
            self.regression_model.coef_[0]
        )

        # ----------------------------------------------------------
        # Future predictions
        # ----------------------------------------------------------

        future_steps = 10

        future_X = np.arange(
            len(df),
            len(df) + future_steps,
            dtype=float,
        ).reshape(-1, 1)

        future_predictions = (
            self.regression_model.predict(
                future_X
            )
        )

        # ----------------------------------------------------------
        # Consistency
        # ----------------------------------------------------------

        consistency_score = (
            self._calculate_consistency(
                df["knowledge_gain"].to_numpy(
                    dtype=float
                )
            )
        )

        return {
            "learning_rate": learning_rate,
            "trend_direction": self._get_trend_direction(
                learning_rate
            ),
            "consistency_score": consistency_score,
            "predictions": (
                future_predictions
                .tolist()
            ),
        }

    @staticmethod
    def _get_trend_direction(
        learning_rate: float,
        threshold: float = 1e-6,
    ) -> str:
        """Convert learning rate into a qualitative trend."""

        if learning_rate > threshold:
            return "increasing"

        if learning_rate < -threshold:
            return "decreasing"

        return "stable"

    @staticmethod
    def _calculate_consistency(
        knowledge_gains: np.ndarray,
    ) -> float:
        """
        Calculate learning consistency.

        Lower variance means more consistent learning.
        """

        if len(knowledge_gains) < 2:
            return 1.0

        variance = float(
            np.var(knowledge_gains)
        )

        return float(
            1.0 / (1.0 + variance)
        )

    def detect_learning_plateaus(
        self,
        performance_history: List[float],
    ) -> List[Dict[str, Any]]:
        """
        Detect periods where performance remains nearly constant.

        A plateau is detected when the standard deviation of a
        rolling window is below the configured threshold.
        """

        if not performance_history:
            return []

        window_size = 5
        threshold = 0.05

        if len(performance_history) < window_size:
            return []

        performance = np.asarray(
            performance_history,
            dtype=float,
        )

        if not np.isfinite(performance).all():
            raise ValueError(
                "performance_history contains "
                "invalid numeric values."
            )

        plateaus: List[Dict[str, Any]] = []

        for i in range(
            len(performance) - window_size + 1
        ):
            window = performance[
                i : i + window_size
            ]

            if np.std(window) < threshold:
                plateaus.append(
                    {
                        "start_index": i,
                        "end_index": (
                            i + window_size - 1
                        ),
                        "duration": window_size,
                        "avg_performance": float(
                            np.mean(window)
                        ),
                    }
                )

        return plateaus
