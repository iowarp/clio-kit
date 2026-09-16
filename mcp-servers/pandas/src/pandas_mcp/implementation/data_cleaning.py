"""
Data cleaning capabilities for handling missing data and outliers.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import Optional, List
import traceback


def handle_missing_data(
    file_path: str,
    strategy: str = "detect",
    method: Optional[str] = None,
    columns: Optional[List[str]] = None,
) -> dict:
    """
    Handle missing data in various ways.

    Args:
        file_path: Path to the data file
        strategy: Strategy to handle missing data (detect, remove, impute)
        method: Imputation method; interpolate fills interior numeric gaps linearly by row.
        columns: Specific columns to process

    Returns:
        Dictionary with missing data handling results
    """
    try:
        if not os.path.exists(file_path):
            return {
                "success": False,
                "error": f"File not found: {file_path}",
                "error_type": "FileNotFoundError",
            }

        # Load data
        df = pd.read_csv(file_path)
        original_shape = df.shape

        if columns:
            df[columns].copy()
        else:
            df.copy()

        # Detect missing data
        missing_info = {
            "total_missing": int(df.isnull().sum().sum()),
            "missing_by_column": df.isnull().sum().to_dict(),
            "missing_percentage": (df.isnull().sum() / len(df) * 100).to_dict(),
            "rows_with_missing": int(df.isnull().any(axis=1).sum()),
            "complete_rows": int(df.dropna().shape[0]),
        }

        if strategy == "detect":
            return {
                "success": True,
                "file_path": file_path,
                "original_shape": original_shape,
                "missing_data_info": missing_info,
                "message": f"Found {missing_info['total_missing']} missing values",
            }

        elif strategy == "remove":
            # Remove rows with missing data
            df_cleaned = df.dropna(subset=columns)
            removed_rows = len(df) - len(df_cleaned)

            # Save cleaned data
            output_path = str(
                Path(file_path).with_name(f"{Path(file_path).stem}_no_missing.csv")
            )
            df_cleaned.to_csv(output_path, index=False)

            return {
                "success": True,
                "file_path": file_path,
                "output_file": output_path,
                "original_shape": original_shape,
                "new_shape": df_cleaned.shape,
                "removed_rows": removed_rows,
                "missing_data_info": missing_info,
                "message": f"Removed {removed_rows} rows with missing data",
            }

        elif strategy == "impute":
            if method is None:
                method = "mean"

            supported = {
                "mean",
                "median",
                "mode",
                "forward_fill",
                "backward_fill",
                "interpolate",
            }
            if method not in supported:
                raise ValueError(f"Unknown imputation method: {method}")
            df_imputed = df.copy()
            imputation_info: dict[str, dict[str, str | float | int]] = {}
            for col in columns if columns is not None else df_imputed.columns:
                series = df_imputed[col]
                missing_before = int(series.isna().sum())
                if not missing_before:
                    continue
                numeric = pd.api.types.is_numeric_dtype(series)
                fill_value: str | float = "not applied (no observed value)"
                if method == "forward_fill":
                    df_imputed[col] = series.ffill()
                    fill_value = "forward_fill"
                elif method == "backward_fill":
                    df_imputed[col] = series.bfill()
                    fill_value = "backward_fill"
                elif method == "interpolate" and numeric:
                    df_imputed[col] = series.interpolate(
                        method="linear", limit_area="inside"
                    )
                    fill_value = "linear interpolation (interior gaps only)"
                elif method == "mode":
                    modes = series.mode()
                    if not modes.empty:
                        value = modes.iloc[0]
                        df_imputed[col] = series.fillna(value)
                        fill_value = float(value) if numeric else str(value)
                elif method in {"mean", "median"} and numeric:
                    value = series.mean() if method == "mean" else series.median()
                    if pd.notna(value):
                        df_imputed[col] = series.fillna(value)
                        fill_value = float(value)
                elif not numeric:
                    fill_value = "not applied (non-numeric column)"
                imputation_info[col] = {
                    "method": method,
                    "fill_value": fill_value,
                    "imputed_count": missing_before - int(df_imputed[col].isna().sum()),
                }

            # Save imputed data
            output_path = str(
                Path(file_path).with_name(f"{Path(file_path).stem}_imputed.csv")
            )
            df_imputed.to_csv(output_path, index=False)

            return {
                "success": True,
                "file_path": file_path,
                "output_file": output_path,
                "original_shape": original_shape,
                "imputation_method": method,
                "imputation_info": imputation_info,
                "missing_data_info": missing_info,
                "message": f"Imputed missing values using {method} method",
            }

        else:
            return {
                "success": False,
                "error": f"Unknown strategy: {strategy}",
                "error_type": "ValueError",
            }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": traceback.format_exc(),
        }


def clean_data(
    file_path: str,
    remove_duplicates: bool = False,
    detect_outliers: bool = False,
    convert_types: bool = False,
) -> dict:
    """
    Clean data by removing duplicates, detecting outliers, and converting types.

    Args:
        file_path: Path to the data file
        remove_duplicates: Whether to remove duplicate rows
        detect_outliers: Whether to detect outliers
        convert_types: Whether to optimize data types

    Returns:
        Dictionary with data cleaning results
    """
    try:
        if not os.path.exists(file_path):
            return {
                "success": False,
                "error": f"File not found: {file_path}",
                "error_type": "FileNotFoundError",
            }

        # Load data
        df = pd.read_csv(file_path)
        original_shape = df.shape
        original_memory = df.memory_usage(deep=True).sum()

        cleaning_results = {
            "original_shape": original_shape,
            "original_memory_mb": round(original_memory / (1024 * 1024), 2),
        }

        # Remove duplicates
        if remove_duplicates:
            duplicates_count = df.duplicated().sum()
            df = df.drop_duplicates()
            cleaning_results["duplicates_removed"] = int(duplicates_count)

        # Detect outliers
        outliers_info = {}
        if detect_outliers:
            numeric_cols = df.select_dtypes(include=[np.number]).columns

            for col in numeric_cols:
                # Using IQR method
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
                outliers_info[col] = {
                    "outlier_count": len(outliers),
                    "outlier_percentage": round(len(outliers) / len(df) * 100, 2),
                    "lower_bound": float(lower_bound),
                    "upper_bound": float(upper_bound),
                    "outlier_values": outliers.tolist()[:10],  # First 10 outliers
                }

        # Convert types
        type_changes = {}
        if convert_types:
            for col in df.columns:
                df[col].dtype

                if df[col].dtype == "object":
                    # Try to convert to numeric
                    try:
                        numeric_series = pd.to_numeric(df[col], errors="coerce")
                        if not numeric_series.isnull().all():
                            df[col] = numeric_series
                            type_changes[col] = f"object -> {numeric_series.dtype}"
                            continue
                    except (ValueError, TypeError):
                        pass

                    # Try to convert to datetime
                    try:
                        datetime_series = pd.to_datetime(df[col], errors="coerce")
                        if not datetime_series.isnull().all():
                            df[col] = datetime_series
                            type_changes[col] = "object -> datetime64[ns]"
                            continue
                    except (ValueError, TypeError):
                        pass

                    # Convert to category if it has repeated values
                    unique_ratio = df[col].nunique() / len(df[col])
                    if unique_ratio < 0.5:
                        df[col] = df[col].astype("category")
                        type_changes[col] = "object -> category"

        # Final stats
        final_shape = df.shape
        final_memory = df.memory_usage(deep=True).sum()

        cleaning_results.update(
            {
                "final_shape": final_shape,
                "final_memory_mb": round(final_memory / (1024 * 1024), 2),
                "memory_reduction_mb": round(
                    (original_memory - final_memory) / (1024 * 1024), 2
                ),
                "outliers_info": outliers_info if detect_outliers else None,
                "type_changes": type_changes if convert_types else None,
            }
        )

        # Save cleaned data
        output_path = file_path.replace(".csv", "_cleaned.csv")
        df.to_csv(output_path, index=False)

        return {
            "success": True,
            "file_path": file_path,
            "output_file": output_path,
            "cleaning_results": cleaning_results,
            "message": f"Data cleaned successfully. Shape: {original_shape} -> {final_shape}",
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": traceback.format_exc(),
        }
