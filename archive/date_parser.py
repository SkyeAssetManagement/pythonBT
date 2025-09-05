import pandas as pd
from datetime import datetime
import re

class FlexibleDateParser:
    """
    Flexible date/time parser that can handle multiple date formats including:
    - Separate Date and Time columns (e.g., "2004-01-02" and "09:55")
    - Combined Date/Time column (e.g., "2/01/2004 15:59")
    - Various date formats (YYYY-MM-DD, DD/MM/YYYY, MM/DD/YYYY, etc.)
    """
    
    @staticmethod
    def detect_date_format(date_str):
        """
        Detect the date format from a sample date string
        Returns format string for pd.to_datetime
        """
        date_str = str(date_str).strip()
        
        # Common date formats to try
        formats = [
            # ISO formats
            ('%Y-%m-%d', r'^\d{4}-\d{1,2}-\d{1,2}$'),
            ('%Y/%m/%d', r'^\d{4}/\d{1,2}/\d{1,2}$'),
            
            # European formats (day first)
            ('%d/%m/%Y', r'^\d{1,2}/\d{1,2}/\d{4}$'),
            ('%d-%m-%Y', r'^\d{1,2}-\d{1,2}-\d{4}$'),
            ('%d.%m.%Y', r'^\d{1,2}\.\d{1,2}\.\d{4}$'),
            
            # American formats (month first)
            ('%m/%d/%Y', r'^\d{1,2}/\d{1,2}/\d{4}$'),
            ('%m-%d-%Y', r'^\d{1,2}-\d{1,2}-\d{4}$'),
            
            # Two-digit year formats
            ('%d/%m/%y', r'^\d{1,2}/\d{1,2}/\d{2}$'),
            ('%m/%d/%y', r'^\d{1,2}/\d{1,2}/\d{2}$'),
            ('%y-%m-%d', r'^\d{2}-\d{1,2}-\d{1,2}$'),
        ]
        
        # Check if it includes time
        if ' ' in date_str:
            date_part, time_part = date_str.split(' ', 1)
            
            # Combined datetime formats
            datetime_formats = [
                ('%d/%m/%Y %H:%M', r'^\d{1,2}/\d{1,2}/\d{4} \d{1,2}:\d{2}$'),
                ('%m/%d/%Y %H:%M', r'^\d{1,2}/\d{1,2}/\d{4} \d{1,2}:\d{2}$'),
                ('%Y-%m-%d %H:%M', r'^\d{4}-\d{1,2}-\d{1,2} \d{1,2}:\d{2}$'),
                ('%d/%m/%Y %H:%M:%S', r'^\d{1,2}/\d{1,2}/\d{4} \d{1,2}:\d{2}:\d{2}$'),
                ('%m/%d/%Y %H:%M:%S', r'^\d{1,2}/\d{1,2}/\d{4} \d{1,2}:\d{2}:\d{2}$'),
                ('%Y-%m-%d %H:%M:%S', r'^\d{4}-\d{1,2}-\d{1,2} \d{1,2}:\d{2}:\d{2}$'),
            ]
            
            for fmt, pattern in datetime_formats:
                if re.match(pattern, date_str):
                    try:
                        datetime.strptime(date_str, fmt)
                        return fmt
                    except ValueError:
                        continue
        
        # Try date-only formats
        for fmt, pattern in formats:
            if re.match(pattern, date_str):
                try:
                    datetime.strptime(date_str, fmt)
                    return fmt
                except ValueError:
                    continue
        
        return None
    
    @staticmethod
    def parse_dates(df, date_column=None, time_column=None, datetime_column=None, dayfirst=None):
        """
        Parse dates from a DataFrame with flexible format detection
        
        Parameters:
        -----------
        df : pandas.DataFrame
            The DataFrame containing date/time information
        date_column : str, optional
            Name of the date column (if dates and times are separate)
        time_column : str, optional
            Name of the time column (if dates and times are separate)
        datetime_column : str, optional
            Name of the combined datetime column
        dayfirst : bool, optional
            If True, interpret the first value as day (DD/MM format)
            If None, will try to auto-detect
        
        Returns:
        --------
        pandas.Series
            A Series of parsed datetime objects
        """
        
        # Case 1: Combined datetime column
        if datetime_column and datetime_column in df.columns:
            sample = df[datetime_column].dropna().iloc[0] if not df[datetime_column].dropna().empty else None
            
            if sample:
                # Try to detect format from sample
                fmt = FlexibleDateParser.detect_date_format(sample)
                
                if fmt:
                    try:
                        return pd.to_datetime(df[datetime_column], format=fmt)
                    except:
                        pass
                
                # Fall back to pandas inference
                if dayfirst is not None:
                    return pd.to_datetime(df[datetime_column], dayfirst=dayfirst)
                else:
                    # Try to infer from data
                    return FlexibleDateParser._infer_datetime(df[datetime_column])
        
        # Case 2: Separate date and time columns
        if date_column and date_column in df.columns:
            if time_column and time_column in df.columns:
                # Combine date and time
                datetime_str = df[date_column].astype(str) + ' ' + df[time_column].astype(str)
                
                sample = datetime_str.dropna().iloc[0] if not datetime_str.dropna().empty else None
                if sample:
                    fmt = FlexibleDateParser.detect_date_format(sample)
                    if fmt:
                        try:
                            return pd.to_datetime(datetime_str, format=fmt)
                        except:
                            pass
                
                # Fall back to pandas inference
                if dayfirst is not None:
                    return pd.to_datetime(datetime_str, dayfirst=dayfirst)
                else:
                    return FlexibleDateParser._infer_datetime(datetime_str)
            else:
                # Date only
                sample = df[date_column].dropna().iloc[0] if not df[date_column].dropna().empty else None
                if sample:
                    fmt = FlexibleDateParser.detect_date_format(sample)
                    if fmt:
                        try:
                            return pd.to_datetime(df[date_column], format=fmt)
                        except:
                            pass
                
                # Fall back to pandas inference
                if dayfirst is not None:
                    return pd.to_datetime(df[date_column], dayfirst=dayfirst)
                else:
                    return FlexibleDateParser._infer_datetime(df[date_column])
        
        # Case 3: Try to find datetime column automatically
        datetime_columns = []
        for col in df.columns:
            col_lower = col.lower()
            if any(term in col_lower for term in ['date', 'time', 'datetime', 'timestamp']):
                datetime_columns.append(col)
        
        if len(datetime_columns) == 1:
            # Single datetime column found
            return FlexibleDateParser.parse_dates(df, datetime_column=datetime_columns[0], dayfirst=dayfirst)
        elif len(datetime_columns) == 2:
            # Possibly separate date and time columns
            date_col = None
            time_col = None
            for col in datetime_columns:
                col_lower = col.lower()
                if 'time' in col_lower and 'date' not in col_lower:
                    time_col = col
                else:
                    date_col = col
            
            if date_col:
                return FlexibleDateParser.parse_dates(df, date_column=date_col, time_column=time_col, dayfirst=dayfirst)
        
        raise ValueError("Could not identify date/time columns in DataFrame")
    
    @staticmethod
    def _infer_datetime(series):
        """
        Try to intelligently infer datetime format
        """
        # First try with dayfirst=False (American format)
        try:
            result = pd.to_datetime(series, dayfirst=False, errors='coerce')
            if result.notna().sum() / len(result) > 0.9:  # If >90% parsed successfully
                return result
        except:
            pass
        
        # Then try with dayfirst=True (European format)
        try:
            result = pd.to_datetime(series, dayfirst=True, errors='coerce')
            if result.notna().sum() / len(result) > 0.9:  # If >90% parsed successfully
                return result
        except:
            pass
        
        # Fall back to pandas default
        return pd.to_datetime(series, errors='coerce')
    
    @staticmethod
    def get_date_columns(df):
        """
        Identify potential date/time columns in a DataFrame
        
        Returns:
        --------
        dict
            Dictionary with 'date_column', 'time_column', and 'datetime_column' keys
        """
        result = {
            'date_column': None,
            'time_column': None,
            'datetime_column': None
        }
        
        for col in df.columns:
            col_lower = col.lower()
            
            # Check for combined datetime column
            if any(term in col_lower for term in ['datetime', 'date/time', 'date_time', 'timestamp']):
                result['datetime_column'] = col
            # Check for date column
            elif 'date' in col_lower and 'time' not in col_lower:
                result['date_column'] = col
            # Check for time column
            elif 'time' in col_lower and 'date' not in col_lower:
                result['time_column'] = col
        
        return result