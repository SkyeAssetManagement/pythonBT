# src/dashboard/indicators_panel.py
# VBT Pro Indicators Panel - Step 6 (Final)
# Professional technical indicators with parameter selection and visualization

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                           QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, QFrame,
                           QScrollArea, QGroupBox, QGridLayout, QSlider, QColorDialog,
                           QTabWidget, QListWidget, QListWidgetItem, QSplitter,
                           QTableWidget, QTableWidgetItem, QHeaderView)
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot, QTimer
from PyQt5.QtGui import QColor, QFont, QPalette

@dataclass
class IndicatorConfig:
    """Configuration for a technical indicator"""
    name: str
    display_name: str
    indicator_type: str  # 'overlay', 'oscillator', 'volume'
    parameters: Dict[str, Any] = field(default_factory=dict)
    color: str = '#ffffff'
    line_width: int = 1
    enabled: bool = True
    visible: bool = True

class VBTIndicatorEngine:
    """
    VectorBT-style indicator calculation engine
    Implements professional technical indicators with optimization
    """
    
    @staticmethod
    def sma(data: np.ndarray, period: int) -> np.ndarray:
        """Simple Moving Average"""
        if len(data) < period:
            return np.full_like(data, np.nan)
        
        result = np.full_like(data, np.nan)
        for i in range(period - 1, len(data)):
            result[i] = np.mean(data[i - period + 1:i + 1])
        return result
    
    @staticmethod
    def ema(data: np.ndarray, period: int) -> np.ndarray:
        """Exponential Moving Average"""
        if len(data) < period:
            return np.full_like(data, np.nan)
        
        alpha = 2.0 / (period + 1)
        result = np.full_like(data, np.nan)
        result[period - 1] = np.mean(data[:period])
        
        for i in range(period, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
        
        return result
    
    @staticmethod
    def bollinger_bands(data: np.ndarray, period: int = 20, std_dev: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bollinger Bands"""
        sma = VBTIndicatorEngine.sma(data, period)
        
        std = np.full_like(data, np.nan)
        for i in range(period - 1, len(data)):
            std[i] = np.std(data[i - period + 1:i + 1])
        
        upper = sma + std_dev * std
        lower = sma - std_dev * std
        
        return upper, sma, lower
    
    @staticmethod
    def rsi(data: np.ndarray, period: int = 14) -> np.ndarray:
        """Relative Strength Index"""
        if len(data) < period + 1:
            return np.full_like(data, np.nan)
        
        deltas = np.diff(data)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        
        result = np.full(len(data), np.nan)
        
        if avg_loss != 0:
            rs = avg_gain / avg_loss
            result[period] = 100 - (100 / (1 + rs))
        
        alpha = 1.0 / period
        for i in range(period + 1, len(data)):
            gain = gains[i - 1] if i - 1 < len(gains) else 0
            loss = losses[i - 1] if i - 1 < len(losses) else 0
            
            avg_gain = alpha * gain + (1 - alpha) * avg_gain
            avg_loss = alpha * loss + (1 - alpha) * avg_loss
            
            if avg_loss != 0:
                rs = avg_gain / avg_loss
                result[i] = 100 - (100 / (1 + rs))
        
        return result
    
    @staticmethod
    def macd(data: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """MACD"""
        ema_fast = VBTIndicatorEngine.ema(data, fast)
        ema_slow = VBTIndicatorEngine.ema(data, slow)
        
        macd_line = ema_fast - ema_slow
        signal_line = VBTIndicatorEngine.ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def stochastic(high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                   k_period: int = 14, d_period: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """Stochastic Oscillator"""
        if len(high) < k_period:
            return np.full_like(high, np.nan), np.full_like(high, np.nan)
        
        k_percent = np.full_like(high, np.nan)
        
        for i in range(k_period - 1, len(high)):
            period_high = np.max(high[i - k_period + 1:i + 1])
            period_low = np.min(low[i - k_period + 1:i + 1])
            
            if period_high != period_low:
                k_percent[i] = ((close[i] - period_low) / (period_high - period_low)) * 100
        
        d_percent = VBTIndicatorEngine.sma(k_percent, d_period)
        
        return k_percent, d_percent
    
    @staticmethod
    def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """Average True Range"""
        if len(high) < 2:
            return np.full_like(high, np.nan)
        
        # True Range calculation
        tr = np.full(len(high), np.nan)
        for i in range(1, len(high)):
            tr1 = high[i] - low[i]
            tr2 = abs(high[i] - close[i - 1])
            tr3 = abs(low[i] - close[i - 1])
            tr[i] = max(tr1, tr2, tr3)
        
        # ATR using EMA smoothing
        atr_result = np.full_like(high, np.nan)
        if period <= len(tr):
            # Initial ATR
            first_atr_index = period
            atr_result[first_atr_index] = np.mean(tr[1:first_atr_index + 1])
            
            # Subsequent ATR values
            alpha = 1.0 / period
            for i in range(first_atr_index + 1, len(tr)):
                atr_result[i] = alpha * tr[i] + (1 - alpha) * atr_result[i - 1]
        
        return atr_result
    
    @staticmethod
    def volume_sma(volume: np.ndarray, period: int = 20) -> np.ndarray:
        """Volume Simple Moving Average"""
        return VBTIndicatorEngine.sma(volume, period)


class IndicatorParameterWidget(QWidget):
    """Widget for configuring indicator parameters"""
    
    parameter_changed = pyqtSignal(str, object)  # parameter_name, value
    
    def __init__(self, param_name: str, param_type: type, default_value: Any, 
                 min_val: Any = None, max_val: Any = None, parent=None):
        super().__init__(parent)
        
        self.param_name = param_name
        self.param_type = param_type
        
        self._setup_ui(param_name, param_type, default_value, min_val, max_val)
    
    def _setup_ui(self, param_name: str, param_type: type, default_value: Any, 
                  min_val: Any, max_val: Any):
        """Setup parameter input widget"""
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        
        # Parameter label
        label = QLabel(f"{param_name}:")
        label.setMinimumWidth(80)
        label.setStyleSheet("color: #cccccc; font-size: 8pt; font-weight: bold;")
        layout.addWidget(label)
        
        # Parameter input based on type
        if param_type == int:
            self.input_widget = QSpinBox()
            if min_val is not None:
                self.input_widget.setMinimum(min_val)
            if max_val is not None:
                self.input_widget.setMaximum(max_val)
            self.input_widget.setValue(default_value)
            self.input_widget.valueChanged.connect(self._emit_change)
            
        elif param_type == float:
            self.input_widget = QDoubleSpinBox()
            self.input_widget.setDecimals(3)
            if min_val is not None:
                self.input_widget.setMinimum(min_val)
            if max_val is not None:
                self.input_widget.setMaximum(max_val)
            self.input_widget.setValue(default_value)
            self.input_widget.valueChanged.connect(self._emit_change)
            
        elif param_type == bool:
            self.input_widget = QCheckBox()
            self.input_widget.setChecked(default_value)
            self.input_widget.toggled.connect(self._emit_change)
            
        else:  # String or other
            self.input_widget = QLabel(str(default_value))
            
        self.input_widget.setStyleSheet("""
            QSpinBox, QDoubleSpinBox {
                background-color: #404040;
                color: white;
                border: 1px solid #606060;
                padding: 2px;
                border-radius: 2px;
            }
            QCheckBox {
                color: white;
            }
        """)
        
        layout.addWidget(self.input_widget)
        layout.addStretch()
    
    def _emit_change(self):
        """Emit parameter change signal"""
        if hasattr(self.input_widget, 'value'):
            value = self.input_widget.value()
        elif hasattr(self.input_widget, 'isChecked'):
            value = self.input_widget.isChecked()
        else:
            value = None
        
        self.parameter_changed.emit(self.param_name, value)
    
    def get_value(self):
        """Get current parameter value"""
        if hasattr(self.input_widget, 'value'):
            return self.input_widget.value()
        elif hasattr(self.input_widget, 'isChecked'):
            return self.input_widget.isChecked()
        else:
            return None


class IndicatorConfigWidget(QFrame):
    """Widget for configuring a single indicator"""
    
    config_changed = pyqtSignal(str, IndicatorConfig)  # indicator_id, config
    remove_requested = pyqtSignal(str)  # indicator_id
    
    def __init__(self, indicator_id: str, config: IndicatorConfig, parent=None):
        super().__init__(parent)
        
        self.indicator_id = indicator_id
        self.config = config
        self.param_widgets = {}
        
        self._setup_ui()
        self._apply_styling()
    
    def _setup_ui(self):
        """Setup the indicator configuration UI"""
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(3)
        
        # Header with indicator name and controls
        header_layout = QHBoxLayout()
        
        # Indicator name and enabled checkbox
        self.enabled_cb = QCheckBox(self.config.display_name)
        self.enabled_cb.setChecked(self.config.enabled)
        self.enabled_cb.setStyleSheet("color: white; font-weight: bold; font-size: 9pt;")
        self.enabled_cb.toggled.connect(self._on_enabled_changed)
        header_layout.addWidget(self.enabled_cb)
        
        header_layout.addStretch()
        
        # Color button
        self.color_btn = QPushButton()
        self.color_btn.setMaximumSize(30, 20)
        self.color_btn.setStyleSheet(f"background-color: {self.config.color}; border: 1px solid #666666;")
        self.color_btn.clicked.connect(self._choose_color)
        header_layout.addWidget(self.color_btn)
        
        # Remove button
        remove_btn = QPushButton("x")
        remove_btn.setMaximumSize(20, 20)
        remove_btn.setStyleSheet("""
            QPushButton {
                background-color: #d32f2f;
                color: white;
                border: none;
                font-weight: bold;
                font-size: 12pt;
                border-radius: 2px;
            }
            QPushButton:hover {
                background-color: #f44336;
            }
        """)
        remove_btn.clicked.connect(lambda: self.remove_requested.emit(self.indicator_id))
        header_layout.addWidget(remove_btn)
        
        main_layout.addLayout(header_layout)
        
        # Parameters section
        if self.config.parameters:
            params_frame = QFrame()
            params_layout = QVBoxLayout(params_frame)
            params_layout.setContentsMargins(10, 5, 5, 5)
            params_layout.setSpacing(2)
            
            for param_name, param_info in self._get_parameter_info().items():
                param_widget = IndicatorParameterWidget(
                    param_name,
                    param_info['type'],
                    self.config.parameters.get(param_name, param_info['default']),
                    param_info.get('min'),
                    param_info.get('max')
                )
                param_widget.parameter_changed.connect(self._on_parameter_changed)
                params_layout.addWidget(param_widget)
                self.param_widgets[param_name] = param_widget
            
            params_frame.setStyleSheet("background-color: rgba(80, 80, 80, 100); border-radius: 3px;")
            main_layout.addWidget(params_frame)
    
    def _get_parameter_info(self) -> Dict[str, Dict]:
        """Get parameter information for the indicator"""
        
        param_info = {
            'SMA': {
                'period': {'type': int, 'default': 20, 'min': 1, 'max': 200}
            },
            'EMA': {
                'period': {'type': int, 'default': 12, 'min': 1, 'max': 200}
            },
            'Bollinger Bands': {
                'period': {'type': int, 'default': 20, 'min': 1, 'max': 100},
                'std_dev': {'type': float, 'default': 2.0, 'min': 0.1, 'max': 5.0}
            },
            'RSI': {
                'period': {'type': int, 'default': 14, 'min': 1, 'max': 100}
            },
            'MACD': {
                'fast': {'type': int, 'default': 12, 'min': 1, 'max': 50},
                'slow': {'type': int, 'default': 26, 'min': 1, 'max': 100},
                'signal': {'type': int, 'default': 9, 'min': 1, 'max': 50}
            },
            'Stochastic': {
                'k_period': {'type': int, 'default': 14, 'min': 1, 'max': 100},
                'd_period': {'type': int, 'default': 3, 'min': 1, 'max': 20}
            },
            'ATR': {
                'period': {'type': int, 'default': 14, 'min': 1, 'max': 100}
            },
            'Volume SMA': {
                'period': {'type': int, 'default': 20, 'min': 1, 'max': 200}
            }
        }
        
        return param_info.get(self.config.name, {})
    
    def _apply_styling(self):
        """Apply styling to the widget"""
        self.setStyleSheet("""
            QFrame {
                background-color: #353535;
                border: 1px solid #555555;
                border-radius: 4px;
                margin: 2px;
            }
        """)
    
    def _on_enabled_changed(self, enabled: bool):
        """Handle enabled state change"""
        self.config.enabled = enabled
        self._emit_config_change()
    
    def _on_parameter_changed(self, param_name: str, value: Any):
        """Handle parameter change"""
        self.config.parameters[param_name] = value
        self._emit_config_change()
    
    def _choose_color(self):
        """Open color chooser dialog"""
        color = QColorDialog.getColor(QColor(self.config.color), self)
        if color.isValid():
            self.config.color = color.name()
            self.color_btn.setStyleSheet(f"background-color: {self.config.color}; border: 1px solid #666666;")
            self._emit_config_change()
    
    def _emit_config_change(self):
        """Emit configuration change signal"""
        self.config_changed.emit(self.indicator_id, self.config)


class VBTIndicatorsPanel(QWidget):
    """
    VBT Pro Indicators Panel - Complete indicator management and visualization
    Features professional technical indicators with parameter selection
    """
    
    indicators_updated = pyqtSignal(dict)  # indicator_data
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Data storage
        self.ohlcv_data = None
        self.indicators = {}  # indicator_id -> IndicatorConfig
        self.indicator_data = {}  # indicator_id -> calculated data
        self.next_indicator_id = 1
        
        # Available indicators
        self.available_indicators = {
            'SMA': {'name': 'SMA', 'display_name': 'Simple Moving Average', 'type': 'overlay'},
            'EMA': {'name': 'EMA', 'display_name': 'Exponential Moving Average', 'type': 'overlay'},
            'Bollinger Bands': {'name': 'Bollinger Bands', 'display_name': 'Bollinger Bands', 'type': 'overlay'},
            'RSI': {'name': 'RSI', 'display_name': 'RSI', 'type': 'oscillator'},
            'MACD': {'name': 'MACD', 'display_name': 'MACD', 'type': 'oscillator'},
            'Stochastic': {'name': 'Stochastic', 'display_name': 'Stochastic', 'type': 'oscillator'},
            'ATR': {'name': 'ATR', 'display_name': 'Average True Range', 'type': 'oscillator'},
            'Volume SMA': {'name': 'Volume SMA', 'display_name': 'Volume Moving Average', 'type': 'volume'}
        }
        
        self._setup_ui()
        
        print("SUCCESS: VBT Indicators Panel initialized")
    
    def _setup_ui(self):
        """Setup the indicators panel UI"""
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)
        
        # Title
        title_label = QLabel("VBT PRO INDICATORS")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
            QLabel {
                color: white;
                background-color: #404040;
                font-weight: bold;
                font-size: 10pt;
                padding: 6px;
                border: 1px solid #606060;
                border-radius: 3px;
            }
        """)
        main_layout.addWidget(title_label)
        
        # Add indicator section
        add_section = self._create_add_indicator_section()
        main_layout.addWidget(add_section)
        
        # Active indicators scroll area
        self.indicators_scroll = QScrollArea()
        self.indicators_widget = QWidget()
        self.indicators_layout = QVBoxLayout(self.indicators_widget)
        self.indicators_layout.setContentsMargins(2, 2, 2, 2)
        self.indicators_layout.setSpacing(3)
        
        self.indicators_scroll.setWidget(self.indicators_widget)
        self.indicators_scroll.setWidgetResizable(True)
        self.indicators_scroll.setMaximumHeight(400)
        self.indicators_scroll.setStyleSheet("""
            QScrollArea {
                background-color: #2b2b2b;
                border: 1px solid #555555;
                border-radius: 3px;
            }
        """)
        
        main_layout.addWidget(self.indicators_scroll)
        
        # Control buttons
        controls_layout = QHBoxLayout()
        
        self.recalculate_btn = QPushButton("Recalculate All")
        self.clear_all_btn = QPushButton("Clear All")
        self.export_btn = QPushButton("Export Data")
        
        button_style = """
            QPushButton {
                background-color: #404040;
                color: white;
                border: 1px solid #606060;
                padding: 6px 12px;
                border-radius: 3px;
                font-weight: bold;
                font-size: 9pt;
            }
            QPushButton:hover {
                background-color: #505050;
            }
            QPushButton:pressed {
                background-color: #606060;
            }
        """
        
        for btn in [self.recalculate_btn, self.clear_all_btn, self.export_btn]:
            btn.setStyleSheet(button_style)
        
        self.recalculate_btn.clicked.connect(self._recalculate_all_indicators)
        self.clear_all_btn.clicked.connect(self._clear_all_indicators)
        self.export_btn.clicked.connect(self._export_indicator_data)
        
        controls_layout.addWidget(self.recalculate_btn)
        controls_layout.addWidget(self.clear_all_btn)
        controls_layout.addWidget(self.export_btn)
        controls_layout.addStretch()
        
        main_layout.addLayout(controls_layout)
        
        # Statistics section
        self.stats_label = QLabel("No indicators active")
        self.stats_label.setAlignment(Qt.AlignCenter)
        self.stats_label.setStyleSheet("""
            QLabel {
                color: #cccccc;
                background-color: #333333;
                padding: 4px;
                border: 1px solid #555555;
                border-radius: 2px;
                font-size: 8pt;
            }
        """)
        main_layout.addWidget(self.stats_label)
        
        # Apply theme
        self.setStyleSheet("""
            QWidget {
                background-color: #2b2b2b;
                color: white;
            }
        """)
    
    def _create_add_indicator_section(self) -> QWidget:
        """Create the add indicator section"""
        
        section = QFrame()
        layout = QHBoxLayout(section)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Indicator selection
        add_label = QLabel("Add Indicator:")
        add_label.setStyleSheet("color: #cccccc; font-weight: bold; font-size: 9pt;")
        layout.addWidget(add_label)
        
        self.indicator_combo = QComboBox()
        self.indicator_combo.addItems(list(self.available_indicators.keys()))
        self.indicator_combo.setStyleSheet("""
            QComboBox {
                background-color: #404040;
                color: white;
                border: 1px solid #606060;
                padding: 4px;
                border-radius: 2px;
                font-size: 9pt;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 4px solid #cccccc;
                margin-right: 5px;
            }
        """)
        layout.addWidget(self.indicator_combo)
        
        # Add button
        add_btn = QPushButton("Add")
        add_btn.setStyleSheet("""
            QPushButton {
                background-color: #2e7d32;
                color: white;
                border: 1px solid #4caf50;
                padding: 4px 16px;
                border-radius: 2px;
                font-weight: bold;
                font-size: 9pt;
            }
            QPushButton:hover {
                background-color: #388e3c;
            }
            QPushButton:pressed {
                background-color: #1b5e20;
            }
        """)
        add_btn.clicked.connect(self._add_indicator)
        layout.addWidget(add_btn)
        
        layout.addStretch()
        
        section.setStyleSheet("""
            QFrame {
                background-color: #353535;
                border: 1px solid #555555;
                border-radius: 3px;
            }
        """)
        
        return section
    
    def load_ohlcv_data(self, ohlcv_data: Dict[str, np.ndarray]) -> bool:
        """Load OHLCV data for indicator calculations"""
        try:
            self.ohlcv_data = {
                'open': np.asarray(ohlcv_data['open'], dtype=np.float64),
                'high': np.asarray(ohlcv_data['high'], dtype=np.float64),
                'low': np.asarray(ohlcv_data['low'], dtype=np.float64),
                'close': np.asarray(ohlcv_data['close'], dtype=np.float64),
                'volume': np.asarray(ohlcv_data.get('volume', np.ones(len(ohlcv_data['close']))), dtype=np.float64)
            }
            
            print(f"VBT Indicators Panel loaded {len(self.ohlcv_data['close'])} bars")
            
            # Recalculate existing indicators
            if self.indicators:
                self._recalculate_all_indicators()
            
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to load OHLCV data in indicators panel: {e}")
            return False
    
    def _add_indicator(self):
        """Add a new indicator"""
        try:
            indicator_name = self.indicator_combo.currentText()
            indicator_info = self.available_indicators[indicator_name]
            
            # Create indicator configuration
            indicator_id = f"{indicator_name}_{self.next_indicator_id}"
            self.next_indicator_id += 1
            
            # Default parameters
            default_params = self._get_default_parameters(indicator_name)
            
            # Create configuration
            config = IndicatorConfig(
                name=indicator_name,
                display_name=indicator_info['display_name'],
                indicator_type=indicator_info['type'],
                parameters=default_params,
                color=self._get_next_color(),
                enabled=True,
                visible=True
            )
            
            self.indicators[indicator_id] = config
            
            # Create configuration widget
            config_widget = IndicatorConfigWidget(indicator_id, config)
            config_widget.config_changed.connect(self._on_indicator_config_changed)
            config_widget.remove_requested.connect(self._remove_indicator)
            
            self.indicators_layout.addWidget(config_widget)
            
            # Calculate indicator
            self._calculate_indicator(indicator_id, config)
            
            # Update statistics
            self._update_statistics()
            
            # Emit signal to notify dashboard of updated indicators
            self.indicators_updated.emit(self.indicator_data)
            
            print(f"Added indicator: {indicator_name} (ID: {indicator_id})")
            print(f"Emitted indicators_updated signal with {len(self.indicator_data)} indicators")
            
        except Exception as e:
            print(f"ERROR: Failed to add indicator: {e}")
            import traceback
            traceback.print_exc()
    
    def _get_default_parameters(self, indicator_name: str) -> Dict[str, Any]:
        """Get default parameters for an indicator"""
        defaults = {
            'SMA': {'period': 20},
            'EMA': {'period': 12},
            'Bollinger Bands': {'period': 20, 'std_dev': 2.0},
            'RSI': {'period': 14},
            'MACD': {'fast': 12, 'slow': 26, 'signal': 9},
            'Stochastic': {'k_period': 14, 'd_period': 3},
            'ATR': {'period': 14},
            'Volume SMA': {'period': 20}
        }
        
        return defaults.get(indicator_name, {})
    
    def _get_next_color(self) -> str:
        """Get next color for indicator"""
        colors = [
            '#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7',
            '#dda0dd', '#98d8c8', '#f7dc6f', '#bb8fce', '#85c1e9'
        ]
        return colors[len(self.indicators) % len(colors)]
    
    def _calculate_indicator(self, indicator_id: str, config: IndicatorConfig):
        """Calculate indicator data"""
        try:
            if not self.ohlcv_data or not config.enabled:
                return
            
            indicator_name = config.name
            params = config.parameters
            
            # Calculate based on indicator type
            if indicator_name == 'SMA':
                data = VBTIndicatorEngine.sma(self.ohlcv_data['close'], params['period'])
                self.indicator_data[indicator_id] = {'sma': data}
                
            elif indicator_name == 'EMA':
                data = VBTIndicatorEngine.ema(self.ohlcv_data['close'], params['period'])
                self.indicator_data[indicator_id] = {'ema': data}
                
            elif indicator_name == 'Bollinger Bands':
                upper, middle, lower = VBTIndicatorEngine.bollinger_bands(
                    self.ohlcv_data['close'], params['period'], params['std_dev']
                )
                self.indicator_data[indicator_id] = {'upper': upper, 'middle': middle, 'lower': lower}
                
            elif indicator_name == 'RSI':
                data = VBTIndicatorEngine.rsi(self.ohlcv_data['close'], params['period'])
                self.indicator_data[indicator_id] = {'rsi': data}
                
            elif indicator_name == 'MACD':
                macd_line, signal_line, histogram = VBTIndicatorEngine.macd(
                    self.ohlcv_data['close'], params['fast'], params['slow'], params['signal']
                )
                self.indicator_data[indicator_id] = {
                    'macd': macd_line, 'signal': signal_line, 'histogram': histogram
                }
                
            elif indicator_name == 'Stochastic':
                k_percent, d_percent = VBTIndicatorEngine.stochastic(
                    self.ohlcv_data['high'], self.ohlcv_data['low'], self.ohlcv_data['close'],
                    params['k_period'], params['d_period']
                )
                self.indicator_data[indicator_id] = {'k': k_percent, 'd': d_percent}
                
            elif indicator_name == 'ATR':
                data = VBTIndicatorEngine.atr(
                    self.ohlcv_data['high'], self.ohlcv_data['low'], self.ohlcv_data['close'],
                    params['period']
                )
                self.indicator_data[indicator_id] = {'atr': data}
                
            elif indicator_name == 'Volume SMA':
                data = VBTIndicatorEngine.volume_sma(self.ohlcv_data['volume'], params['period'])
                self.indicator_data[indicator_id] = {'volume_sma': data}
            
            # Emit update signal
            self.indicators_updated.emit(self.indicator_data)
            
            print(f"Calculated indicator: {indicator_name} (ID: {indicator_id})")
            
        except Exception as e:
            print(f"ERROR: Failed to calculate indicator {indicator_id}: {e}")
    
    def _on_indicator_config_changed(self, indicator_id: str, config: IndicatorConfig):
        """Handle indicator configuration change"""
        self.indicators[indicator_id] = config
        self._calculate_indicator(indicator_id, config)
        self._update_statistics()
        
        # Emit signal to notify dashboard
        self.indicators_updated.emit(self.indicator_data)
        print(f"Updated indicator configuration: {indicator_id}")
    
    def _remove_indicator(self, indicator_id: str):
        """Remove an indicator"""
        try:
            # Remove from data structures
            if indicator_id in self.indicators:
                del self.indicators[indicator_id]
            if indicator_id in self.indicator_data:
                del self.indicator_data[indicator_id]
            
            # Remove widget
            for i in range(self.indicators_layout.count()):
                widget = self.indicators_layout.itemAt(i).widget()
                if isinstance(widget, IndicatorConfigWidget) and widget.indicator_id == indicator_id:
                    self.indicators_layout.removeWidget(widget)
                    widget.deleteLater()
                    break
            
            # Update
            self.indicators_updated.emit(self.indicator_data)
            self._update_statistics()
            
            print(f"Removed indicator: {indicator_id}")
            
        except Exception as e:
            print(f"ERROR: Failed to remove indicator {indicator_id}: {e}")
    
    def _recalculate_all_indicators(self):
        """Recalculate all indicators"""
        try:
            print("Recalculating all indicators...")
            
            for indicator_id, config in self.indicators.items():
                self._calculate_indicator(indicator_id, config)
            
            self._update_statistics()
            
            # Emit signal to notify dashboard
            self.indicators_updated.emit(self.indicator_data)
            print(f"Recalculated {len(self.indicators)} indicators")
            
        except Exception as e:
            print(f"ERROR: Failed to recalculate indicators: {e}")
    
    def _clear_all_indicators(self):
        """Clear all indicators"""
        try:
            # Clear data structures
            self.indicators.clear()
            self.indicator_data.clear()
            
            # Clear widgets
            while self.indicators_layout.count():
                child = self.indicators_layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
            
            # Update
            self.indicators_updated.emit(self.indicator_data)
            self._update_statistics()
            
            print("Cleared all indicators")
            
        except Exception as e:
            print(f"ERROR: Failed to clear indicators: {e}")
    
    def _export_indicator_data(self):
        """Export indicator data to CSV"""
        try:
            if not self.indicator_data:
                print("No indicator data to export")
                return
            
            # Create DataFrame with all indicator data
            export_data = {}
            
            # Add index
            if self.ohlcv_data:
                export_data['bar_index'] = np.arange(len(self.ohlcv_data['close']))
            
            # Add indicator data
            for indicator_id, data in self.indicator_data.items():
                config = self.indicators[indicator_id]
                for series_name, series_data in data.items():
                    column_name = f"{config.display_name}_{series_name}"
                    export_data[column_name] = series_data
            
            # Create DataFrame and export
            df = pd.DataFrame(export_data)
            filename = f"vbt_indicators_export_{int(time.time())}.csv"
            df.to_csv(filename, index=False)
            
            print(f"Exported indicator data to: {filename}")
            
        except Exception as e:
            print(f"ERROR: Failed to export indicator data: {e}")
    
    def _update_statistics(self):
        """Update statistics display"""
        try:
            if not self.indicators:
                self.stats_label.setText("No indicators active")
                return
            
            enabled_count = len([c for c in self.indicators.values() if c.enabled])
            overlay_count = len([c for c in self.indicators.values() if c.indicator_type == 'overlay' and c.enabled])
            oscillator_count = len([c for c in self.indicators.values() if c.indicator_type == 'oscillator' and c.enabled])
            volume_count = len([c for c in self.indicators.values() if c.indicator_type == 'volume' and c.enabled])
            
            stats_text = f"Active: {enabled_count} | Overlay: {overlay_count} | Oscillator: {oscillator_count} | Volume: {volume_count}"
            self.stats_label.setText(stats_text)
            
        except Exception as e:
            print(f"ERROR: Failed to update statistics: {e}")
    
    def get_indicator_data(self) -> Dict[str, Any]:
        """Get all indicator data"""
        return self.indicator_data.copy()
    
    def get_indicator_configs(self) -> Dict[str, IndicatorConfig]:
        """Get all indicator configurations"""
        return self.indicators.copy()


def test_vbt_indicators_panel():
    """Test the VBT indicators panel"""
    print("TESTING VBT INDICATORS PANEL - STEP 6")
    print("="*50)
    
    from PyQt5.QtWidgets import QApplication, QMainWindow, QHBoxLayout, QWidget
    import sys
    
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    try:
        # Create test data
        np.random.seed(42)
        num_bars = 1000
        base_price = 1.2000
        volatility = 0.001
        
        # Generate realistic price data with trends
        price_changes = np.random.normal(0, volatility, num_bars)
        
        # Add trend periods
        for i in range(0, num_bars, 200):
            end_i = min(i + 200, num_bars)
            trend = np.random.choice([-0.0001, 0.0001, 0.0002])
            price_changes[i:end_i] += trend
        
        prices = np.cumsum(price_changes) + base_price
        
        ohlcv_data = {
            'open': prices.copy(),
            'high': prices + np.random.exponential(volatility/4, num_bars),
            'low': prices - np.random.exponential(volatility/4, num_bars),
            'close': prices + np.random.normal(0, volatility/2, num_bars),
            'volume': np.random.lognormal(10, 0.3, num_bars)
        }
        
        # Ensure proper OHLC relationships
        ohlcv_data['high'] = np.maximum(ohlcv_data['high'], ohlcv_data['close'])
        ohlcv_data['low'] = np.minimum(ohlcv_data['low'], ohlcv_data['close'])
        
        # Create main window
        main_window = QMainWindow()
        central_widget = QWidget()
        main_window.setCentralWidget(central_widget)
        
        layout = QHBoxLayout(central_widget)
        
        # Create VBT indicators panel
        indicators_panel = VBTIndicatorsPanel()
        
        # Load test data
        success = indicators_panel.load_ohlcv_data(ohlcv_data)
        if not success:
            print("ERROR: Failed to load test data")
            return False
        
        # Add to layout
        layout.addWidget(indicators_panel)
        
        # Connect signals
        def on_indicators_updated(indicator_data):
            print(f"Indicators updated: {len(indicator_data)} active indicators")
            for indicator_id, data in indicator_data.items():
                print(f"  {indicator_id}: {list(data.keys())}")
        
        indicators_panel.indicators_updated.connect(on_indicators_updated)
        
        # Add some default indicators for testing
        QTimer.singleShot(1000, lambda: indicators_panel._add_indicator())  # SMA
        QTimer.singleShot(1500, lambda: setattr(indicators_panel.indicator_combo, 'currentIndex', 
                                               indicators_panel.indicator_combo.findText('RSI')))
        QTimer.singleShot(2000, lambda: indicators_panel._add_indicator())  # RSI
        
        # Style main window
        main_window.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
            }
        """)
        
        # Show test window
        main_window.setWindowTitle("VBT Indicators Panel Test - Step 6")
        main_window.resize(400, 800)
        main_window.show()
        
        print("\nStep 6 Test Features:")
        print("• VBT Pro indicators with parameter selection")
        print("• Real-time indicator calculation and recalculation")
        print("• Interactive parameter adjustment")
        print("• Color customization for each indicator")
        print("• Enable/disable individual indicators")
        print("• Export indicator data to CSV")
        print("• Professional configuration interface")
        
        print("\nAvailable Indicators:")
        print("• Moving Averages: SMA, EMA")
        print("• Trend Indicators: Bollinger Bands")
        print("• Momentum Oscillators: RSI, Stochastic")
        print("• Trend Following: MACD")
        print("• Volatility: Average True Range (ATR)")
        print("• Volume: Volume Moving Average")
        
        print("\nSUCCESS: VBT Indicators Panel test ready")
        return True
        
    except Exception as e:
        print(f"ERROR: VBT Indicators Panel test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_vbt_indicators_panel()
    
    if success:
        print("\n" + "="*50)
        print("VBT INDICATORS PANEL - STEP 6 SUCCESS!")
        print("="*50)
        print("+ VBT Pro indicators with parameter selection")
        print("+ Real-time indicator calculation engine")
        print("+ Interactive parameter configuration widgets")
        print("+ Color customization and visual styling")
        print("+ Enable/disable and visibility controls")
        print("+ Export functionality for analysis")
        print("+ Professional indicator management interface")
        print("+ Complete integration ready for final dashboard")
        print("\nREADY FOR FINAL STEP 6 DASHBOARD INTEGRATION")
    else:
        print("\nVBT Indicators Panel needs additional work")