from typing import Any, Callable, Dict, Optional
from PySide6.QtCore import QObject, Signal, QTimer


class SignalBlocker:
    """Context manager for blocking signals"""
    
    def __init__(self, *objects: QObject):
        self.objects = objects
        self.states = []
    
    def __enter__(self):
        for obj in self.objects:
            self.states.append(obj.blockSignals(True))
        return self
    
    def __exit__(self, *args):
        for obj, state in zip(self.objects, self.states):
            obj.blockSignals(state)


class Debouncer(QObject):
    """Debounces signal emissions"""
    
    def __init__(self, timeout_ms: int = 300):
        super().__init__()
        self.timeout_ms = timeout_ms
        self.timer = QTimer()
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self._emit_debounced)
        self.pending_args = None
        self.pending_kwargs = None
        self.callback = None
    
    def debounce(self, callback: Callable, *args, **kwargs):
        """Debounce callback execution"""
        self.callback = callback
        self.pending_args = args
        self.pending_kwargs = kwargs
        self.timer.start(self.timeout_ms)
    
    def _emit_debounced(self):
        """Emit debounced signal"""
        if self.callback and self.pending_args is not None:
            self.callback(*self.pending_args, **self.pending_kwargs)
            self.pending_args = None
            self.pending_kwargs = None
            self.callback = None
    
    def cancel(self):
        """Cancel pending debounce"""
        self.timer.stop()
        self.pending_args = None
        self.pending_kwargs = None
        self.callback = None


class Throttler(QObject):
    """Throttles signal emissions"""
    
    def __init__(self, interval_ms: int = 100):
        super().__init__()
        self.interval_ms = interval_ms
        self.last_emission = 0
        self.timer = QTimer()
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self._emit_throttled)
        self.pending_args = None
        self.pending_kwargs = None
        self.callback = None
        self.has_pending = False
    
    def throttle(self, callback: Callable, *args, **kwargs):
        """Throttle callback execution"""
        import time
        now = int(time.time() * 1000)
        
        if now - self.last_emission >= self.interval_ms:
            # Can emit immediately
            self.last_emission = now
            callback(*args, **kwargs)
        else:
            # Store for later emission
            self.callback = callback
            self.pending_args = args
            self.pending_kwargs = kwargs
            self.has_pending = True
            
            if not self.timer.isActive():
                remaining = self.interval_ms - (now - self.last_emission)
                self.timer.start(remaining)
    
    def _emit_throttled(self):
        """Emit throttled signal"""
        if self.has_pending and self.callback:
            import time
            self.last_emission = int(time.time() * 1000)
            self.callback(*self.pending_args, **self.pending_kwargs)
            self.has_pending = False
            self.pending_args = None
            self.pending_kwargs = None
            self.callback = None


class SignalMultiplexer(QObject):
    """Multiplexes multiple signals into one"""
    
    def __init__(self):
        super().__init__()
        self.sources: Dict[str, QObject] = {}
    
    def add_source(self, name: str, source: QObject, signal_name: str):
        """Add signal source"""
        signal = getattr(source, signal_name, None)
        if signal and isinstance(signal, Signal):
            signal.connect(lambda *args: self._forward(name, *args))
            self.sources[name] = source
    
    def _forward(self, source_name: str, *args):
        """Forward signal"""
        self.forwarded.emit(source_name, *args)
    
    forwarded = Signal(str, object)


class SignalInspector(QObject):
    """Inspects and logs signals for debugging"""
    
    def __init__(self, enabled: bool = True):
        super().__init__()
        self.enabled = enabled
        self.history: list = []
    
    def inspect(self, signal: Signal, *args, **kwargs):
        """Inspect signal emission"""
        if not self.enabled:
            return
        
        import time
        from datetime import datetime
        
        info = {
            'timestamp': datetime.now().isoformat(),
            'signal': signal,
            'args': args,
            'kwargs': kwargs
        }
        
        self.history.append(info)
        print(f"[Signal] {signal}: {args} {kwargs}")
    
    def get_history(self, limit: int = None) -> list:
        """Get signal history"""
        if limit:
            return self.history[-limit:]
        return self.history
    
    def clear_history(self):
        """Clear signal history"""
        self.history.clear()