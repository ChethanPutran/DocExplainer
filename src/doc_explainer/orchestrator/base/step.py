from enum import Enum
from .node import Node
import functools
from typing import Callable, Any, Dict, Optional
from .node import Node
from .context import PipelineContext

class StepStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CACHED = "cached"
    SKIPPED = "skipped"
    

class Step:
    """Wraps a function to create a step."""
    def __init__(self, fn: Callable, name: Optional[str] = None, annotations: Optional[Dict[str, Any]] = None):
        self.fn = fn
        self.name = name or fn.__name__
        self.annotations = annotations or {}
        functools.update_wrapper(self, fn)

    def __call__(self, *args, **kwargs) -> Node:
        # When called inside a pipeline context, create a Node
        # Capture the current context from the global or thread-local store
        ctx = get_current_context()
        if ctx is None:
            # If not inside a pipeline, execute directly (fallback)
            return self.fn(*args, **kwargs)

        # Build inputs dict from positional args and kwargs
        # We'll map positional args to parameter names using introspection
        import inspect
        sig = inspect.signature(self.fn)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        input_kwargs = dict(bound.arguments)

        node = Node(
            step_name=self.name,
            inputs=input_kwargs,
            step=self,  # store reference
            annotations=self.annotations   # <-- pass annotations
        )
        ctx.add_node(node)
        return node

    def __repr__(self):
        return f"Step({self.name})"


# Global context (simplified, use thread-local for production)
_context_stack = []


def get_current_context() -> Optional[PipelineContext]:
    return _context_stack[-1] if _context_stack else None


def push_context(ctx: PipelineContext):
    _context_stack.append(ctx)


def pop_context():
    _context_stack.pop()


def step(fn: Callable = None, *, name: Optional[str] = None, **annotations):
    def decorator(func):
        return Step(func, name=name, annotations=annotations)
    if fn is None:
        return decorator
    return decorator(fn)