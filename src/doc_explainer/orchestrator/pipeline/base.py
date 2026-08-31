from abc import ABC, abstractmethod
from typing import Any, Optional
import logging
from datetime import datetime

from ..base.interfaces import Pipeline
from ..base.exceptions import ValidationError, PipelineError
from ..models.requests import BaseRequest
from ..models.responses import BaseResponse


class BasePipeline(Pipeline, ABC):
    """Base class for all pipelines"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(self.__class__.__name__)
    
    def process(self, request: BaseRequest) -> BaseResponse:
        """Process request through pipeline"""
        start_time = datetime.now()
        
        try:
            # Validate request
            if not self.validate(request):
                raise ValidationError(f"Invalid request: {request}")
            
            # Log start
            self.logger.info(f"Processing {self.__class__.__name__} for user {request.user_id}")
            
            # Process
            result = self._process(request)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            result.processing_time_ms = processing_time
            
            return result
            
        except ValidationError as e:
            self.logger.error(f"Validation error: {e}")
            return self._create_error_response(str(e), request)
        except PipelineError as e:
            self.logger.error(f"Pipeline error: {e}")
            return self._create_error_response(str(e), request)
        except Exception as e:
            self.logger.exception(f"Unexpected error: {e}")
            return self._create_error_response(f"Unexpected error: {str(e)}", request)
   
    def _process(self, request: BaseRequest) -> BaseResponse:
        """Internal process method to be implemented by subclasses"""
        pass
    
    def validate(self, request: BaseRequest) -> bool:
        """Validate request - can be overridden by subclasses"""
        return request is not None and hasattr(request, 'user_id')
    
    def _create_error_response(self, error_message: str, request: BaseRequest) -> BaseResponse:
        """Create error response"""
        return BaseResponse(
            success=False,
            message="Processing failed",
            error=error_message
        )


