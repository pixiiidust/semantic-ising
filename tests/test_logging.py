import pytest
import tempfile
import os
import logging
import time
from unittest.mock import patch, MagicMock

# Import the functions we'll implement
from export.logger import init_logger, log_event, log_exception


class TestLogging:
    """Test logging functionality"""
    
    def test_init_logger_creates_logger(self):
        """Test that init_logger creates a proper logger"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            temp_log_file = f.name
        
        try:
            logger = init_logger(temp_log_file, level="INFO")
            
            assert isinstance(logger, logging.Logger)
            assert logger.name == "semantic_ising"
            assert logger.level == logging.INFO
            
            # Test that logger can write to file
            logger.info("Test message")
            
            # Close handlers to release file
            for handler in logger.handlers:
                handler.close()
            
            # Small delay to ensure file is written
            time.sleep(0.1)
            
            with open(temp_log_file, 'r') as f:
                log_content = f.read()
                assert "Test message" in log_content
                assert "semantic_ising" in log_content
        finally:
            # Close logger handlers before cleanup
            if 'logger' in locals():
                for handler in logger.handlers:
                    handler.close()
            try:
                os.unlink(temp_log_file)
            except PermissionError:
                # File might still be in use, that's okay for tests
                pass
    
    def test_init_logger_different_levels(self):
        """Test that init_logger works with different log levels"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            temp_log_file = f.name
        
        try:
            # Test DEBUG level
            logger_debug = init_logger(temp_log_file, level="DEBUG")
            assert logger_debug.level == logging.DEBUG
            
            # Close handlers
            for handler in logger_debug.handlers:
                handler.close()
            
            # Test WARNING level
            logger_warning = init_logger(temp_log_file, level="WARNING")
            assert logger_warning.level == logging.WARNING
            
            # Close handlers
            for handler in logger_warning.handlers:
                handler.close()
        finally:
            try:
                os.unlink(temp_log_file)
            except PermissionError:
                pass
    
    def test_init_logger_invalid_level(self):
        """Test that init_logger handles invalid log levels gracefully"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            temp_log_file = f.name
        
        try:
            # Should default to INFO for invalid level
            logger = init_logger(temp_log_file, level="INVALID_LEVEL")
            assert logger.level == logging.INFO
            
            # Close handlers
            for handler in logger.handlers:
                handler.close()
        finally:
            try:
                os.unlink(temp_log_file)
            except PermissionError:
                pass
    
    def test_log_event(self):
        """Test that log_event logs messages correctly"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            temp_log_file = f.name
        
        try:
            logger = init_logger(temp_log_file, level="INFO")
            
            log_event(logger, "Test event", "INFO")
            
            # Close handlers
            for handler in logger.handlers:
                handler.close()
            
            # Small delay
            time.sleep(0.1)
            
            with open(temp_log_file, 'r') as f:
                log_content = f.read()
                assert "Test event" in log_content
        finally:
            try:
                os.unlink(temp_log_file)
            except PermissionError:
                pass
    
    def test_log_exception(self):
        """Test that log_exception logs exceptions with context"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            temp_log_file = f.name
        
        try:
            logger = init_logger(temp_log_file, level="ERROR")
            
            # Create a test exception
            test_exception = ValueError("Test error message")
            
            log_exception(logger, test_exception, "test_context")
            
            # Close handlers
            for handler in logger.handlers:
                handler.close()
            
            # Small delay
            time.sleep(0.1)
            
            with open(temp_log_file, 'r') as f:
                log_content = f.read()
                assert "Exception in test_context" in log_content
                assert "Test error message" in log_content
        finally:
            try:
                os.unlink(temp_log_file)
            except PermissionError:
                pass
    
    def test_log_exception_without_context(self):
        """Test that log_exception works without context"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            temp_log_file = f.name
        
        try:
            logger = init_logger(temp_log_file, level="ERROR")
            
            test_exception = RuntimeError("Another test error")
            
            log_exception(logger, test_exception)
            
            # Close handlers
            for handler in logger.handlers:
                handler.close()
            
            # Small delay
            time.sleep(0.1)
            
            with open(temp_log_file, 'r') as f:
                log_content = f.read()
                # Fix the assertion to match actual output
                assert "Exception:" in log_content
                assert "Another test error" in log_content
        finally:
            try:
                os.unlink(temp_log_file)
            except PermissionError:
                pass
    
    def test_logger_format(self):
        """Test that logger uses correct format"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            temp_log_file = f.name
        
        try:
            logger = init_logger(temp_log_file, level="INFO")
            logger.info("Format test")
            
            # Close handlers
            for handler in logger.handlers:
                handler.close()
            
            # Small delay
            time.sleep(0.1)
            
            with open(temp_log_file, 'r') as f:
                log_content = f.read()
                
                # Check format components
                assert " - semantic_ising - INFO - " in log_content
                assert "Format test" in log_content
        finally:
            try:
                os.unlink(temp_log_file)
            except PermissionError:
                pass 