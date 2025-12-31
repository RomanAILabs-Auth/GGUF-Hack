#!/usr/bin/env python3
"""
World-Class GGUF Editor Application
A comprehensive GUI tool for editing GGUF model files with Easy and Advanced modes.
"""
#Copyright Daniel Harding - RomanAILabs
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import json
import os
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import sys
import traceback
import logging
import time
import inspect
from functools import wraps
import threading

# Setup logging for debugging
LOG_DIR = Path.home() / ".gguf_editor_logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"gguf_editor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Enhanced logging configuration with more detail
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - [%(levelname)8s] - [%(filename)s:%(lineno)d] - %(funcName)s() - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info("="*80)
logger.info(f"GGUF Editor started at {datetime.now().isoformat()}")
logger.info(f"Log file: {LOG_FILE}")
logger.info(f"Python version: {sys.version}")
logger.info(f"Platform: {sys.platform}")
logger.info("="*80)

# Helper function for object inspection
def inspect_object(obj, obj_name="object", max_depth=2, current_depth=0):
    """Inspect an object and return detailed information for logging."""
    if current_depth >= max_depth:
        return f"{obj_name}: <max depth reached>"
    
    try:
        info_parts = [f"{obj_name}:"]
        info_parts.append(f"  Type: {type(obj).__name__}")
        info_parts.append(f"  Module: {type(obj).__module__}")
        
        # Get attributes
        if hasattr(obj, '__dict__'):
            attrs = {k: v for k, v in obj.__dict__.items() if not k.startswith('_')}
            if attrs:
                info_parts.append(f"  Attributes ({len(attrs)}):")
                for key, value in list(attrs.items())[:10]:  # Limit to first 10
                    value_str = str(value)
                    if len(value_str) > 100:
                        value_str = value_str[:97] + "..."
                    info_parts.append(f"    {key}: {value_str}")
                if len(attrs) > 10:
                    info_parts.append(f"    ... and {len(attrs) - 10} more")
        
        # Get methods
        methods = [m for m in dir(obj) if not m.startswith('_') and callable(getattr(obj, m, None))]
        if methods:
            info_parts.append(f"  Methods ({len(methods)}): {', '.join(methods[:20])}")
            if len(methods) > 20:
                info_parts.append(f"    ... and {len(methods) - 20} more")
        
        # Check for specific attributes that might be relevant
        special_attrs = ['keys', 'get_value', 'tensors', 'tensor_count', 'architecture', 'metadata']
        for attr in special_attrs:
            if hasattr(obj, attr):
                try:
                    value = getattr(obj, attr)
                    if callable(value):
                        info_parts.append(f"  Has method: {attr}()")
                    else:
                        value_str = str(value)
                        if len(value_str) > 100:
                            value_str = value_str[:97] + "..."
                        info_parts.append(f"  Has attribute: {attr} = {value_str}")
                except Exception as e:
                    info_parts.append(f"  Has attribute: {attr} (error accessing: {e})")
        
        return "\n".join(info_parts)
    except Exception as e:
        return f"{obj_name}: <error inspecting: {e}>"

# Decorator for function entry/exit logging
def log_function_call(func):
    """Decorator to log function entry, parameters, and exit."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        start_time = time.time()
        
        # Log entry
        logger.debug(f"→ ENTERING {func_name}()")
        
        # Log arguments (sanitized)
        if args:
            args_str = ", ".join([f"{type(a).__name__}" for a in args[:3]])
            if len(args) > 3:
                args_str += f" ... (+{len(args)-3} more)"
            logger.debug(f"  Args: {args_str}")
        
        if kwargs:
            kwargs_str = ", ".join([f"{k}={type(v).__name__}" for k, v in list(kwargs.items())[:5]])
            if len(kwargs) > 5:
                kwargs_str += f" ... (+{len(kwargs)-5} more)"
            logger.debug(f"  Kwargs: {kwargs_str}")
        
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            logger.debug(f"← EXITING {func_name}() - SUCCESS (took {elapsed:.3f}s)")
            
            # Log return value type
            if result is not None:
                logger.debug(f"  Return type: {type(result).__name__}")
                if isinstance(result, (dict, list)):
                    logger.debug(f"  Return size: {len(result)} items")
            
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"← EXITING {func_name}() - EXCEPTION after {elapsed:.3f}s: {type(e).__name__}: {e}")
            raise
    
    return wrapper

# Try to import GGUF editor functionality
try:
    from citadel_fusion_editor import GGUFEditor
    USE_EDITOR = True
    logger.info("Successfully imported GGUFEditor from citadel_fusion_editor")
except ImportError as e:
    USE_EDITOR = False
    logger.warning(f"Failed to import GGUFEditor from citadel_fusion_editor: {e}")
    # Create a minimal fallback
    class GGUFEditor:
        def __init__(self, file_path: str):
            self.file_path = Path(file_path)
            self.metadata = {}
            self.reader = None
            logger.info(f"Created fallback GGUFEditor for: {file_path}")
        
        @log_function_call
        def load_metadata_only(self):
            """Optimized load that only reads metadata, not tensors (saves memory for large files)."""
            try:
                logger.info("="*60)
                logger.info("LOAD_METADATA_ONLY: Starting metadata-only load operation")
                logger.info(f"  File path: {self.file_path}")
                logger.info(f"  File exists: {self.file_path.exists()}")
                
                if self.file_path.exists():
                    file_size = self.file_path.stat().st_size
                    file_size_mb = file_size / (1024 * 1024)
                    file_size_gb = file_size / (1024 * 1024 * 1024)
                    logger.info(f"  File size: {file_size} bytes ({file_size_mb:.2f} MB / {file_size_gb:.2f} GB)")
                
                logger.info("Attempting to import gguf library...")
                import gguf
                logger.info(f"gguf library imported successfully from: {gguf.__file__ if hasattr(gguf, '__file__') else 'unknown'}")
                logger.info(f"gguf version: {getattr(gguf, '__version__', 'unknown')}")
                
                logger.info(f"Creating GGUFReader for metadata-only loading: {self.file_path}")
                if not self.file_path.exists():
                    raise FileNotFoundError(f"File not found: {self.file_path}")
                
                # Use memory mapping if available to reduce memory usage
                try:
                    logger.debug("Instantiating GGUFReader...")
                    start_time = time.time()
                    # Try to create reader without loading all tensors
                    self.reader = gguf.GGUFReader(str(self.file_path))
                    elapsed = time.time() - start_time
                    logger.info(f"GGUFReader created successfully in {elapsed:.3f}s")
                    
                    # Inspect the reader object
                    reader_info = inspect_object(self.reader, "GGUFReader")
                    logger.debug(f"GGUFReader inspection:\n{reader_info}")
                    
                except MemoryError as e:
                    logger.error(f"Out of memory creating GGUFReader: {e}")
                    logger.error(f"  File size: {self.file_path.stat().st_size / (1024**3):.2f} GB")
                    raise MemoryError(
                        f"File too large to load in memory ({self.file_path.stat().st_size / (1024**3):.2f} GB).\n"
                        "Try using a machine with more RAM or edit metadata directly in the file."
                    )
                except Exception as e:
                    logger.error(f"Exception creating GGUFReader: {type(e).__name__}: {e}")
                    logger.error(f"  Traceback:\n{traceback.format_exc()}")
                    raise
                
                logger.info("Reading metadata keys only (skipping tensor data)...")
                metadata = {}
                
                # Get keys without loading full file
                try:
                    logger.debug("Attempting to get metadata keys from reader...")
                    logger.debug(f"  reader type: {type(self.reader)}")
                    logger.debug(f"  has 'keys' method: {hasattr(self.reader, 'keys')}")
                    logger.debug(f"  has '__iter__': {hasattr(self.reader, '__iter__')}")
                    logger.debug(f"  dir(reader) contains: {[x for x in dir(self.reader) if not x.startswith('_')][:20]}")
                    
                    # Use the correct API: reader.fields.keys()
                    if hasattr(self.reader, 'fields') and hasattr(self.reader.fields, 'keys'):
                        logger.debug("  Using reader.fields.keys()...")
                        keys_result = self.reader.fields.keys()
                        logger.debug(f"  fields.keys() returned: {type(keys_result)}")
                        
                        # Try to convert to list
                        try:
                            keys = list(keys_result)
                            logger.info(f"Found {len(keys)} metadata keys")
                            logger.debug(f"  First 10 keys: {keys[:10]}")
                        except Exception as e:
                            logger.error(f"Failed to convert fields.keys() result to list: {e}")
                            logger.error(f"  keys() result type: {type(keys_result)}")
                            logger.error(f"  keys() result value: {keys_result}")
                            raise
                    else:
                        logger.error("  'fields.keys()' method not found on GGUFReader!")
                        logger.error(f"  Available attributes: {[x for x in dir(self.reader) if not x.startswith('_')][:30]}")
                        raise AttributeError("'GGUFReader' object has no attribute 'fields' or 'fields.keys'")
                        
                except AttributeError as e:
                    logger.error(f"AttributeError getting keys: {e}")
                    logger.error(f"  Full traceback:\n{traceback.format_exc()}")
                    logger.warning("Attempting fallback: trying individual access to common keys")
                    # Fallback: try to read common metadata keys
                    common_keys = [
                        'general.name', 'general.description',
                        'llama.system_prompt', 'chat_template.system_prompt',
                        'llama.persona_prompt', 'chat_template.persona_prompt'
                    ]
                    keys = []
                    for key in common_keys:
                        try:
                            # Try get_field().contents() first (correct API)
                            if hasattr(self.reader, 'get_field'):
                                field = self.reader.get_field(key)
                                if field is not None:
                                    logger.debug(f"  Successfully found field for {key}")
                                    keys.append(key)
                            # Fallback to get_value if available
                            elif hasattr(self.reader, 'get_value'):
                                logger.debug(f"  Trying to get value for key: {key}")
                                value = self.reader.get_value(key)
                                logger.debug(f"  Successfully got value for {key}")
                                keys.append(key)
                            else:
                                logger.warning(f"  No method to access metadata available")
                        except Exception as e2:
                            logger.debug(f"  Failed to get value for {key}: {e2}")
                            pass
                    logger.info(f"Fallback: Found {len(keys)} common metadata keys")
                except Exception as e:
                    logger.error(f"Unexpected error getting keys: {type(e).__name__}: {e}")
                    logger.error(f"  Full traceback:\n{traceback.format_exc()}")
                    raise
                
                # Load metadata values one at a time (more memory efficient)
                logger.info(f"Loading {len(keys)} metadata values...")
                start_time = time.time()
                successful_keys = 0
                failed_keys = 0
                
                for i, key in enumerate(keys):
                    try:
                        logger.debug(f"  [{i+1}/{len(keys)}] Loading key: {key}")
                        value = self.reader.get_value(key)
                        value_type = type(value).__name__
                        value_size = len(str(value)) if isinstance(value, str) else "N/A"
                        logger.debug(f"    Type: {value_type}, Size: {value_size}")
                        
                        # Truncate very large string values to prevent memory issues
                        if isinstance(value, str) and len(value) > 100000:
                            logger.warning(f"Truncating large value for key '{key}' (was {len(value)} chars)")
                            value = value[:100000] + "... [truncated]"
                        
                        metadata[key] = value
                        successful_keys += 1
                        
                        if (i + 1) % 100 == 0:
                            elapsed = time.time() - start_time
                            rate = (i + 1) / elapsed if elapsed > 0 else 0
                            logger.debug(f"Progress: {i + 1}/{len(keys)} keys loaded ({rate:.1f} keys/sec)")
                            
                    except MemoryError as e:
                        logger.error(f"Out of memory loading key '{key}': {e}")
                        logger.error(f"  Progress: {i}/{len(keys)} keys loaded successfully before error")
                        raise MemoryError(
                            f"Out of memory while loading metadata key '{key}'.\n"
                            "The file may be too large for this system."
                        )
                    except Exception as e:
                        failed_keys += 1
                        logger.warning(f"Failed to load key '{key}': {type(e).__name__}: {e}")
                        logger.debug(f"  Traceback for key '{key}':\n{traceback.format_exc()}")
                        pass
                
                elapsed = time.time() - start_time
                logger.info(f"Metadata loading complete:")
                logger.info(f"  Total keys: {len(keys)}")
                logger.info(f"  Successful: {successful_keys}")
                logger.info(f"  Failed: {failed_keys}")
                logger.info(f"  Time taken: {elapsed:.3f}s")
                logger.info(f"  Average rate: {successful_keys/elapsed:.1f} keys/sec" if elapsed > 0 else "  Average rate: N/A")
                
                self.metadata = metadata
                logger.info(f"Successfully loaded {len(metadata)} metadata entries (metadata-only mode)")
                logger.info("="*60)
                return metadata
            except ImportError as e:
                logger.error(f"gguf library not available: {e}")
                raise ImportError("gguf library required. Install with: pip install gguf")
            except MemoryError:
                raise  # Re-raise memory errors as-is
            except Exception as e:
                logger.error(f"Error loading GGUF file: {e}\n{traceback.format_exc()}")
                raise
        
        def load(self):
            """Standard load method (loads everything)."""
            try:
                logger.info("Attempting to import gguf library...")
                import gguf
                logger.info("gguf library imported successfully")
                
                logger.info(f"Creating GGUFReader for: {self.file_path}")
                if not self.file_path.exists():
                    raise FileNotFoundError(f"File not found: {self.file_path}")
                
                self.reader = gguf.GGUFReader(str(self.file_path))
                logger.info("GGUFReader created successfully")
                
                logger.info("Reading metadata keys...")
                metadata = {}
                
                # Use the correct API: reader.fields.keys()
                if hasattr(self.reader, 'fields') and hasattr(self.reader.fields, 'keys'):
                    keys = list(self.reader.fields.keys())
                    logger.info(f"Found {len(keys)} metadata keys")
                else:
                    logger.error("Cannot access reader.fields.keys()")
                    raise AttributeError("'GGUFReader' object has no attribute 'fields' or 'fields.keys'")
                
                for key in keys:
                    try:
                        # Get value using correct API
                        value = None
                        if hasattr(self.reader, 'get_value'):
                            try:
                                value = self.reader.get_value(key)
                            except:
                                # Fall back to get_field().contents()
                                field = self.reader.get_field(key)
                                if field is not None:
                                    value = field.contents()
                        elif hasattr(self.reader, 'get_field'):
                            field = self.reader.get_field(key)
                            if field is not None:
                                value = field.contents()
                        
                        if value is not None:
                            metadata[key] = value
                            logger.debug(f"Loaded key: {key} (type: {type(value).__name__})")
                    except Exception as e:
                        logger.warning(f"Failed to load key '{key}': {e}")
                        pass
                
                self.metadata = metadata
                logger.info(f"Successfully loaded {len(metadata)} metadata entries")
                return metadata
            except ImportError as e:
                logger.error(f"gguf library not available: {e}")
                raise ImportError("gguf library required. Install with: pip install gguf")
            except Exception as e:
                logger.error(f"Error loading GGUF file: {e}\n{traceback.format_exc()}")
                raise
        
        def get_value(self, key: str, default: Any = None):
            if self.reader:
                try:
                    return self.reader.get_value(key)
                except:
                    return self.metadata.get(key, default)
            return self.metadata.get(key, default)
        
        def create_backup(self):
            import shutil
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.file_path.parent / f"{self.file_path.stem}_backup_{timestamp}.gguf"
            logger.info(f"Creating backup: {backup_path}")
            shutil.copy2(self.file_path, backup_path)
            return str(backup_path)
        
        def save(self, output_path=None, name=None, system_prompt=None, 
                 persona_prompt=None, custom_metadata=None):
            """Save modified GGUF file with new metadata."""
            if output_path is None:
                output_path = str(self.file_path)
            
            output_path = Path(output_path)
            logger.info(f"Saving to: {output_path}")
            
            try:
                import gguf
                logger.info("gguf library available for saving")
            except ImportError:
                error_msg = (
                    "gguf library required for saving.\n\n"
                    "Install with: pip install gguf\n\n"
                    "Or use the CLI tool: python3 citadel_fusion_editor.py"
                )
                logger.error(f"gguf library not available: {error_msg}")
                raise ImportError(error_msg)
            
            try:
                # Read the original file
                logger.info("="*60)
                logger.info("SAVE OPERATION: Reading original GGUF file...")
                logger.info(f"  Source file: {self.file_path}")
                logger.info(f"  Output file: {output_path}")
                logger.info(f"  Source exists: {self.file_path.exists()}")
                logger.info(f"  Source size: {self.file_path.stat().st_size / (1024**2):.2f} MB" if self.file_path.exists() else "N/A")
                
                start_time = time.time()
                logger.debug("Creating GGUFReader...")
                reader = gguf.GGUFReader(str(self.file_path))
                elapsed = time.time() - start_time
                logger.info(f"GGUFReader created in {elapsed:.3f}s")
                
                # Inspect reader
                reader_info = inspect_object(reader, "GGUFReader (save)")
                logger.debug(f"Reader inspection:\n{reader_info}")
                
                # Create writer
                arch = reader.architecture if hasattr(reader, 'architecture') else 'llama'
                logger.info(f"Creating GGUFWriter with architecture: {arch}")
                logger.debug(f"  Architecture source: {'reader.architecture' if hasattr(reader, 'architecture') else 'default (llama)'}")
                writer = gguf.GGUFWriter(str(output_path), arch=arch)
                logger.info("GGUFWriter created successfully")
                
                # Copy tensors (this is memory-intensive for large files)
                # Try different ways to get tensor count/access tensors
                tensor_count = 0
                try:
                    # Try attribute access using getattr for safety
                    tensor_count = getattr(reader, 'tensor_count', None)
                    if tensor_count is None:
                        # Try tensors attribute
                        tensors_list = getattr(reader, 'tensors', None)
                        if tensors_list is not None:
                            tensor_count = len(tensors_list)
                        else:
                            # Try __len__
                            try:
                                tensor_count = len(reader)
                            except (TypeError, AttributeError):
                                # Try to get count from metadata
                                try:
                                    tensor_count_val = reader.get_value('GGUF.tensor_count', None)
                                    if tensor_count_val is not None:
                                        tensor_count = int(tensor_count_val)
                                except:
                                    tensor_count = 0
                except Exception as e:
                    logger.warning(f"Could not determine tensor count: {e}")
                    tensor_count = 0
                
                logger.info(f"Copying tensors (count: {tensor_count if tensor_count > 0 else 'unknown'})...")
                
                # Copy tensors - try different iteration methods
                tensors_copied = 0
                try:
                    # Method 1: Try iterating if reader is iterable
                    if hasattr(reader, '__iter__'):
                        for tensor in reader:
                            try:
                                writer.add_tensor(
                                    tensor.name,
                                    tensor.tensor_type,
                                    tensor.shape,
                                    tensor.data
                                )
                                tensors_copied += 1
                                if tensors_copied % 100 == 0:
                                    logger.debug(f"Copied {tensors_copied} tensors...")
                            except Exception as e:
                                logger.warning(f"Failed to copy tensor: {e}")
                                continue
                    # Method 2: Try get_tensor with index (if we have a count)
                    if tensor_count > 0:
                        for i in range(tensor_count):
                            try:
                                tensor = reader.get_tensor(i)
                                writer.add_tensor(
                                    tensor.name,
                                    tensor.tensor_type,
                                    tensor.shape,
                                    tensor.data
                                )
                                tensors_copied += 1
                                if (i + 1) % 100 == 0:
                                    logger.debug(f"Copied {i + 1}/{tensor_count} tensors...")
                            except (IndexError, AttributeError, StopIteration):
                                # Reached end of tensors
                                break
                            except Exception as e:
                                logger.warning(f"Failed to copy tensor {i}: {e}")
                                continue
                    # Method 3: Try accessing tensors attribute (if iteration didn't work)
                    if tensors_copied == 0 and hasattr(reader, 'tensors'):
                        for tensor in reader.tensors:
                            try:
                                writer.add_tensor(
                                    tensor.name,
                                    tensor.tensor_type,
                                    tensor.shape,
                                    tensor.data
                                )
                                tensors_copied += 1
                            except Exception as e:
                                logger.warning(f"Failed to copy tensor: {e}")
                                continue
                    
                    # Method 4: Last resort - iterate with get_tensor until it fails
                    if tensors_copied == 0:
                        # Last resort: try iterating with get_tensor until it fails
                        i = 0
                        while True:
                            try:
                                tensor = reader.get_tensor(i)
                                writer.add_tensor(
                                    tensor.name,
                                    tensor.tensor_type,
                                    tensor.shape,
                                    tensor.data
                                )
                                tensors_copied += 1
                                i += 1
                                if i % 100 == 0:
                                    logger.debug(f"Copied {i} tensors...")
                            except (IndexError, AttributeError, StopIteration, KeyError):
                                # No more tensors
                                break
                            except Exception as e:
                                logger.warning(f"Failed to copy tensor {i}: {e}")
                                i += 1
                                continue
                
                except Exception as e:
                    logger.error(f"Error copying tensors: {e}")
                    raise Exception(f"Failed to copy tensors from source file: {e}")
                
                logger.info(f"Successfully copied {tensors_copied} tensors")
                
                if tensors_copied == 0:
                    logger.warning("No tensors were copied - file may be incomplete")
                
                # Copy existing metadata (excluding ones we'll replace)
                exclude_keys = {
                    'general.name', 'general.description',
                    'llama.system_prompt', 'chat_template.system_prompt',
                    'llama.persona_prompt', 'chat_template.persona_prompt'
                }
                
                logger.info("Copying existing metadata...")
                logger.debug("="*60)
                logger.debug("METADATA COPY OPERATION")
                logger.debug(f"  Reader type: {type(reader)}")
                logger.debug(f"  Reader module: {type(reader).__module__}")
                
                # Inspect reader object before attempting to get keys
                reader_info = inspect_object(reader, "GGUFReader (save operation)")
                logger.debug(f"Reader inspection:\n{reader_info}")
                
                # Check for keys method
                # Try to get keys using the correct API (reader.fields.keys())
                try:
                    logger.debug("Attempting to access reader.fields.keys()...")
                    
                    # Check if fields attribute exists
                    if hasattr(reader, 'fields'):
                        logger.debug(f"  reader.fields type: {type(reader.fields)}")
                        if hasattr(reader.fields, 'keys'):
                            keys_result = reader.fields.keys()
                            logger.debug(f"  reader.fields.keys() returned: {type(keys_result)}")
                            
                            # Convert to list
                            try:
                                keys = list(keys_result)
                                logger.info(f"Successfully retrieved {len(keys)} metadata keys to copy")
                                logger.debug(f"  First 10 keys: {keys[:10]}")
                            except Exception as e:
                                logger.error(f"Failed to convert fields.keys() result to list: {e}")
                                logger.error(f"  keys() result type: {type(keys_result)}")
                                logger.error(f"  keys() result repr: {repr(keys_result)}")
                                raise
                        else:
                            logger.error("  reader.fields exists but has no 'keys' method!")
                            raise AttributeError("'fields' object has no attribute 'keys'")
                    else:
                        logger.error("  reader has no 'fields' attribute!")
                        logger.error(f"  Available attributes: {[x for x in dir(reader) if not x.startswith('_')][:30]}")
                        raise AttributeError("'GGUFReader' object has no attribute 'fields'")
                        
                except AttributeError as e:
                    error_msg = str(e)
                    logger.error("="*60)
                    logger.error("CRITICAL ERROR: Cannot access reader.fields.keys()")
                    logger.error(f"  Error: {error_msg}")
                    logger.error(f"  Exception type: {type(e).__name__}")
                    logger.error(f"  Full traceback:\n{traceback.format_exc()}")
                    logger.error("="*60)
                    
                    # Try alternative methods to get metadata
                    logger.warning("Attempting alternative methods to access metadata...")
                    alternative_keys = []
                    
                    # Try direct iteration over fields if it's iterable
                    if hasattr(reader, 'fields') and hasattr(reader.fields, '__iter__'):
                        logger.debug("  Trying to iterate over reader.fields...")
                        try:
                            for key in reader.fields:
                                alternative_keys.append(key)
                            logger.info(f"  Found {len(alternative_keys)} keys via fields iteration")
                        except Exception as e2:
                            logger.warning(f"  fields iteration failed: {e2}")
                    
                    # Try get_field with known keys to build a list
                    if not alternative_keys and hasattr(reader, 'get_field'):
                        logger.debug("  Trying get_field with common keys...")
                        common_keys = [
                            'general.name', 'general.description',
                            'llama.system_prompt', 'chat_template.system_prompt',
                            'llama.persona_prompt', 'chat_template.persona_prompt',
                            'general.architecture', 'general.file_type',
                            'GGUF.version', 'GGUF.tensor_count', 'GGUF.kv_count'
                        ]
                        for key in common_keys:
                            try:
                                field = reader.get_field(key)
                                if field is not None:
                                    alternative_keys.append(key)
                            except:
                                pass
                        logger.info(f"  Found {len(alternative_keys)} keys via common keys test")
                    
                    if alternative_keys:
                        logger.info(f"Using {len(alternative_keys)} keys from alternative method")
                        keys = alternative_keys
                    else:
                        logger.error("All alternative methods failed. Cannot proceed with metadata copy.")
                        raise AttributeError(f"Cannot access metadata keys: {error_msg}")
                        
                except Exception as e:
                    logger.error(f"Unexpected error getting keys: {type(e).__name__}: {e}")
                    logger.error(f"  Full traceback:\n{traceback.format_exc()}")
                    raise
                
                # Copy metadata keys
                logger.info(f"Copying {len(keys)} metadata keys (excluding {len(exclude_keys)} keys)...")
                start_time = time.time()
                copied_count = 0
                skipped_count = 0
                failed_count = 0
                
                for i, key in enumerate(keys):
                    if key in exclude_keys:
                        skipped_count += 1
                        logger.debug(f"  [{i+1}/{len(keys)}] Skipping excluded key: {key}")
                        continue
                    
                    try:
                        logger.debug(f"  [{i+1}/{len(keys)}] Copying key: {key}")
                        
                        # Get value using the correct API
                        value = None
                        if hasattr(reader, 'get_value'):
                            # Try get_value method if it exists (for compatibility)
                            try:
                                value = reader.get_value(key)
                                logger.debug(f"    Got value via get_value() method")
                            except Exception as e:
                                logger.debug(f"    get_value() failed: {e}, trying get_field().contents()")
                                # Fall back to get_field().contents()
                                field = reader.get_field(key)
                                if field is not None:
                                    value = field.contents()
                                else:
                                    logger.warning(f"    Field not found for key: {key}")
                                    failed_count += 1
                                    continue
                        elif hasattr(reader, 'get_field'):
                            # Use get_field().contents() method
                            field = reader.get_field(key)
                            if field is not None:
                                value = field.contents()
                                logger.debug(f"    Got value via get_field().contents()")
                            else:
                                logger.warning(f"    Field not found for key: {key}")
                                failed_count += 1
                                continue
                        else:
                            logger.error(f"    No method to get value for key: {key}")
                            failed_count += 1
                            continue
                        
                        if value is None:
                            logger.warning(f"    Value is None for key: {key}")
                            failed_count += 1
                            continue
                        
                        value_type = type(value).__name__
                        logger.debug(f"    Value type: {value_type}")
                        
                        if isinstance(value, str):
                            writer.add_string(key, value)
                            logger.debug(f"    Added as string (length: {len(value)})")
                        elif isinstance(value, int):
                            if -2147483648 <= value <= 2147483647:
                                writer.add_int32(key, value)
                                logger.debug(f"    Added as int32: {value}")
                            else:
                                writer.add_int64(key, value)
                                logger.debug(f"    Added as int64: {value}")
                        elif isinstance(value, float):
                            writer.add_float32(key, value)
                            logger.debug(f"    Added as float32: {value}")
                        elif isinstance(value, bool):
                            writer.add_bool(key, value)
                            logger.debug(f"    Added as bool: {value}")
                        elif isinstance(value, list) and value and isinstance(value[0], str):
                            writer.add_array(key, value)
                            logger.debug(f"    Added as array (length: {len(value)})")
                        else:
                            logger.warning(f"    Unsupported type {value_type} for key '{key}', skipping")
                            skipped_count += 1
                            continue
                        
                        copied_count += 1
                        
                        if (i + 1) % 100 == 0:
                            elapsed = time.time() - start_time
                            logger.debug(f"    Progress: {i + 1}/{len(keys)} keys processed ({copied_count} copied, {failed_count} failed)")
                            
                    except Exception as e:
                        failed_count += 1
                        logger.warning(f"Failed to copy metadata key '{key}': {type(e).__name__}: {e}")
                        logger.debug(f"  Traceback for key '{key}':\n{traceback.format_exc()}")
                        continue
                
                elapsed = time.time() - start_time
                logger.info(f"Metadata copy complete:")
                logger.info(f"  Total keys: {len(keys)}")
                logger.info(f"  Copied: {copied_count}")
                logger.info(f"  Skipped (excluded): {skipped_count}")
                logger.info(f"  Failed: {failed_count}")
                logger.info(f"  Time taken: {elapsed:.3f}s")
                logger.debug("="*60)
                
                # Add/update custom metadata
                logger.info("Adding new/updated metadata...")
                if name:
                    writer.add_string('general.name', str(name))
                    writer.add_string('general.description', str(name))
                
                if system_prompt:
                    writer.add_string('llama.system_prompt', system_prompt)
                    writer.add_string('chat_template.system_prompt', system_prompt)
                
                if persona_prompt:
                    writer.add_string('llama.persona_prompt', persona_prompt)
                    writer.add_string('chat_template.persona_prompt', persona_prompt)
                
                if custom_metadata:
                    for key, value in custom_metadata.items():
                        try:
                            if isinstance(value, str):
                                writer.add_string(key, value)
                            elif isinstance(value, int):
                                if -2147483648 <= value <= 2147483647:
                                    writer.add_int32(key, value)
                                else:
                                    writer.add_int64(key, value)
                            elif isinstance(value, float):
                                writer.add_float32(key, value)
                            elif isinstance(value, bool):
                                writer.add_bool(key, value)
                            elif isinstance(value, list) and value and isinstance(value[0], str):
                                writer.add_array(key, value)
                        except Exception as e:
                            logger.warning(f"Failed to add custom metadata '{key}': {e}")
                            continue
                
                # Write the file
                logger.info("Writing GGUF file...")
                writer.write_header_to_file()
                writer.write_kv_data_to_file()
                writer.write_tensors_to_file()
                writer.close()
                
                logger.info(f"Successfully saved to: {output_path}")
                return str(output_path)
                
            except MemoryError as e:
                error_msg = (
                    f"Out of memory while saving.\n\n"
                    f"File is too large ({self.file_path.stat().st_size / (1024**3):.2f} GB).\n"
                    f"This operation requires significant RAM.\n\n"
                    f"Try:\n"
                    f"1. Free up RAM\n"
                    f"2. Use a machine with more memory\n"
                    f"3. Use CLI tool for large files"
                )
                logger.error(f"MemoryError during save: {error_msg}")
                raise MemoryError(error_msg)
            except Exception as e:
                error_msg = f"Failed to save GGUF file: {e}"
                logger.error(f"Error during save: {error_msg}\n{traceback.format_exc()}")
                raise Exception(error_msg)


class ModernColors:
    """Modern color scheme for the application."""
    BG_PRIMARY = "#1e1e1e"
    BG_SECONDARY = "#252525"
    BG_TERTIARY = "#2d2d2d"
    FG_PRIMARY = "#ffffff"
    FG_SECONDARY = "#cccccc"
    ACCENT = "#0078d4"
    ACCENT_HOVER = "#106ebe"
    SUCCESS = "#00ff00"
    WARNING = "#ffaa00"
    ERROR = "#ff4444"
    BORDER = "#3a3a3a"


class EasyEditorPanel:
    """Easy mode editor with simple, user-friendly fields."""
    
    def __init__(self, parent, editor_app):
        self.parent = parent
        self.app = editor_app
        self.frame = ttk.Frame(parent)
        self.setup_ui()
    
    def setup_ui(self):
        """Create the easy editor UI."""
        # Title
        title_label = tk.Label(
            self.frame,
            text="✨ Easy Editor Mode",
            font=("Segoe UI", 18, "bold"),
            bg=ModernColors.BG_PRIMARY,
            fg=ModernColors.ACCENT
        )
        title_label.pack(pady=(20, 30))
        
        # Model Name Section
        name_frame = ttk.LabelFrame(self.frame, text="🏷️  Model Name", padding=15)
        name_frame.pack(fill="x", padx=20, pady=10)
        
        self.name_var = tk.StringVar()
        name_entry = ttk.Entry(name_frame, textvariable=self.name_var, font=("Segoe UI", 11), width=60)
        name_entry.pack(fill="x")
        name_entry.insert(0, "Enter custom model name...")
        
        # System Prompt Section
        system_frame = ttk.LabelFrame(self.frame, text="⚙️  System Prompt", padding=15)
        system_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        self.system_text = scrolledtext.ScrolledText(
            system_frame,
            wrap=tk.WORD,
            font=("Segoe UI", 10),
            height=8,
            bg=ModernColors.BG_TERTIARY,
            fg=ModernColors.FG_PRIMARY,
            insertbackground=ModernColors.FG_PRIMARY
        )
        self.system_text.pack(fill="both", expand=True)
        
        # Persona Prompt Section
        persona_frame = ttk.LabelFrame(self.frame, text="👤  Persona Prompt", padding=15)
        persona_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        self.persona_text = scrolledtext.ScrolledText(
            persona_frame,
            wrap=tk.WORD,
            font=("Segoe UI", 10),
            height=8,
            bg=ModernColors.BG_TERTIARY,
            fg=ModernColors.FG_PRIMARY,
            insertbackground=ModernColors.FG_PRIMARY
        )
        self.persona_text.pack(fill="both", expand=True)
        
        # Character counters
        counter_frame = ttk.Frame(self.frame)
        counter_frame.pack(fill="x", padx=20, pady=5)
        
        self.system_counter = tk.Label(
            counter_frame,
            text="System: 0 characters",
            font=("Segoe UI", 9),
            bg=ModernColors.BG_PRIMARY,
            fg=ModernColors.FG_SECONDARY
        )
        self.system_counter.pack(side="left")
        
        self.persona_counter = tk.Label(
            counter_frame,
            text="Persona: 0 characters",
            font=("Segoe UI", 9),
            bg=ModernColors.BG_PRIMARY,
            fg=ModernColors.FG_SECONDARY
        )
        self.persona_counter.pack(side="right")
        
        # Update counters on text change
        self.system_text.bind("<KeyRelease>", self.update_counters)
        self.persona_text.bind("<KeyRelease>", self.update_counters)
    
    def update_counters(self, event=None):
        """Update character counters."""
        system_len = len(self.system_text.get("1.0", tk.END)) - 1
        persona_len = len(self.persona_text.get("1.0", tk.END)) - 1
        self.system_counter.config(text=f"System: {system_len} characters")
        self.persona_counter.config(text=f"Persona: {persona_len} characters")
    
    def load_data(self, editor):
        """Load data from GGUF editor into easy mode fields."""
        name = editor.get_value('general.name', '')
        if not name:
            name = editor.get_value('general.description', '')
        
        system = editor.get_value('llama.system_prompt', '')
        if not system:
            system = editor.get_value('chat_template.system_prompt', '')
        
        persona = editor.get_value('llama.persona_prompt', '')
        if not persona:
            persona = editor.get_value('chat_template.persona_prompt', '')
        
        self.name_var.set(str(name) if name else "")
        self.system_text.delete("1.0", tk.END)
        self.system_text.insert("1.0", str(system) if system else "")
        self.persona_text.delete("1.0", tk.END)
        self.persona_text.insert("1.0", str(persona) if persona else "")
        self.update_counters()
    
    def get_data(self):
        """Get data from easy mode fields."""
        return {
            'name': self.name_var.get().strip() or None,
            'system_prompt': self.system_text.get("1.0", tk.END).strip() or None,
            'persona_prompt': self.persona_text.get("1.0", tk.END).strip() or None
        }


class AdvancedEditorPanel:
    """Advanced mode editor with full metadata tree view and editing."""
    
    def __init__(self, parent, editor_app):
        self.parent = parent
        self.app = editor_app
        self.frame = ttk.Frame(parent)
        self.metadata_dict = {}
        self.setup_ui()
    
    def setup_ui(self):
        """Create the advanced editor UI."""
        # Title
        title_label = tk.Label(
            self.frame,
            text="🔧 Advanced Editor Mode",
            font=("Segoe UI", 18, "bold"),
            bg=ModernColors.BG_PRIMARY,
            fg=ModernColors.ACCENT
        )
        title_label.pack(pady=(20, 10))
        
        # Toolbar
        toolbar = ttk.Frame(self.frame)
        toolbar.pack(fill="x", padx=20, pady=10)
        
        refresh_btn = ttk.Button(
            toolbar,
            text="🔄 Refresh Metadata",
            command=self.refresh_metadata
        )
        refresh_btn.pack(side="left", padx=5)
        
        add_btn = ttk.Button(
            toolbar,
            text="➕ Add Key",
            command=self.add_metadata_key
        )
        add_btn.pack(side="left", padx=5)
        
        delete_btn = ttk.Button(
            toolbar,
            text="➖ Delete Selected",
            command=self.delete_selected
        )
        delete_btn.pack(side="left", padx=5)
        
        # Search bar
        search_frame = ttk.Frame(self.frame)
        search_frame.pack(fill="x", padx=20, pady=5)
        
        ttk.Label(search_frame, text="🔍 Search:").pack(side="left", padx=5)
        self.search_var = tk.StringVar()
        self.search_var.trace("w", self.filter_metadata)
        search_entry = ttk.Entry(search_frame, textvariable=self.search_var, width=40)
        search_entry.pack(side="left", padx=5, fill="x", expand=True)
        
        # Metadata tree
        tree_frame = ttk.Frame(self.frame)
        tree_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Treeview with scrollbars
        tree_scroll_y = ttk.Scrollbar(tree_frame, orient="vertical")
        tree_scroll_x = ttk.Scrollbar(tree_frame, orient="horizontal")
        
        self.tree = ttk.Treeview(
            tree_frame,
            columns=("Value", "Type"),
            show="tree headings",
            yscrollcommand=tree_scroll_y.set,
            xscrollcommand=tree_scroll_x.set,
            selectmode="browse"
        )
        
        tree_scroll_y.config(command=self.tree.yview)
        tree_scroll_x.config(command=self.tree.xview)
        
        # Configure columns
        self.tree.heading("#0", text="Key", anchor="w")
        self.tree.heading("Value", text="Value", anchor="w")
        self.tree.heading("Type", text="Type", anchor="w")
        
        self.tree.column("#0", width=300, minwidth=200)
        self.tree.column("Value", width=400, minwidth=200)
        self.tree.column("Type", width=100, minwidth=80)
        
        self.tree.pack(side="left", fill="both", expand=True)
        tree_scroll_y.pack(side="right", fill="y")
        tree_scroll_x.pack(side="bottom", fill="x")
        
        # Bind double-click to edit
        self.tree.bind("<Double-1>", self.edit_selected_item)
        
        # Edit panel
        edit_frame = ttk.LabelFrame(self.frame, text="✏️  Edit Selected Item", padding=15)
        edit_frame.pack(fill="x", padx=20, pady=10)
        
        edit_inner = ttk.Frame(edit_frame)
        edit_inner.pack(fill="x")
        
        ttk.Label(edit_inner, text="Key:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.edit_key_var = tk.StringVar()
        edit_key_entry = ttk.Entry(edit_inner, textvariable=self.edit_key_var, width=50)
        edit_key_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        
        ttk.Label(edit_inner, text="Value:").grid(row=1, column=0, sticky="nw", padx=5, pady=5)
        self.edit_value_text = scrolledtext.ScrolledText(
            edit_inner,
            wrap=tk.WORD,
            height=4,
            width=50,
            font=("Consolas", 9)
        )
        self.edit_value_text.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
        
        ttk.Label(edit_inner, text="Type:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.edit_type_var = tk.StringVar(value="string")
        type_combo = ttk.Combobox(
            edit_inner,
            textvariable=self.edit_type_var,
            values=["string", "int", "float", "bool", "array"],
            state="readonly",
            width=47
        )
        type_combo.grid(row=2, column=1, sticky="w", padx=5, pady=5)
        
        edit_inner.grid_columnconfigure(1, weight=1)
        
        save_item_btn = ttk.Button(
            edit_frame,
            text="💾 Save Item",
            command=self.save_edited_item
        )
        save_item_btn.pack(pady=10)
    
    def load_metadata(self, metadata_dict):
        """Load metadata into the tree view."""
        self.metadata_dict = metadata_dict
        self.refresh_tree()
    
    def refresh_tree(self, filter_text=""):
        """Refresh the tree view with current metadata."""
        # Clear existing items
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Add items
        filter_lower = filter_text.lower()
        for key, value in sorted(self.metadata_dict.items()):
            if filter_text and filter_lower not in key.lower() and filter_lower not in str(value).lower():
                continue
            
            value_str = str(value)
            if len(value_str) > 100:
                value_str = value_str[:97] + "..."
            
            value_type = type(value).__name__
            if isinstance(value, list):
                value_type = f"array[{len(value)}]"
            
            self.tree.insert("", "end", text=key, values=(value_str, value_type))
    
    def filter_metadata(self, *args):
        """Filter metadata tree by search text."""
        search_text = self.search_var.get()
        self.refresh_tree(search_text)
    
    def refresh_metadata(self):
        """Refresh metadata from the loaded file."""
        if self.app.editor:
            try:
                metadata = self.app.editor.load()
                self.load_metadata(metadata)
                messagebox.showinfo("Success", "Metadata refreshed successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to refresh metadata: {e}")
    
    def edit_selected_item(self, event=None):
        """Edit the selected item in the tree."""
        selection = self.tree.selection()
        if not selection:
            return
        
        item = self.tree.item(selection[0])
        key = item['text']
        value = self.metadata_dict.get(key, "")
        
        self.edit_key_var.set(key)
        self.edit_value_text.delete("1.0", tk.END)
        self.edit_value_text.insert("1.0", str(value))
        
        # Determine type
        value_type = type(value).__name__
        if isinstance(value, list):
            self.edit_type_var.set("array")
            self.edit_value_text.insert("1.0", json.dumps(value, indent=2))
        elif isinstance(value, bool):
            self.edit_type_var.set("bool")
        elif isinstance(value, int):
            self.edit_type_var.set("int")
        elif isinstance(value, float):
            self.edit_type_var.set("float")
        else:
            self.edit_type_var.set("string")
    
    def save_edited_item(self):
        """Save the edited item to metadata."""
        key = self.edit_key_var.get().strip()
        if not key:
            messagebox.showwarning("Warning", "Key cannot be empty!")
            return
        
        value_text = self.edit_value_text.get("1.0", tk.END).strip()
        value_type = self.edit_type_var.get()
        
        try:
            if value_type == "int":
                value = int(value_text)
            elif value_type == "float":
                value = float(value_text)
            elif value_type == "bool":
                value = value_text.lower() in ("true", "1", "yes", "on")
            elif value_type == "array":
                value = json.loads(value_text)
            else:
                value = value_text
            
            self.metadata_dict[key] = value
            self.refresh_tree(self.search_var.get())
            messagebox.showinfo("Success", f"Saved: {key}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save item: {e}")
    
    def add_metadata_key(self):
        """Add a new metadata key."""
        dialog = tk.Toplevel(self.app.root)
        dialog.title("Add Metadata Key")
        dialog.geometry("500x300")
        dialog.configure(bg=ModernColors.BG_PRIMARY)
        
        ttk.Label(dialog, text="Key:").pack(pady=5)
        key_var = tk.StringVar()
        ttk.Entry(dialog, textvariable=key_var, width=60).pack(pady=5)
        
        ttk.Label(dialog, text="Value:").pack(pady=5)
        value_text = scrolledtext.ScrolledText(dialog, height=8, width=60)
        value_text.pack(pady=5, padx=10, fill="both", expand=True)
        
        def save_new_key():
            key = key_var.get().strip()
            if not key:
                messagebox.showwarning("Warning", "Key cannot be empty!")
                return
            
            value = value_text.get("1.0", tk.END).strip()
            self.metadata_dict[key] = value
            self.refresh_tree(self.search_var.get())
            dialog.destroy()
            messagebox.showinfo("Success", f"Added key: {key}")
        
        ttk.Button(dialog, text="Add", command=save_new_key).pack(pady=10)
    
    def delete_selected(self):
        """Delete the selected metadata key."""
        selection = self.tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select an item to delete!")
            return
        
        item = self.tree.item(selection[0])
        key = item['text']
        
        if messagebox.askyesno("Confirm Delete", f"Delete key '{key}'?"):
            del self.metadata_dict[key]
            self.refresh_tree(self.search_var.get())
            messagebox.showinfo("Success", f"Deleted: {key}")
    
    def get_metadata(self):
        """Get the current metadata dictionary."""
        return self.metadata_dict.copy()


class GGUFEditorApp:
    """Main application class for GGUF Editor."""
    
    def __init__(self, root):
        self.root = root
        self.editor = None
        self.current_file = None
        self.is_easy_mode = True
        
        self.setup_window()
        self.setup_styles()
        self.setup_ui()
    
    def setup_window(self):
        """Configure the main window."""
        self.root.title("✨ World-Class GGUF Editor")
        self.root.geometry("1200x800")
        self.root.configure(bg=ModernColors.BG_PRIMARY)
        
        # Center window on screen
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f"{width}x{height}+{x}+{y}")
    
    def setup_styles(self):
        """Configure modern styling."""
        style = ttk.Style()
        style.theme_use("clam")
        
        # Configure styles
        style.configure("TFrame", background=ModernColors.BG_PRIMARY)
        style.configure("TLabelFrame", background=ModernColors.BG_PRIMARY, foreground=ModernColors.FG_PRIMARY)
        style.configure("TLabelFrame.Label", background=ModernColors.BG_PRIMARY, foreground=ModernColors.ACCENT)
        
        style.configure("TButton",
                       background=ModernColors.ACCENT,
                       foreground=ModernColors.FG_PRIMARY,
                       borderwidth=0,
                       focuscolor="none",
                       padding=10)
        style.map("TButton",
                 background=[("active", ModernColors.ACCENT_HOVER),
                           ("pressed", ModernColors.ACCENT_HOVER)])
        
        style.configure("TEntry",
                       fieldbackground=ModernColors.BG_TERTIARY,
                       foreground=ModernColors.FG_PRIMARY,
                       borderwidth=1)
        
        style.configure("Treeview",
                       background=ModernColors.BG_SECONDARY,
                       foreground=ModernColors.FG_PRIMARY,
                       fieldbackground=ModernColors.BG_SECONDARY,
                       borderwidth=1)
        style.configure("Treeview.Heading",
                       background=ModernColors.BG_TERTIARY,
                       foreground=ModernColors.FG_PRIMARY,
                       borderwidth=1)
    
    def setup_ui(self):
        """Create the main UI."""
        # Header bar
        header = tk.Frame(self.root, bg=ModernColors.BG_SECONDARY, height=80)
        header.pack(fill="x")
        header.pack_propagate(False)
        
        # Title
        title_label = tk.Label(
            header,
            text="✨ World-Class GGUF Editor",
            font=("Segoe UI", 24, "bold"),
            bg=ModernColors.BG_SECONDARY,
            fg=ModernColors.ACCENT
        )
        title_label.pack(side="left", padx=20, pady=20)
        
        # File info
        self.file_label = tk.Label(
            header,
            text="No file loaded",
            font=("Segoe UI", 10),
            bg=ModernColors.BG_SECONDARY,
            fg=ModernColors.FG_SECONDARY
        )
        self.file_label.pack(side="left", padx=20, pady=20)
        
        # Toolbar
        toolbar = tk.Frame(header, bg=ModernColors.BG_SECONDARY)
        toolbar.pack(side="right", padx=20, pady=20)
        
        load_btn = ttk.Button(toolbar, text="📂 Load GGUF File", command=self.load_file)
        load_btn.pack(side="left", padx=5)
        
        self.mode_btn = ttk.Button(
            toolbar,
            text="🔧 Switch to Advanced Mode",
            command=self.toggle_mode
        )
        self.mode_btn.pack(side="left", padx=5)
        
        save_btn = ttk.Button(toolbar, text="💾 Save Changes", command=self.save_file)
        save_btn.pack(side="left", padx=5)
        
        backup_btn = ttk.Button(toolbar, text="📋 Create Backup", command=self.create_backup)
        backup_btn.pack(side="left", padx=5)
        
        # Main content area
        self.content_frame = ttk.Frame(self.root)
        self.content_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Initialize editors
        self.easy_editor = EasyEditorPanel(self.content_frame, self)
        self.advanced_editor = AdvancedEditorPanel(self.content_frame, self)
        
        # Show easy mode by default
        self.show_easy_mode()
    
    def show_debug_window(self, error_details: str):
        """Show a detailed debug window with error information."""
        debug_window = tk.Toplevel(self.root)
        debug_window.title("🐛 Debug Information - GGUF Editor")
        debug_window.geometry("900x700")
        debug_window.configure(bg=ModernColors.BG_PRIMARY)
        
        # Make window resizable
        debug_window.resizable(True, True)
        
        # Title
        title_label = tk.Label(
            debug_window,
            text="🐛 Error Debug Information",
            font=("Segoe UI", 16, "bold"),
            bg=ModernColors.BG_PRIMARY,
            fg=ModernColors.ERROR
        )
        title_label.pack(pady=10)
        
        # Log file location
        log_label = tk.Label(
            debug_window,
            text=f"📋 Full log saved to: {LOG_FILE}",
            font=("Segoe UI", 9),
            bg=ModernColors.BG_PRIMARY,
            fg=ModernColors.FG_SECONDARY
        )
        log_label.pack(pady=5)
        
        # Error details text area
        text_frame = ttk.Frame(debug_window)
        text_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        error_text = scrolledtext.ScrolledText(
            text_frame,
            wrap=tk.WORD,
            font=("Consolas", 10),
            bg=ModernColors.BG_TERTIARY,
            fg=ModernColors.FG_PRIMARY,
            insertbackground=ModernColors.FG_PRIMARY
        )
        error_text.pack(fill="both", expand=True)
        error_text.insert("1.0", error_details)
        error_text.config(state="disabled")
        
        # Buttons
        button_frame = ttk.Frame(debug_window)
        button_frame.pack(pady=10)
        
        copy_btn = ttk.Button(
            button_frame,
            text="📋 Copy Error Details",
            command=lambda: self.copy_to_clipboard(debug_window, error_details)
        )
        copy_btn.pack(side="left", padx=5)
        
        open_log_btn = ttk.Button(
            button_frame,
            text="📂 Open Log File",
            command=lambda: self.open_log_file()
        )
        open_log_btn.pack(side="left", padx=5)
        
        close_btn = ttk.Button(
            button_frame,
            text="Close",
            command=debug_window.destroy
        )
        close_btn.pack(side="left", padx=5)
    
    def copy_to_clipboard(self, window, text):
        """Copy text to clipboard."""
        window.clipboard_clear()
        window.clipboard_append(text)
        messagebox.showinfo("Copied", "Error details copied to clipboard!")
    
    def open_log_file(self):
        """Open the log file in system default editor."""
        import subprocess
        import platform
        
        try:
            if platform.system() == 'Windows':
                os.startfile(LOG_FILE)
            elif platform.system() == 'Darwin':  # macOS
                subprocess.run(['open', str(LOG_FILE)])
            else:  # Linux
                subprocess.run(['xdg-open', str(LOG_FILE)])
        except Exception as e:
            messagebox.showerror("Error", f"Could not open log file:\n{e}\n\nLog location: {LOG_FILE}")
    
    def show_loading_progress(self, file_path: str):
        """Show a progress window for loading large files."""
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Loading GGUF File...")
        progress_window.geometry("500x150")
        progress_window.configure(bg=ModernColors.BG_PRIMARY)
        progress_window.transient(self.root)
        progress_window.grab_set()
        
        # Center the window
        progress_window.update_idletasks()
        x = (progress_window.winfo_screenwidth() // 2) - (500 // 2)
        y = (progress_window.winfo_screenheight() // 2) - (150 // 2)
        progress_window.geometry(f"500x150+{x}+{y}")
        
        # Prevent closing during load
        progress_window.protocol("WM_DELETE_WINDOW", lambda: None)
        
        # Title
        title_label = tk.Label(
            progress_window,
            text="Loading GGUF File...",
            font=("Segoe UI", 14, "bold"),
            bg=ModernColors.BG_PRIMARY,
            fg=ModernColors.ACCENT
        )
        title_label.pack(pady=10)
        
        # File name
        file_name = os.path.basename(file_path)
        file_label = tk.Label(
            progress_window,
            text=file_name,
            font=("Segoe UI", 10),
            bg=ModernColors.BG_PRIMARY,
            fg=ModernColors.FG_SECONDARY
        )
        file_label.pack(pady=5)
        
        # Progress bar
        progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(
            progress_window,
            variable=progress_var,
            maximum=100,
            length=450,
            mode='determinate'
        )
        progress_bar.pack(pady=10)
        
        # Status label
        status_label = tk.Label(
            progress_window,
            text="Initializing...",
            font=("Segoe UI", 9),
            bg=ModernColors.BG_PRIMARY,
            fg=ModernColors.FG_SECONDARY
        )
        status_label.pack(pady=5)
        
        # Store references
        progress_window.progress_var = progress_var
        progress_window.status_label = status_label
        
        return progress_window
    
    def show_saving_progress(self, file_path: str):
        """Show a progress window for saving files."""
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Saving GGUF File...")
        progress_window.geometry("500x150")
        progress_window.configure(bg=ModernColors.BG_PRIMARY)
        progress_window.transient(self.root)
        progress_window.grab_set()
        
        # Center the window
        progress_window.update_idletasks()
        x = (progress_window.winfo_screenwidth() // 2) - (500 // 2)
        y = (progress_window.winfo_screenheight() // 2) - (150 // 2)
        progress_window.geometry(f"500x150+{x}+{y}")
        
        # Prevent closing during save
        progress_window.protocol("WM_DELETE_WINDOW", lambda: None)
        
        # Title
        title_label = tk.Label(
            progress_window,
            text="Saving GGUF File...",
            font=("Segoe UI", 14, "bold"),
            bg=ModernColors.BG_PRIMARY,
            fg=ModernColors.ACCENT
        )
        title_label.pack(pady=10)
        
        # File name
        file_name = os.path.basename(file_path) if file_path else "Unknown"
        file_label = tk.Label(
            progress_window,
            text=file_name,
            font=("Segoe UI", 10),
            bg=ModernColors.BG_PRIMARY,
            fg=ModernColors.FG_SECONDARY
        )
        file_label.pack(pady=5)
        
        # Progress bar (indeterminate mode for save)
        progress_bar = ttk.Progressbar(
            progress_window,
            maximum=100,
            length=450,
            mode='indeterminate'
        )
        progress_bar.pack(pady=10)
        progress_bar.start(10)  # Start animated progress bar
        
        # Status label
        status_label = tk.Label(
            progress_window,
            text="Saving file... Please wait...",
            font=("Segoe UI", 9),
            bg=ModernColors.BG_PRIMARY,
            fg=ModernColors.FG_SECONDARY
        )
        status_label.pack(pady=5)
        
        # Store references
        progress_window.progress_bar = progress_bar
        progress_window.status_label = status_label
        
        return progress_window
    
    def update_progress(self, progress_window, status: str, value: int):
        """Update the progress window."""
        if progress_window and progress_window.winfo_exists():
            progress_window.progress_var.set(value)
            progress_window.status_label.config(text=status)
            self.root.update_idletasks()
    
    def update_save_progress(self, progress_window, status: str, value: int = None):
        """Update the save progress window."""
        if progress_window and progress_window.winfo_exists():
            if hasattr(progress_window, 'status_label'):
                progress_window.status_label.config(text=status)
            self.root.update_idletasks()
    
    def close_progress(self, progress_window):
        """Close the progress window."""
        if progress_window and progress_window.winfo_exists():
            if hasattr(progress_window, 'progress_bar'):
                progress_window.progress_bar.stop()
            progress_window.destroy()
    
    def _show_save_error(self, error: Exception, file_path: str):
        """Show save error dialog."""
        error_type = type(error).__name__
        error_msg = str(error)
        logger.error("="*80)
        logger.error(f"SAVE OPERATION FAILED: {error_type}")
        logger.error(f"  Error message: {error_msg}")
        logger.error(f"  Full traceback:\n{traceback.format_exc()}")
        logger.error("="*80)
        
        # Show detailed error in debug window
        full_error = f"""Save Operation Error Details:
{'='*60}
ERROR TYPE: {error_type}
{'='*60}

Error Message: {error_msg}

File: {self.current_file if self.current_file else 'N/A'}
Output Path: {file_path if file_path else 'N/A'}
Mode: {'Easy' if self.is_easy_mode else 'Advanced'}

Full Traceback:
{traceback.format_exc()}

Log File: {LOG_FILE}
"""
        messagebox.showerror("Error", f"Failed to save file:\n{error}\n\nClick OK to view detailed debug information.")
        self.show_debug_window(full_error)
    
    @log_function_call
    def load_file(self):
        """Load a GGUF file with comprehensive error handling and debugging."""
        logger.info("="*80)
        logger.info("FILE LOAD OPERATION: Starting")
        logger.info("="*80)
        
        logger.debug("Opening file dialog...")
        file_path = filedialog.askopenfilename(
            title="Select GGUF File",
            filetypes=[("GGUF files", "*.gguf"), ("All files", "*.*")]
        )
        
        if not file_path:
            logger.info("File selection cancelled by user")
            return
        
        logger.info(f"Selected file: {file_path}")
        logger.debug(f"  Absolute path: {os.path.abspath(file_path)}")
        logger.debug(f"  Path exists: {os.path.exists(file_path)}")
        logger.debug(f"  Is file: {os.path.isfile(file_path) if os.path.exists(file_path) else 'N/A'}")
        
        # Validate file exists
        if not os.path.exists(file_path):
            error_msg = f"File does not exist: {file_path}"
            logger.error(error_msg)
            messagebox.showerror("File Not Found", error_msg)
            return
        
        # Check file permissions
        if not os.access(file_path, os.R_OK):
            error_msg = f"No read permission for file: {file_path}"
            logger.error(error_msg)
            messagebox.showerror("Permission Denied", error_msg)
            return
        
        # Check file size
        try:
            file_size = os.path.getsize(file_path)
            file_size_mb = file_size / (1024 * 1024)
            file_size_gb = file_size / (1024 * 1024 * 1024)
            logger.info(f"File size: {file_size} bytes ({file_size_mb:.2f} MB / {file_size_gb:.2f} GB)")
            
            if file_size == 0:
                error_msg = "File is empty (0 bytes)"
                logger.error(error_msg)
                messagebox.showerror("Invalid File", error_msg)
                return
            
            # Warn about very large files (>2GB)
            if file_size_gb > 2.0:
                response = messagebox.askyesno(
                    "Large File Warning",
                    f"⚠️ This file is very large ({file_size_gb:.2f} GB).\n\n"
                    f"Loading may take time and use significant memory.\n\n"
                    f"Continue loading?\n\n"
                    f"Tip: We'll only load metadata (not model weights),\n"
                    f"so it should be safe even for large files."
                )
                if not response:
                    logger.info("User cancelled loading large file")
                    return
            
            # Warn about large files (>500MB)
            elif file_size_mb > 500:
                response = messagebox.askyesno(
                    "Large File Notice",
                    f"This file is large ({file_size_mb:.0f} MB).\n\n"
                    f"Loading may take a moment.\n\n"
                    f"Continue?"
                )
                if not response:
                    logger.info("User cancelled loading large file")
                    return
                    
        except Exception as e:
            error_msg = f"Could not get file size: {e}"
            logger.error(error_msg)
            messagebox.showerror("File Error", error_msg)
            return
        
        # Check if it's a valid GGUF file (basic check)
        try:
            with open(file_path, 'rb') as f:
                magic = f.read(4)
                if magic != b'GGUF':
                    logger.warning(f"File magic bytes are '{magic}', expected 'GGUF'")
        except Exception as e:
            logger.warning(f"Could not read magic bytes: {e}")
        
        # Try to load the file
        try:
            # Show progress dialog for large files
            progress_window = None
            if file_size_mb > 100:
                progress_window = self.show_loading_progress(file_path)
                self.root.update()
            
            logger.info("Creating GGUFEditor instance...")
            self.editor = GGUFEditor(file_path)
            logger.info("GGUFEditor instance created successfully")
            
            if progress_window:
                self.update_progress(progress_window, "Loading metadata keys...", 30)
            
            logger.info("Loading metadata...")
            
            # Use optimized loading for large files - only load metadata keys
            try:
                # Try metadata-only first (most memory efficient)
                if hasattr(self.editor, 'load_metadata_only'):
                    logger.info("Using metadata-only loading method...")
                    metadata = self.editor.load_metadata_only()
                    logger.info(f"Metadata loaded successfully (metadata-only mode). Found {len(metadata)} metadata keys")
                else:
                    # Try standard load
                    logger.info("Using standard load method...")
                    metadata = self.editor.load()
                    logger.info(f"Metadata loaded successfully. Found {len(metadata)} metadata keys")
            except (MemoryError, AttributeError) as e:
                # If both fail, try using low-level handler directly
                logger.warning(f"Standard methods failed ({e}), trying low-level handler...")
                try:
                    from gguf_handler import GGUFHandler
                    handler = GGUFHandler(file_path)
                    metadata = handler.read_metadata()
                    # Update editor metadata
                    self.editor.metadata = metadata
                    logger.info(f"Metadata loaded via low-level handler. Found {len(metadata)} metadata keys")
                except Exception as e2:
                    logger.error(f"All loading methods failed: {e2}")
                    raise MemoryError(
                        f"Unable to load file due to memory constraints.\n"
                        f"File size: {file_size_gb:.2f} GB\n"
                        f"Try: (1) Free up RAM, (2) Use a machine with more memory, or (3) Edit metadata directly in file."
                    )
            
            if progress_window:
                self.update_progress(progress_window, "Processing metadata...", 70)
            
            self.current_file = file_path
            
            # Update file label
            file_name = os.path.basename(file_path)
            # file_size, file_size_mb, file_size_gb should be available from earlier in the function
            try:
                size_display = f"{file_size_gb:.2f} GB" if file_size_gb >= 1.0 else f"{file_size_mb:.0f} MB"
            except NameError:
                # Fallback if variables not in scope
                file_size = os.path.getsize(file_path)
                file_size_mb = file_size / (1024 * 1024)
                file_size_gb = file_size / (1024 * 1024 * 1024)
                size_display = f"{file_size_gb:.2f} GB" if file_size_gb >= 1.0 else f"{file_size_mb:.0f} MB"
            self.file_label.config(
                text=f"📄 {file_name} ({size_display})",
                fg=ModernColors.SUCCESS
            )
            
            if progress_window:
                self.update_progress(progress_window, "Loading into editor...", 90)
            
            # Load data into current editor
            logger.info(f"Loading data into {'Easy' if self.is_easy_mode else 'Advanced'} mode editor...")
            if self.is_easy_mode:
                self.easy_editor.load_data(self.editor)
            else:
                self.advanced_editor.load_metadata(metadata)
            
            if progress_window:
                self.update_progress(progress_window, "Complete!", 100)
                self.root.after(300, lambda: self.close_progress(progress_window))
            
            logger.info("File loaded successfully!")
            if not progress_window:
                messagebox.showinfo("Success", f"Loaded: {file_name}\n\nMetadata keys: {len(metadata)}")
            
        except ImportError as e:
            error_msg = f"Missing required library:\n{e}\n\nPlease install: pip install gguf"
            logger.error(f"ImportError: {error_msg}\n{traceback.format_exc()}")
            
            full_error = f"""Import Error Details:
{'='*60}
ERROR: Missing Required Library
{'='*60}

Error Message: {str(e)}

This usually means the 'gguf' library is not installed.

SOLUTION:
1. Open a terminal
2. Run: pip install gguf
3. Restart this application

Full Traceback:
{traceback.format_exc()}

Log File: {LOG_FILE}
"""
            
            messagebox.showerror("Missing Library", error_msg)
            self.show_debug_window(full_error)
            
        except FileNotFoundError as e:
            error_msg = f"File not found:\n{e}"
            logger.error(f"FileNotFoundError: {error_msg}\n{traceback.format_exc()}")
            messagebox.showerror("File Not Found", error_msg)
            
        except PermissionError as e:
            error_msg = f"Permission denied:\n{e}\n\nCheck file permissions."
            logger.error(f"PermissionError: {error_msg}\n{traceback.format_exc()}")
            messagebox.showerror("Permission Denied", error_msg)
            
        except ValueError as e:
            error_msg = f"Invalid GGUF file:\n{e}"
            logger.error(f"ValueError: {error_msg}\n{traceback.format_exc()}")
            
            full_error = f"""File Validation Error:
{'='*60}
ERROR: Invalid GGUF File Format
{'='*60}

Error Message: {str(e)}

This file may be corrupted or not a valid GGUF file.

File: {file_path}
Size: {file_size} bytes

Full Traceback:
{traceback.format_exc()}

Log File: {LOG_FILE}
"""
            
            messagebox.showerror("Invalid File", error_msg)
            self.show_debug_window(full_error)
            
        except MemoryError as e:
            error_msg = f"Out of memory:\n{e}\n\nFile may be too large."
            logger.error(f"MemoryError: {error_msg}\n{traceback.format_exc()}")
            messagebox.showerror("Out of Memory", error_msg)
            
        except Exception as e:
            # Catch-all for any other errors
            error_type = type(e).__name__
            error_msg = f"Unexpected error ({error_type}):\n{str(e)}"
            logger.error(f"{error_type}: {error_msg}\n{traceback.format_exc()}")
            
            full_error = f"""Unexpected Error Details:
{'='*60}
ERROR TYPE: {error_type}
{'='*60}

Error Message: {str(e)}

File: {file_path}
File Size: {file_size} bytes
File Exists: {os.path.exists(file_path)}
File Readable: {os.access(file_path, os.R_OK) if os.path.exists(file_path) else 'N/A'}

Python Version: {sys.version}
USE_EDITOR: {USE_EDITOR}

Full Traceback:
{traceback.format_exc()}

Troubleshooting Steps:
1. Verify the file is a valid GGUF file
2. Check if the file is corrupted
3. Ensure you have sufficient disk space
4. Try opening the file with another tool
5. Check the full log file for more details

Log File: {LOG_FILE}
"""
            
            messagebox.showerror("Error", error_msg + "\n\nClick OK to view detailed debug information.")
            self.show_debug_window(full_error)
    
    def toggle_mode(self):
        """Toggle between Easy and Advanced modes."""
        if self.is_easy_mode:
            # Switching to Advanced
            if not self.editor:
                messagebox.showwarning("Warning", "Please load a GGUF file first!")
                return
            
            # Save current easy mode data
            # Then switch to advanced
            try:
                metadata = self.editor.load()
                self.advanced_editor.load_metadata(metadata)
                self.show_advanced_mode()
                self.mode_btn.config(text="✨ Switch to Easy Mode")
                self.is_easy_mode = False
            except Exception as e:
                messagebox.showerror("Error", f"Failed to switch mode: {e}")
        else:
            # Switching to Easy
            if not self.editor:
                messagebox.showwarning("Warning", "Please load a GGUF file first!")
                return
            
            # Save advanced metadata, then switch to easy
            try:
                self.easy_editor.load_data(self.editor)
                self.show_easy_mode()
                self.mode_btn.config(text="🔧 Switch to Advanced Mode")
                self.is_easy_mode = True
            except Exception as e:
                messagebox.showerror("Error", f"Failed to switch mode: {e}")
    
    def show_easy_mode(self):
        """Show the easy editor panel."""
        self.advanced_editor.frame.pack_forget()
        self.easy_editor.frame.pack(fill="both", expand=True)
    
    def show_advanced_mode(self):
        """Show the advanced editor panel."""
        self.easy_editor.frame.pack_forget()
        self.advanced_editor.frame.pack(fill="both", expand=True)
    
    @log_function_call
    def save_file(self):
        """Save changes to the GGUF file."""
        logger.info("="*80)
        logger.info("FILE SAVE OPERATION: Starting")
        logger.info("="*80)
        
        if not self.editor:
            logger.warning("Save attempted but no editor instance available")
            messagebox.showwarning("Warning", "Please load a GGUF file first!")
            return
        
        logger.debug(f"Editor instance: {type(self.editor)}")
        logger.debug(f"Current file: {self.current_file}")
        logger.debug(f"Easy mode: {self.is_easy_mode}")
        
        # Check if gguf library is available before attempting save
        try:
            logger.debug("Checking for gguf library...")
            import gguf
            logger.info(f"gguf library available for saving (from: {gguf.__file__ if hasattr(gguf, '__file__') else 'unknown'})")
            logger.debug(f"gguf version: {getattr(gguf, '__version__', 'unknown')}")
        except ImportError as e:
            logger.error(f"gguf library not available: {e}")
            logger.error(f"  Import error traceback:\n{traceback.format_exc()}")
            error_msg = (
                "The 'gguf' library is required to save GGUF files.\n\n"
                "To install it, open a terminal and run:\n"
                "  pip install gguf\n\n"
                "Then restart this application and try saving again."
            )
            logger.error(f"gguf library not available: {error_msg}")
            messagebox.showerror("Missing Dependency", error_msg)
            
            # Offer to open terminal or show instructions
            response = messagebox.askyesno(
                "Install gguf Library?",
                "Would you like to see installation instructions?\n\n"
                "You can install it by running:\n"
                "pip install gguf"
            )
            if response:
                # Show detailed instructions
                import webbrowser
                import platform
                instructions = (
                    "To install the gguf library:\n\n"
                    "1. Open a terminal/command prompt\n"
                    "2. Run: pip install gguf\n"
                    "3. Wait for installation to complete\n"
                    "4. Restart this application\n\n"
                    "If you're using a virtual environment, make sure it's activated first."
                )
                messagebox.showinfo("Installation Instructions", instructions)
            return
        
        # Ask user if they want to save to current file or save as
        response = messagebox.askyesnocancel(
            "Save File",
            "Save to current file?\n\nYes = Overwrite current file\nNo = Save As (new file)"
        )
        
        if response is None:  # Cancel
            return
        
        try:
            if response:  # Save to current file
                output_path = None  # Will use current file path
            else:  # Save As
                output_path = filedialog.asksaveasfilename(
                    title="Save GGUF File As",
                    defaultextension=".gguf",
                    filetypes=[("GGUF files", "*.gguf"), ("All files", "*.*")]
                )
                if not output_path:
                    return
            
            # Prepare data for save operation
            logger.info("Preparing data for save operation...")
            save_data = {}
            if self.is_easy_mode:
                logger.debug("Using Easy mode data")
                data = self.easy_editor.get_data()
                save_data = {
                    'output_path': output_path,
                    'name': data['name'],
                    'system_prompt': data['system_prompt'],
                    'persona_prompt': data['persona_prompt'],
                    'custom_metadata': None
                }
            else:
                logger.debug("Using Advanced mode data")
                metadata = self.advanced_editor.get_metadata()
                logger.debug(f"  Total metadata keys: {len(metadata)}")
                
                # Extract common fields for save method
                name = metadata.get('general.name') or metadata.get('general.description')
                system_prompt = metadata.get('llama.system_prompt') or metadata.get('chat_template.system_prompt')
                persona_prompt = metadata.get('llama.persona_prompt') or metadata.get('chat_template.persona_prompt')
                
                # Remove fields that will be handled by save method
                custom_metadata = {k: v for k, v in metadata.items() 
                                 if k not in ('general.name', 'general.description',
                                            'llama.system_prompt', 'chat_template.system_prompt',
                                            'llama.persona_prompt', 'chat_template.persona_prompt')}
                save_data = {
                    'output_path': output_path,
                    'name': str(name) if name else None,
                    'system_prompt': str(system_prompt) if system_prompt else None,
                    'persona_prompt': str(persona_prompt) if persona_prompt else None,
                    'custom_metadata': custom_metadata if custom_metadata else None
                }
            
            # Show progress window
            saved_path = output_path if output_path else self.current_file
            progress_window = self.show_saving_progress(saved_path)
            
            # Run save operation in background thread
            def save_in_thread():
                """Run save operation in background thread."""
                try:
                    logger.info("Starting save operation in background thread...")
                    start_time = time.time()
                    
                    # Update status - Reading source
                    self.root.after(0, lambda: self.update_save_progress(
                        progress_window, "Reading source file..."
                    ))
                    
                    # Update status - Copying tensors
                    self.root.after(0, lambda: self.update_save_progress(
                        progress_window, "Copying model tensors..."
                    ))
                    
                    # Update status - Copying metadata
                    self.root.after(0, lambda: self.update_save_progress(
                        progress_window, "Copying metadata..."
                    ))
                    
                    # Update status - Writing file
                    self.root.after(0, lambda: self.update_save_progress(
                        progress_window, "Writing file to disk..."
                    ))
                    
                    # Perform save (this is the long operation)
                    self.editor.save(
                        output_path=save_data['output_path'],
                        name=save_data['name'],
                        system_prompt=save_data['system_prompt'],
                        persona_prompt=save_data['persona_prompt'],
                        custom_metadata=save_data['custom_metadata']
                    )
                    
                    elapsed = time.time() - start_time
                    logger.info(f"Save operation completed successfully in {elapsed:.3f}s")
                    
                    # Update status - Complete
                    self.root.after(0, lambda: self.update_save_progress(
                        progress_window, f"Complete! Saved in {elapsed:.1f}s"
                    ))
                    
                    # Close progress window and show success after a brief delay
                    self.root.after(500, lambda: self.close_progress(progress_window))
                    self.root.after(600, lambda: messagebox.showinfo(
                        "Success", 
                        f"File saved successfully!\n\n{saved_path}\n\nTime taken: {elapsed:.1f} seconds"
                    ))
                    
                except Exception as save_error:
                    elapsed = time.time() - start_time if 'start_time' in locals() else 0
                    logger.error(f"Save operation failed after {elapsed:.3f}s")
                    logger.error(f"  Error type: {type(save_error).__name__}")
                    logger.error(f"  Error message: {save_error}")
                    logger.error(f"  Full traceback:\n{traceback.format_exc()}")
                    
                    # Close progress window and show error
                    self.root.after(0, lambda: self.close_progress(progress_window))
                    self.root.after(100, lambda: self._show_save_error(save_error, saved_path))
            
            # Start save thread
            save_thread = threading.Thread(target=save_in_thread, daemon=True)
            save_thread.start()
            
        except Exception as e:
            # This only catches errors during data preparation, not during save
            error_type = type(e).__name__
            error_msg = str(e)
            logger.error("="*80)
            logger.error(f"PREPARATION ERROR: {error_type}")
            logger.error(f"  Error message: {error_msg}")
            logger.error(f"  Full traceback:\n{traceback.format_exc()}")
            logger.error("="*80)
            
            messagebox.showerror("Error", f"Failed to prepare save operation:\n{e}")
    
    def create_backup(self):
        """Create a backup of the current file."""
        if not self.editor:
            messagebox.showwarning("Warning", "Please load a GGUF file first!")
            return
        
        try:
            backup_path = self.editor.create_backup()
            messagebox.showinfo("Success", f"Backup created:\n{backup_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create backup:\n{e}")


def main():
    """Main entry point."""
    root = tk.Tk()
    app = GGUFEditorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

