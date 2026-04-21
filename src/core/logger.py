"""
SemanticPrism: Standardized Logging Utility

Ensures granular, phase-specific auditing format across the application natively.
Isolates logging behavior gracefully without hardcoded limits.
"""
import logging
import sys
import os
from datetime import datetime

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Returns a configured logger with standard uniform formatting.
    
    Args:
        name: Name of the module initializing the logger.
        level: Minimum importance level to record.
    """
    logger = logging.getLogger(name)
    
    # Prevent duplicated handlers if initialized multiple times organically natively
    if not logger.handlers:
        logger.setLevel(level)
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        
    return logger


def save_execution_log(metrics: dict, logger_instance: logging.Logger):
    """
    Formulates and securely statically dumps the pipeline execution log accurately cleanly natively smoothly.
    """
    start_datetime = metrics.get("start_datetime", datetime.now())
    try:
        log_md = f"# Pipeline Execution Log - {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}\\n\\n"
        log_md += "## Execution Details\\n"
        log_md += f"- **Date/Time**: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}\\n"
        log_md += f"- **Duration**: {metrics.get('duration', 0.0):.2f} seconds\\n"
        log_md += f"- **Sync/Async**: {'Async' if metrics.get('use_async') else 'Sync'}\\n\\n"
        
        log_md += "## Input Data\\n"
        log_md += f"- **Count of Input Documents**: {metrics.get('doc_count', 0)}\\n"
        log_md += f"- **Document Lengths**: {metrics.get('doc_lengths', [])}\\n\\n"
        
        log_md += "## LLM Usage\\n"
        log_md += f"- **Model Name**: {metrics.get('model_name', 'Unknown')}\\n"
        log_md += f"- **Connection Protocol**: {metrics.get('connection_protocol', 'Unknown')}\\n"
        log_md += f"- **Context Windows Identified/Used**: {metrics.get('all_ctxs', [])}\\n\\n"
        
        log_md += "## Discovery Metrics\\n"
        log_md += f"- **Count of Themes Discovered**: {metrics.get('all_themes_count', 0)}\\n"
        log_md += f"- **Count of Distilled Themes**: {metrics.get('distilled_t_count', 0)}\\n\\n"
        
        log_md += "## Triple Processing\\n"
        log_md += f"- **Total Triples Extracted**: {metrics.get('raw_triples_count', 0)}\\n"
        log_md += "- **Original Unique Elements**:\\n"
        log_md += f"  - Subjects: {metrics.get('orig_subjs', 0)}\\n"
        log_md += f"  - Predicates: {metrics.get('orig_preds', 0)}\\n"
        log_md += f"  - Objects: {metrics.get('orig_objs', 0)}\\n"
        log_md += "- **Normalized Unique Elements**:\\n"
        log_md += f"  - Subjects: {metrics.get('norm_subjs', 0)}\\n"
        log_md += f"  - Predicates: {metrics.get('norm_preds', 0)}\\n"
        log_md += f"  - Objects: {metrics.get('norm_objs', 0)}\\n\\n"
        
        log_md += "## Errors\\n"
        all_errors = metrics.get('all_errors', [])
        if all_errors:
            for err in all_errors:
                log_md += f"- {err}\\n"
        else:
            log_md += "- No errors encountered.\\n"

        os.makedirs("logs", exist_ok=True)
        log_filename = f"logs/run_log_{start_datetime.strftime('%Y%m%d_%H%M%S')}.md"
        with open(log_filename, 'w', encoding='utf-8') as f:
            f.write(log_md)
        logger_instance.info(f"Execution run log effectively preserved natively into {log_filename}")
    except Exception as e:
        logger_instance.warning(f"Run logging preservation physically failed natively: {e}")

