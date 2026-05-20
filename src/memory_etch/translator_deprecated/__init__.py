import warnings

warnings.warn(
    "memory_etch.translator has been moved to memory_etch.translator_deprecated. "
    "This module is a historical reference with intentional syntax errors. "
    "It is not functional and will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)

# Backward-compat imports — these may fail due to missing files or syntax errors.
# The module is preserved for reference only.
try:
    from memory_etch.translator_deprecated.translator_pipeline import TranslatorPipeline  # noqa: F401
except (ImportError, SyntaxError):
    pass

try:
    from memory_etch.translator_deprecated.triple_extractor import TripleExtractor  # noqa: F401
except (ImportError, SyntaxError):
    pass

try:
    from memory_etch.translator_deprecated.hermes_aware_translator import HermesAwareTranslator  # noqa: F401
except (ImportError, SyntaxError):
    pass
