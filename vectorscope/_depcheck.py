import importlib

MODULES = [
    "numpy",
    "scipy",
    "sklearn",
    "umap",
    "fastapi",
    "uvicorn",
    "httpx",
    "pydantic",
    "pydantic_settings",
    "qdrant_client",
    "python_multipart",
    "multipart",
]

for name in MODULES:
    try:
        module = importlib.import_module(name)
        print(f"{name:20s} OK   {getattr(module, '__version__', '?')}")
    except Exception as exc:  # noqa: BLE001
        print(f"{name:20s} BRAK ({type(exc).__name__})")
