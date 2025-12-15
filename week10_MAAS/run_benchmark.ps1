$env:PYTHONPATH = "$PWD\MaAS"
pip install semantic-kernel anthropic google-generativeai sentence-transformers
python MaAS/maas/ext/maas/scripts/run_csqa.py
