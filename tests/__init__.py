import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["USE_DETERMINISTIC"] = "true"
os.environ["DATABASE_URL"] = "sqlite:///:memory:"