#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility entry for Figure 2."""
from __future__ import annotations
import sys
from .make_paper_assets import main

if __name__ == "__main__":
    main(["fig2", *sys.argv[1:]])
