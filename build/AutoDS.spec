# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_dynamic_libs
from PyInstaller.utils.hooks import collect_submodules

datas = [('/home/smorara/Documents/datascience_ai/autods/frontend', 'frontend'), ('/home/smorara/Documents/datascience_ai/autods/backend/geo_data', 'backend/geo_data'), ('/home/smorara/Documents/datascience_ai/autods/assets', 'assets')]
binaries = []
hiddenimports = ['h11', 'python_multipart', 'multipart', 'openpyxl', 'xlsxwriter', 'pymysql', 'psycopg2', 'sqlalchemy.dialects.sqlite', 'sqlalchemy.dialects.mysql', 'sqlalchemy.dialects.postgresql', 'webview', 'webview.platforms.qt', 'qtpy', 'PyQt6.QtWebEngineWidgets', 'PyQt6.QtWebChannel']
datas += collect_data_files('xgboost')
datas += collect_data_files('lightgbm')
datas += collect_data_files('catboost')
binaries += collect_dynamic_libs('xgboost')
binaries += collect_dynamic_libs('lightgbm')
binaries += collect_dynamic_libs('catboost')
hiddenimports += collect_submodules('uvicorn')
hiddenimports += collect_submodules('mlxtend.frequent_patterns')


a = Analysis(
    ['/home/smorara/Documents/datascience_ai/autods/desktop.py'],
    pathex=['/home/smorara/Documents/datascience_ai/autods'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'IPython', 'pytest', 'matplotlib'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='AutoDS',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='AutoDS',
)
