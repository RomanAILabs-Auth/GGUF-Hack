# World-Class GGUF Editor (Python)

A desktop GUI for editing **GGUF** model metadata safely and conveniently. Includes **Easy Mode** (model name + system/persona prompts) and **Advanced Mode** (full metadata tree editor), plus backups, progress dialogs for large files, and logs/debug helpers.

## Features
- 📂 Load `.gguf` files (handles large files with progress UI)
- ✨ Easy Mode
  - Edit Model Name
  - Edit System Prompt + Persona Prompt
  - Character counters
- 🔧 Advanced Mode
  - Full metadata tree view
  - Search/filter keys
  - Add / edit / delete metadata entries
- 💾 Save changes with progress UI
- 🧰 One-click timestamped backups: `*_backup_YYYYMMDD_HHMMSS.gguf`
- 🧾 Logging + “Error Debug Information” dialog (copy report / open log)

## Requirements
- Python 3
- `gguf` Python library

## Install
```bash
pip install gguf

How to run
python3 gguf-editor.py

How to Use

Click 📂 Load GGUF File

Edit in Easy Mode (quick edits) or switch to Advanced Mode (full metadata control)

(Optional) click 🧰 Create Backup

Click 💾 Save Changes

Logs & Troubleshooting

Logs are saved to:

~/.gguf_editor_logs/gguf_editor_YYYYMMDD_HHMMSS.log

If something fails, the app shows an Error Debug Information dialog with:

a copyable error report

a button to open the log file

Safety Notes

This tool edits GGUF metadata (not training weights).

Always keep backups of important model files before saving changes.

License / Disclaimer

Use at your own risk. Test edited files in your target runtime after saving.

Copyright Daniel Harding - RomanAILabs
Credits: OpenAI GPT-5.2 Thinking
