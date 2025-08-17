# Tiemu Project

## Quick Start

1. **Clone the repository:**

   ```sh
   git clone https://github.com/YezQiu/Tiemu.git
   cd Tiemu
   ```

   Or use whatever method to get the whole thing into your computer

2. **Create and activate a virtual environment:**

   ```sh
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**

   ```sh
   pip install -r requirements.txt
   ```

4. **Download the model file:**

   - Download `Qwen3-0.6B-Q8_0.gguf` from [Hugging Face](https://huggingface.co/) or the official source.
   - Place it in the `models` folder.

5. **Run the project:**

   ```sh
   python main.py
   ```

   No lama_cpp:

   ```sh
   python main.py --disable-llm
   ```

## Notes

- The `models` folder and large model files are ignored by git.
- If you encounter issues, check that all dependencies are installed and the model file is present.
- you should still be able to run the thing without the LLM model.
- you can run **without LLM whell package**, use arg: `--disable-llm`.

## Thanks

- HeyCrab3
- AvalonRuFae
- MoonofBridge24
