# Tiemu Project

[简体中文](readme/readme_zh-cn.md) [配置指南(Only-CHS)](readme/Guide.md)

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

## Interactive Console and Archiving

This project has a powerful built-in interactive console that allows you to pause the simulation, observe, and even modify the world's parameters in real-time. Additionally, you can save and load the simulation progress at any time.

### Interactive Console

During the simulation, you can press the `p` key at any time to pause the simulation and enter the interactive console. Here, you can observe and intervene in the evolution of Omphalos like a "La Lou ke" (辣卤客).

The console will display the prompt `(翁法罗斯创世涡心) >`, waiting for you to input commands.

**List of Available Commands:**

- **`c` or `continue`**: Resume the simulation.
- **`n` or `next`**: Execute the next generation (one frame), then pause again. This is very useful for observing evolution frame by frame.
- **`status`**: Display the macro status of the world, including the current generation, population size, ecological diversity, stagnation count, etc.
- **`p <entity_name|baie>`**: Print detailed information of the specified entity or the current "Baie" (白厄), including its Titan affinity and path tendency.
  - > *Example:* `p Neikos-0496`
- **`top [k]`**: Display the top k entities with the highest scores in the current world (default is 5).
  - > *Example:* `top 10`
- **`zeitgeist`**: View the current mainstream ideology and its weight as determined by the Citizens' Council.
- **`blueprint`**: View the current global evolutionary blueprint (base Titan affinities).
- **`set <parameter_name> <value>`**: **Core control function**. Dynamically modify a simulation parameter.
  - > *Example:* `set mutation_rate 0.5` (Immediately adjusts the mutation rate to 0.5)
- **`save <filename.json>`**: Save the current simulation state completely to a file.
- **`load <filename.json>`**: Load a simulation state from a file, completely overwriting the current world.
- **`help`**: Display help information for all available commands.

### Save & Load

You can utilize the save/load functionality in two ways:

- **During the simulation**

After entering the interactive console by pressing the `p` key, you can use the following commands:

- **Save current progress:**

    ```command
    save my_world_state.json
    ```

- **Load previous progress:**

    ```command
    load my_world_state.json
    ```

    *Note:* After loading, all current progress will be overwritten by the archive. You need to enter `c` or `n` to continue running the loaded world.

- **Start from an archive**

You can directly load an archive file via a command-line argument when launching the project. This is very convenient for continuing a previous simulation.

## Thanks

- HeyCrab3
- AvalonRuFae
- MoonofBridge24
