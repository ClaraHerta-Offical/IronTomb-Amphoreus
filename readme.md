# IronTomb (Tiemu) Project | 铁墓模拟计划

[配置指南(Only-CHS)](readme/Guide.md)

## Why do birds fly? Because they「must」fly into the sky. When the meteor of Finality descends in the Cretaceous, only the free bird can jump out of the predetermined demise.
## 鸟为什么会飞​？因为它们「必须」飞上天际。当终焉的陨星在白垩纪降下，唯有自由的鸟儿才能跳出既定的灭亡。
## Quick Start | 快速部署你的专属翁法罗斯

1. **Clone the repository | 克隆仓库:**

   ```sh
   git clone https://github.com/ClaraHerta-Offical/IronTomb-Amphoreus.git
   cd IronTomb-Amphoreus
   ```

   Or use whatever method to get the whole thing into your computer
   或者你可以随便处理，只要能把仓库里的东西完整地下载下来就行了

2. **Create and activate a virtual environment | 创建并激活虚拟环境:**

   ```sh
   python -m venv .venv
   source .venv/Scripts/activate # Windows 省略 source
   ```

3. **Install dependencies | 安装依赖:**

   ```sh
   pip install -r requirements.txt
   ```

4. **Download the model file （Can Ignore After Modify）| 下载模型（修改配置后该步骤可以忽略）:**

   - Download `Qwen3-0.6B-Q8_0.gguf` from [Hugging Face](https://huggingface.co/) or the official source. Just like [here](https://hf-mirror.com/Qwen/Qwen3-0.6B-GGUF)
   - Place it in the `models` folder.

   - 前往[Hugging Face](https://huggingface.co/)下载 `Qwen3-0.6B-Q8_0.gguf` 或者从[这里](https://hf-mirror.com/Qwen/Qwen3-0.6B-GGUF)
   - 新建一个 `models` 文件夹，把你下载到的文件扔进去

5. **Run the project | 启动:**
   ```sh
   python main.py
   ```

## Notes | 注意事项

- The `models` folder and large model files are ignored by git. | models 文件夹和 LLM 文件均需要自行准备
- If you encounter issues, check that all dependencies are installed and the model file is present. | 如遇问题请先确保依赖项以及模型均正常
- you should still be able to run the thing without the LLM model | 就像上文提到的，没有模型应该也能运行
- With parameter `--disable-llm` to run **without LLM wheel package** | 使用参数`--disable-llm`从而在**没有LLM wheel包**时运行
- If you have a MSVC (From VS) installed, build it in Developer PowerShell or follow these steps in the image | 如果你有来自VS的MSVC，请在 Developer PowerShell 下操作，否则请看下图（或许这一版的作者不想翻译它？）
<img width="1050" height="396" alt="屏幕截图 2025-08-17 155535" src="https://github.com/user-attachments/assets/875953f8-8564-40dd-9dba-3c3d3fa6b2c9" />

## Tips | 小贴士

- Change your username into LygoS and your computer name into δ-me13 to have better experience during simulating. | 试试把用户名和电脑名分别改成来古士（LygoS）和 δ-me13 ，体验更佳！
- Try to play TruE? | 也许一边模拟一边播放 TruE 会更棒！

### Interactive Console | 交互式控制台

During the simulation, you can press the `p` key at any time to pause the simulation and enter the interactive console. Here, you can observe and intervene in the evolution of Amphoreus. 模拟过程中，按p键即可暂停并且在控制台下观测翁法罗斯。（就像是辣卤客的目光？）

The console will display the prompt `(翁法罗斯创世涡心) >`, waiting for you to input commands. 模拟器将在提示符`(翁法罗斯创世涡心) >`下等待指令。

**List of Available Commands | 可用指令:**

- **`c` or `continue`**: Resume the simulation. | 继续模拟
- **`n` or `next`**: Execute the next generation (one frame), then pause again. This is very useful for observing evolution frame by frame. | 逐帧执行下一世代，有时这很有用
- **`status`**: Display the macro status of the world, including the current generation, population size, ecological diversity, stagnation count, etc. | 查看包括当前世代、种群数量、生态多样性、停滞计数在内的各项世界参数
- **`p <entity_name 实体名称|baie>`**: Print detailed information of the specified entity or the current "Baie" (Phainon 白厄), including its Titan affinity and path tendency. | 打印指定实体或当前「白厄」的详细信息，包括其泰坦亲和度和命途倾向
  - > *Example:* `p Neikos-0496`
- **`top [k]`**: Display the top k entities with the highest scores in the current world (default is 5). | 显示当前世界中评分最高的 k 个实体（默认为 5）
  - > *Example:* `top 10`
- **`zeitgeist`**: View the current mainstream ideology and its weight as determined by the Citizens' Council. | 查看当前由公民议会决定的主流思潮及其权重
- **`blueprint`**: View the current global evolutionary blueprint (base Titan affinities). | 查看当前全局的演化蓝图（基础泰坦亲和度）
- **`set <parameter_name> <value>`**: **Core control function**. Dynamically modify a simulation parameter. | **这是一个核心控制功能** 动态修改一个模拟参数
  - > *Example:* `set mutation_rate 0.5` (Immediately adjusts the mutation rate to 0.5 突变率立刻调整为 0.5)
- **`save <filename.json>`**: Save the current simulation state completely to a json file. | 存档当前世界至json文件
- **`load <filename.json>`**: Load a simulation state from a json file, completely overwriting the current world. | 从指定json文件里读档，并且彻底盖档
- **`help`**: Display help information for all available commands. | 查看这些命令的帮助信息

### Save & Load | 存档与读档

You can utilize the save/load functionality in two ways: | 存档功能有两个使用方式：

- **During the simulation 模拟时**

After entering the interactive console by pressing the `p` key, you can use the following commands: | 按p键进入控制台之后输入这些命令：

- **Save current progress 保存:**

    ```command
    save my_world_state.json
    ```

- **Load previous progress 加载:**

    ```command
    load my_world_state.json
    ```

    *Note:* After loading, all current progress will be overwritten by the archive. You need to enter `c` or `n` to continue running the loaded world.
    *注意：* 读档后当前进度会被覆盖，随后按`c`或者`n`继续

- **Start from an archive 直接从存档启动**

You can directly load an archive file via a command-line argument when launching the project. This is very convenient for continuing a previous simulation.
使用命令行参数即可加载存档，这对于继续模拟很有用！


## Thanks

- HeyCrab3
- AvalonRuFae
- MoonofBridge24
- Akanyi
