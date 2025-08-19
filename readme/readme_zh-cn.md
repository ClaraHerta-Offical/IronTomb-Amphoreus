# Tiemu 项目

[配置指南(中文)](readme/Guide.md)

## 快速开始

1. **克隆仓库：**

    ```sh
    git clone https://github.com/YezQiu/Tiemu.git
    cd Tiemu
    ```

    或者使用任何其他方法将整个项目获取到您的计算机上。

2. **创建并激活虚拟环境：**

    ```sh
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3. **安装依赖：**

    ```sh
    pip install -r requirements.txt
    ```

4. **下载模型文件：**

    * 从 [Hugging Face](https://huggingface.co/) 或官方源下载 `Qwen3-0.6B-Q8_0.gguf`。
    * 将其放置在 `models` 文件夹中。

5. **运行项目：**

    ```sh
    python main.py
    ```

    不使用 lama_cpp：

    ```sh
    python main.py --disable-llm
    ```

## 注意事项

* `models` 文件夹和大型模型文件已被 Git 忽略。
* 如果您遇到问题，请检查所有依赖项是否已安装以及模型文件是否存在。
* 即使没有 LLM 模型，您也应该能够运行该项目。
* 您可以在**没有 LLM wheel 包**的情况下运行，使用参数：`--disable-llm`。
* 本项目使用 `pyclog` 作为日志记录，快速使用参见 [关于日志系统](readme/clog_zh-cn.md)，详情参阅 [Pyclog Project](https://github.com/Akanyi/pyclog)

## 交互式控制台与存档

本项目内置了一个强大的交互式控制台，允许您在模拟运行时暂停、观察甚至实时修改世界的参数。同时，您还可以随时保存和加载模拟进度。

### 交互式控制台 (Interactive Console)

在模拟运行时，随时按下 `p` 键可以暂停模拟并进入交互式控制台。在这里，你可以像一个辣卤客一样观察和干预翁法罗斯的演化。

控制台会显示提示符 `(翁法罗斯创世涡心) >`，等待您输入命令。

**可用命令列表:**

* **`c` 或 `continue`**: 恢复模拟。
* **`n` 或 `next`**: 执行下一世代（一帧），然后再次暂停。这对于逐帧观察演化非常有用。
* **`status`**: 显示世界的宏观状态，包括当前世代、种群数量、生态多样性、停滞计数等。
* **`p <实体名称|baie>`**: 打印指定实体或当前“白厄”的详细信息，包括其泰坦亲和度和命途倾向。
  * > *示例:* `p Neikos-0496`
* **`top [k]`**: 显示当前世界中评分最高的 k 个实体（默认为 5）。
  * > *示例:* `top 10`
* **`zeitgeist`**: 查看当前由公民议会决定的主流思潮及其权重。
* **`blueprint`**: 查看当前全局的演化蓝图（基础泰坦亲和度）。
* **`set <参数名> <值>`**: **核心控制功能**。动态修改一个模拟参数。
  * > *示例:* `set mutation_rate 0.5` (将突变率立刻调整为 0.5)
* **`save <文件名.json>`**: 将当前模拟状态完整地保存到一个文件中。
* **`load <文件名.json>`**: 从一个文件加载模拟状态，完全覆盖当前世界。
* **`help`**: 显示所有可用命令的帮助信息。

### 存档与读档 (Save & Load)

您可以通过两种方式来利用存档功能：

* **在模拟过程中操作**

在通过 `p` 键进入交互式控制台后，您可以使用以下命令：

* **保存当前进度:**

    ```command
    save my_world_state.json
    ```

* **加载之前的进度:**

    ```command
    load my_world_state.json
    ```

    *注意：* 加载后，当前的所有进度都将被存档覆盖。您需要输入 `c` 或 `n` 来继续运行加载后的世界。

* **从存档启动**

您可以直接在启动项目时通过命令行参数加载一个存档文件，这对于继续上次的模拟非常方便。

* **启动命令:**

    ```sh
    python main.py --load-save my_world_state.json
    ```

这将跳过所有初始化步骤，直接从 `my_world_state.json` 文件记录的状态开始模拟。

## 致谢

* HeyCrab3
* AvalonRuFae
* MoonofBridge24
* Akanyi
