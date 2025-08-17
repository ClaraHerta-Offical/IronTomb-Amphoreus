# Tiemu 项目

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

## 致谢

* HeyCrab3
* AvalonRuFae
* MoonofBridge24
