# 关于日志系统 (.clog)

本项目使用一个名为 `pyclog` 的自定义日志系统来记录模拟的详细过程。与传统的文本日志 (`.log`) 不同，`.clog` 文件是一种**二进制、分块压缩**的格式。

**优点:**

* **高效存储**: 压缩后的日志文件占用空间更小。
* **高性能写入**: 分块写入和缓冲机制减少了对模拟主循环的性能影响。

由于 `.clog` 是二进制文件，你**不能**直接用文本编辑器打开查看。你需要使用配套的命令行工具来将其导出为可读格式。

**如何导出和查看日志:**

你可以使用 `pyclog` 模块提供的命令行工具将 `.clog` 文件转换为 `.txt` 或 `.json` 文件。

**基本命令格式:**

```sh
python -m pyclog.cli --input <你的日志文件.clog> --output <输出文件名> --format <格式>
```

**参数说明:**

* `--input` / `-i`: 指定要读取的 `.clog` 文件路径。
* `--output` / `-o`: 指定导出文件的输出路径。
* `--format` / `-f`: 导出格式，可以是 `text` (纯文本) 或 `json`。默认为 `text`。
* `--compress` / `-c`: 对输出文件使用的压缩格式，可以是 `none`, `gzip`, `zstd`。默认为 `none`。

**示例:**

假设你要将 `simulation-2025-08-20_10-30-00.clog` 转换为一个名为 `log_export.txt` 的纯文本文件：

```sh
python -m pyclog.cli --input simulation-2025-08-20_10-30-00.clog --output log_export.txt --format text
```

如果你想将其导出为 JSON 格式：

```sh
python -m pyclog.cli -i simulation-2025-08-20_10-30-00.clog -o log_export.json -f json
```
